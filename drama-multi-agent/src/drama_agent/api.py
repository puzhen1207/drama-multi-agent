"""FastAPI 服务封装：
- GET  /                      → 前端页面（frontend/index.html）
- GET  /health                → 健康检查
- GET  /v1/tools              → 已注册的 MCP 工具列表
- POST /v1/generate           → 同步生成（阻塞）
- POST /v1/stream             → SSE 流式生成（前端可视化用）
- POST /v1/async/generate     → 异步提交（返回 task_id）
- GET  /v1/async/{task_id}    → 查询异步任务状态/结果

记忆模块：
- GET  /v1/sessions            → 列出用户所有会话
- GET  /v1/sessions/{session_id} → 获取会话详情
- DELETE /v1/sessions/{session_id} → 删除会话
- POST /v1/sessions/{session_id}/writeback → 把高分内容回写知识库

SSE 关键改进（相比 drama-multi-agent 原版）：
- 每个事件严格使用 "event: xxx\ndata: {...}\n\n" 格式；
- data 部分通过 json.dumps(..., ensure_ascii=False) 编码；
- 通过 StreamingResponse 以 UTF-8 流式输出，避免中文被 unicode-escape；
- 返回头显式设置 Cache-Control / X-Accel-Buffering: no，避免 Nginx 层缓存。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, settings
from .graph import list_tools, run_workflow, run_workflow_with_events
from .logging_setup import setup_logging
from .memory import get_session_manager
from .models import FinalResponse
from .tools.vector_retriever import ensure_builtin_knowledge, get_vector_store

setup_logging()

import logging  # noqa: E402
logger = logging.getLogger("drama_agent.api")


app = FastAPI(
    title="短剧多智能体内容生产系统（drama-multi-agent1）",
    version="2.1.0",
    description="基于手动调度的多 Agent 系统：解析 → 检索 → 润色 → 合规审核（支持会话记忆、用户画像、反思日志）",
)


# ============= 启动时自动构建基础知识库 =============


@app.on_event("startup")
def _on_startup() -> None:
    try:
        ensure_builtin_knowledge()
    except Exception as e:
        logger.warning(f"[API] 启动时构建知识库失败：{e}")


# ============= 请求 / 响应模型 =============


class GenerateRequest(BaseModel):
    raw_input: str = Field(..., min_length=1, max_length=10_000,
                           description="用户原始输入（最多 10k 字）")
    user_id: str = Field("guest", max_length=128, description="用户标识（可选）")
    session_id: Optional[str] = Field(default=None, max_length=128, description="会话 ID（不传则自动新建）")


class GenerateResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WritebackRequest(BaseModel):
    title: str = Field(..., description="素材标题（建议使用剧名/场景名）")
    category: str = Field("剧本", description="素材分类：剧本/文案/人设/规则")
    content: Optional[str] = Field(default=None, description="正文（不传则从会话中读取 draft_content）")
    score_threshold: float = Field(0.85, ge=0.0, le=1.0, description="只有审核分 >= 阈值才允许回写")


# ============= 路由：基础 =============


@app.get("/", response_class=HTMLResponse)
def root():
    index_html = PROJECT_ROOT / "frontend" / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            "<h1>服务运行中（drama-multi-agent1）</h1>"
            "<p>前端资源未找到；请把 frontend/index.html 放入项目根目录。</p>"
            "<p>API 文档：<a href='/docs'>/docs</a></p>",
            status_code=200,
        )
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.1.0",
        "uptime_seconds": int(time.time() - getattr(app, "_start_time", time.time())),
        "memory_module": True,
    }


@app.get("/v1/tools")
def get_tools() -> Dict[str, List[str]]:
    return {"tools": list_tools()}


# ============= 路由：同步 =============


@app.post("/v1/generate")
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        resp: FinalResponse = run_workflow(req.raw_input, req.user_id, req.session_id)
        return GenerateResponse(status="ok", data=resp.model_dump())
    except Exception as e:
        logger.exception(f"[API] /v1/generate 异常：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= 路由：SSE 流式 =============


@app.post("/v1/stream")
async def stream_generate(req: GenerateRequest):
    """Server-Sent Events 流式输出。

    事件类型：
    - start: 工作流开始（含输入、session_id）
    - node_start: 节点开始
    - node_done: 节点结束（含 duration_ms、summary）
    - node_error: 节点异常
    - workflow_done: 工作流结束
    - final: 最终输出（含 content、audit_result）
    - error: 全局异常
    """

    loop_shim: Dict[str, Any] = {"queue": []}
    done_flag: Dict[str, bool] = {"value": False}

    def _sync_callback(event: dict):
        loop_shim["queue"].append(event)

    def _run_sync():
        try:
            resp = run_workflow_with_events(
                req.raw_input, req.user_id,
                event_callback=_sync_callback, session_id=req.session_id,
            )
            loop_shim["queue"].append({"type": "final", "data": resp.model_dump()})
        except Exception as e:
            logger.exception(f"[API] /v1/stream 工作流异常：{e}")
            loop_shim["queue"].append({"type": "error", "message": str(e)})
        finally:
            done_flag["value"] = True

    thread = threading.Thread(target=_run_sync, daemon=True)
    thread.start()

    async def _sse_generator():
        # 起始事件（可选：让前端立刻知道已连上）
        yield _format_sse("start", {
            "input": req.raw_input[:200],
            "session_id": req.session_id or "auto",
        })
        # 轮询队列：一边读一边 flush
        while True:
            if loop_shim["queue"]:
                ev = loop_shim["queue"].pop(0)
                # 允许节点直接写 dict 以外的类型
                if not isinstance(ev, dict):
                    continue
                yield _format_sse(ev.get("type", "message"),
                                  {k: v for k, v in ev.items() if k != "type"})
                continue
            if done_flag["value"]:
                break
            # 让出 CPU
            import asyncio
            await asyncio.sleep(0.05)
        # 结束标记
        yield _format_sse("workflow_complete", {"status": "ok"})

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Accel-Charset": "utf-8",
        },
    )


def _format_sse(typ: str, payload: Dict[str, Any]) -> bytes:
    """把一个事件编码成 SSE 字节流（UTF-8，不做 unicode-escape）。"""
    merged = {"type": typ}
    merged.update(payload)
    try:
        data = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # 兜底：如果 payload 里有不可 JSON 序列化的对象（例如 pydantic 实例）
        def _default(o: Any) -> Any:
            if hasattr(o, "model_dump"):
                return o.model_dump()
            if hasattr(o, "dict"):
                return o.dict()
            return str(o)
        data = json.dumps(merged, ensure_ascii=False, separators=(",", ":"), default=_default)
    line = f"event: {typ}\ndata: {data}\n\n"
    return line.encode("utf-8")


# ============= 路由：异步任务 =============


MAX_ASYNC_TASKS = 200
_async_tasks_lock = threading.Lock()
_async_tasks: Dict[str, Dict[str, Any]] = {}


@app.post("/v1/async/generate")
def async_generate(req: GenerateRequest) -> GenerateResponse:
    task_id = uuid.uuid4().hex
    with _async_tasks_lock:
        _async_tasks[task_id] = {
            "status": "pending",
            "created_at": time.time(),
            "session_id": req.session_id,
            "user_id": req.user_id,
        }
        # 淘汰老任务
        if len(_async_tasks) > MAX_ASYNC_TASKS:
            non_pending = sorted(
                [(k, v) for k, v in _async_tasks.items() if v.get("status") != "pending"],
                key=lambda kv: kv[1].get("created_at", 0),
            )
            to_remove = len(_async_tasks) - MAX_ASYNC_TASKS
            for k, _ in non_pending[:to_remove]:
                _async_tasks.pop(k, None)

    def _run():
        try:
            resp = run_workflow(req.raw_input, req.user_id, req.session_id)
            with _async_tasks_lock:
                _async_tasks[task_id] = {
                    "status": "ok",
                    "result": resp.model_dump(),
                    "created_at": time.time(),
                }
        except Exception as e:
            with _async_tasks_lock:
                _async_tasks[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "created_at": time.time(),
                }

    threading.Thread(target=_run, daemon=True).start()
    return GenerateResponse(task_id=task_id, status="pending")


@app.get("/v1/async/{task_id}")
def get_async_status(task_id: str) -> GenerateResponse:
    with _async_tasks_lock:
        task = _async_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return GenerateResponse(
        task_id=task_id,
        status=task["status"],
        data=task.get("result"),
        error=task.get("error"),
    )


# ============= 记忆模块：会话管理 =============


@app.get("/v1/sessions")
def list_sessions(user_id: Optional[str] = Query(default=None, description="可选：按 user_id 过滤")) -> Dict[str, Any]:
    sm = get_session_manager()
    sessions = sm.list_sessions(user_id=user_id)
    return {"total": len(sessions), "sessions": sessions}


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    sm = get_session_manager()
    session = sm.get_or_create(session_id, "guest")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_ts": session.created_ts,
        "updated_ts": session.updated_ts,
        "messages": [
            {"role": m.role, "content": m.content[:500], "ts": m.ts}
            for m in session.messages
        ],
        "profile": session.profile.model_dump() if hasattr(session.profile, "model_dump") else session.profile.__dict__,
        "reflections": [r.model_dump() for r in session.reflections] if hasattr(session, "reflections") else [],
    }


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    sm = get_session_manager()
    existed = sm.delete(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return {"status": "ok", "deleted": session_id}


# ============= 记忆模块：高分内容回写 =============


@app.post("/v1/sessions/{session_id}/writeback")
def writeback_to_knowledge(session_id: str, req: WritebackRequest) -> Dict[str, Any]:
    sm = get_session_manager()
    session = sm.get_or_create(session_id, "guest")

    content_to_write: str = req.content or ""
    if not content_to_write:
        for msg in reversed(session.messages):
            if msg.role == "assistant":
                content_to_write = msg.content
                break
    if not content_to_write or len(content_to_write.strip()) < 50:
        raise HTTPException(status_code=400,
                            detail="没有足够的内容用于回写（至少 50 字）")

    # 审核分数校验
    recent_audit_score: Optional[float] = None
    passed: bool = True
    if session.reflections:
        latest = session.reflections[-1]
        if hasattr(latest, "audit_score_after"):
            recent_audit_score = float(latest.audit_score_after)
        if recent_audit_score is not None:
            passed = recent_audit_score >= req.score_threshold

    if not passed:
        raise HTTPException(
            status_code=400,
            detail=f"审核分 {recent_audit_score} 低于阈值 {req.score_threshold}，不允许回写",
        )

    try:
        vs = get_vector_store()
        vs.add_documents([{
            "title": req.title,
            "category": req.category,
            "content": content_to_write,
        }])
        vs.save()
        return {"status": "ok", "title": req.title, "category": req.category,
                "chars": len(content_to_write)}
    except Exception as e:
        logger.exception(f"[API] 回写知识库失败：{e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= 启动标记 =============


app._start_time = time.time()  # type: ignore[attr-defined]
