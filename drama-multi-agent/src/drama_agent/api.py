"""
FastAPI 服务封装：
- GET  /                       → 前端页面（frontend/index.html）
- GET  /health                 → 健康检查
- GET  /v1/tools               → 已注册的 MCP 工具列表
- POST /v1/generate            → 同步生成（默认阻塞式）
- POST /v1/stream              → SSE 流式生成（前端可视化用）
- POST /v1/async/generate      → 异步提交（返回 task_id）
- GET  /v1/async/{task_id}     → 查异步任务状态和结果
- GET  /docs                   → Swagger

# 记忆模块新增接口：
- GET  /v1/sessions            → 列出当前用户的所有会话（含画像摘要）
- GET  /v1/sessions/{session_id} → 获取单个会话详情（历史消息 + 反思日志）
- DELETE /v1/sessions/{session_id} → 删除一个会话
- POST /v1/sessions/{session_id}/writeback → 把高分内容回写到知识库

安全 / 并发：
- async 任务仓库用 threading.Lock + 最大保留 MAX_ASYNC_TASKS，避免内存泄漏
- 所有 POST 接口做参数校验（Pydantic BaseModel）
- 不向外暴露 settings 中的敏感字段
"""
from __future__ import annotations

import asyncio
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
from .tools.vector_retriever import get_vector_store

setup_logging()

app = FastAPI(
    title="短剧多智能体内容生产系统（含记忆模块）",
    version="2.0.0",
    description="基于 LangGraph 的短剧内容整理 / 文案生成 / 答疑 / 合规审核系统（支持会话记忆、用户画像、反思日志、内容回写）",
)

MAX_ASYNC_TASKS = 200
_async_tasks_lock = threading.Lock()
_async_tasks: Dict[str, Dict[str, Any]] = {}

FRONTEND_DIR = PROJECT_ROOT / "frontend"


# =============================================================================
# 请求 / 响应模型
# =============================================================================


class GenerateRequest(BaseModel):
    raw_input: str = Field(..., min_length=1, max_length=10_000, description="用户原始输入（最多 10k 字）")
    user_id: str = Field("guest", max_length=128, description="用户标识（可选）")
    session_id: Optional[str] = Field(default=None, max_length=128, description="会话 ID（不传会自动新建）")


class GenerateResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "ok"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WritebackRequest(BaseModel):
    title: str = Field(..., description="素材标题（建议用短剧剧名 / 场景名）")
    category: str = Field("剧本", description="素材分类：剧本 / 文案 / 人设 / 规则")
    content: Optional[str] = Field(default=None, description="要写入的正文（不传则从 session 中取 draft_content）")
    score_threshold: float = Field(0.85, ge=0.0, le=1.0, description="只有审核分 >= 阈值才允许回写")


# =============================================================================
# 路由
# =============================================================================


@app.get("/", response_class=HTMLResponse)
def root():
    index_html = FRONTEND_DIR / "index.html"
    if not index_html.exists():
        return HTMLResponse(
            "<h1>服务运行中（记忆模块 v2.0）</h1>"
            "<p>前端资源未找到；请将 frontend/index.html 放入项目根目录。</p>"
            "<p>API 文档：<a href='/docs'>/docs</a></p>",
            status_code=200,
        )
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict[str, Any]:
    start_ts = getattr(app, "_start_time", None)
    return {
        "status": "ok",
        "version": "2.0.0",
        "uptime_seconds": int(time.time() - start_ts) if start_ts is not None else 0,
        "async_pending": sum(1 for t in _async_tasks.values() if t.get("status") == "pending"),
        "memory_module": True,
    }


@app.get("/v1/tools")
def get_tools() -> Dict[str, List[str]]:
    return {"tools": list_tools()}


# ---------------------------------------------------------------------------
# 同步 / 异步 / 流式生成
# ---------------------------------------------------------------------------


@app.post("/v1/generate")
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        resp: FinalResponse = run_workflow(req.raw_input, req.user_id, req.session_id)
        return GenerateResponse(status="ok", data=resp.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/stream")
async def stream_generate(req: GenerateRequest):
    """Server-Sent Events 流式输出。"""
    loop = asyncio.get_event_loop()
    event_queue: asyncio.Queue[dict] = asyncio.Queue()

    def _sync_callback(event: dict):
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except Exception:
            pass

    def _run_sync():
        try:
            resp = run_workflow_with_events(
                req.raw_input, req.user_id,
                event_callback=_sync_callback, session_id=req.session_id
            )
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "final", "data": resp.model_dump()},
            )
        except Exception as e:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "error", "message": str(e)},
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "_done"})

    thread = threading.Thread(target=_run_sync, daemon=True)
    thread.start()

    async def _sse_generator():
        yield _format_sse("start", {"input": req.raw_input[:200],
                                     "session_id": req.session_id or "auto"})
        while True:
            event = await event_queue.get()
            if event.get("type") == "_done":
                break
            yield _format_sse(event["type"], {k: v for k, v in event.items() if k != "type"})

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(typ: str, payload: dict) -> str:
    merged: Dict[str, Any] = {"type": typ}
    merged.update(payload)
    data = json.dumps(merged, ensure_ascii=False)
    return f"event: {typ}\ndata: {data}\n\n"


@app.post("/v1/async/generate")
def async_generate(req: GenerateRequest) -> GenerateResponse:
    """提交异步任务；通过 /v1/async/{task_id} 查询结果。"""
    task_id = uuid.uuid4().hex
    with _async_tasks_lock:
        _async_tasks[task_id] = {
            "status": "pending",
            "created_at": time.time(),
            "session_id": req.session_id,
            "user_id": req.user_id,
        }
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


# ---------------------------------------------------------------------------
# 记忆模块接口：会话管理
# ---------------------------------------------------------------------------


@app.get("/v1/sessions")
def list_sessions(user_id: Optional[str] = Query(default=None, description="可选：按 user_id 过滤")) -> Dict[str, Any]:
    """列出所有会话（摘要）。"""
    sm = get_session_manager()
    sessions = sm.list_sessions(user_id=user_id)
    return {"total": len(sessions), "sessions": sessions}


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """获取单个会话详情（完整消息历史 + 画像 + 反思日志）。"""
    sm = get_session_manager()
    session = sm.get_or_create(session_id, "guest")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_ts": session.created_ts,
        "updated_ts": session.updated_ts,
        "messages": [
            {"role": m.role, "content": m.content[:500], "ts": m.ts} for m in session.messages
        ],
        "profile": session.profile.model_dump(),
        "reflections": [r.model_dump() for r in session.reflections],
    }


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    """删除一个会话。"""
    sm = get_session_manager()
    existed = sm.delete(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    return {"status": "ok", "deleted": session_id}


# ---------------------------------------------------------------------------
# 记忆模块接口：高分内容回写知识库
# ---------------------------------------------------------------------------


@app.post("/v1/sessions/{session_id}/writeback")
def writeback_to_knowledge(session_id: str, req: WritebackRequest) -> Dict[str, Any]:
    """
    把审核通过的高质量内容回写到向量知识库：
    1. 检查该会话最近一次生成的审核分数是否 >= score_threshold
    2. 通过则把内容添加到向量索引，作为后续检索的素材
    """
    sm = get_session_manager()
    session = sm.get_or_create(session_id, "guest")

    # 从会话中取最近的 assistant 消息作为内容（或直接用请求体 content）
    content_to_write: str = req.content or ""
    if not content_to_write:
        for msg in reversed(session.messages):
            if msg.role == "assistant":
                content_to_write = msg.content
                break

    if not content_to_write or len(content_to_write.strip()) < 50:
        raise HTTPException(status_code=400, detail="没有足够的内容用于回写（至少 50 字）")

    # 阈值检查：如果有审核结果就检查分数；否则允许显式调用方自行判断
    recent_audit_score: Optional[float] = None
    passed: bool = True
    if session.reflections:
        latest_refl = session.reflections[-1]
        recent_audit_score = latest_refl.audit_score_after
        passed = recent_audit_score >= req.score_threshold

    if not passed:
        raise HTTPException(
            status_code=400,
            detail=f"审核分 {recent_audit_score} < 阈值 {req.score_threshold}，不允许回写",
        )

    # 回写到 FAISS
    try:
        vs = get_vector_store()
        vs.add_documents([{
            "title": req.title,
            "content": content_to_write,
            "category": req.category,
        }])
        vs.save()
        return {
            "status": "ok",
            "message": f"已将《{req.title}》回写到知识库",
            "content_length": len(content_to_write),
            "category": req.category,
            "audit_score": recent_audit_score,
            "total_docs": vs.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回写失败：{e}")


# ---------------------------------------------------------------------------
# 挂载静态文件 / 记录启动时间
# ---------------------------------------------------------------------------


if FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app._start_time = time.time()  # type: ignore[attr-defined]
