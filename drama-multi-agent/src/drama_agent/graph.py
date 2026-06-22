"""LangGraph 工作流调度（手动调度版本）。

关键改进（相比 drama-multi-agent 原版）：
- 统一使用纯手动调度（run_workflow → _manual_fallback），避免 LangGraph 不同版本
  对 dict 类型 state 的 merge/replace 行为不一致导致 content 丢失；
- 节点之间通过 dict.update 合并部分字段，对 draft_content 采用"覆盖而非合并"的语义；
- degrade_mode 只有在节点显式抛出 DramaAgentError / 非预期异常时才会设置；
- 每一个节点都封装在 _safe_node 里，捕获异常并写入 state，避免单个节点失败
  导致前端拿到空内容。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from .agents.audit_agent import run_audit
from .agents.parser_agent import run_parse
from .agents.polish_agent import run_polish
from .agents.retriever_agent import run_retrieve
from .config import settings
from .exceptions import DramaAgentError
from .logging_setup import get_logger
from .memory import get_session_manager
from .models import FinalResponse, ParsedTask, ReflectionEntry

logger = get_logger("graph")


# ============= 事件发射（前端可视化用）=============


def _set_event_callback(ctx: Dict[str, Any], cb: Optional[Callable[[dict], None]]) -> None:
    ctx["event_callback"] = cb


def _emit(ctx: Dict[str, Any], event: dict) -> None:
    cb = ctx.get("event_callback")
    if cb is not None:
        try:
            cb(event)
        except Exception:
            pass


# ============= 节点包装：异常捕获 + 事件发射 =============


def _safe_node(
    ctx: Dict[str, Any],
    func: Callable[..., Optional[Dict[str, Any]]],
    name: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """把一个 Agent 函数包装为节点；异常时写入 degrade_mode，不中断工作流。"""
    _emit(ctx, {"type": "node_start", "node": name, "ts": time.time()})
    t0 = time.time()
    try:
        result = func(state) or {}
        dt_ms = (time.time() - t0) * 1000
        logger.info(f"node[{name}] ok 耗时={dt_ms:.0f}ms")
        _emit(ctx, {
            "type": "node_done",
            "node": name,
            "ts": time.time(),
            "duration_ms": round(dt_ms, 1),
            "summary": _node_output_summary(name, result),
        })
        return result
    except DramaAgentError as e:
        logger.warning(f"node[{name}] DramaAgentError: {e}")
        _emit(ctx, {"type": "node_error", "node": name, "error": str(e)})
        return {"degrade_mode": True, "error_info": str(e), "node_failed": name}
    except Exception as e:
        logger.exception(f"node[{name}] 未预期异常: {e}")
        _emit(ctx, {"type": "node_error", "node": name, "error": f"{type(e).__name__}: {e}"})
        return {"degrade_mode": True,
                "error_info": f"{type(e).__name__}: {e}",
                "node_failed": name}


def _node_output_summary(name: str, result: Dict[str, Any]) -> str:
    try:
        if name == "parse_node":
            pt = result.get("parsed_task")
            if pt is not None:
                task_type = getattr(pt, "task_type", "?") if not isinstance(pt, dict) else pt.get("task_type", "?")
                topic = getattr(pt, "topic", "") if not isinstance(pt, dict) else pt.get("topic", "")
                return f"类型={task_type}, 主题={str(topic)[:30]}"
        elif name == "retrieve_node":
            mats = result.get("retrieved_materials") or []
            return f"召回 {len(mats)} 条素材"
        elif name == "polish_node":
            content = result.get("draft_content", "")
            return f"生成内容 {len(str(content))} 字"
        elif name == "audit_node":
            a = result.get("audit_result")
            if a is not None:
                score = getattr(a, "score", 0) if not isinstance(a, dict) else a.get("score", 0)
                issues = getattr(a, "issues", []) if not isinstance(a, dict) else a.get("issues", [])
                return f"passed={getattr(a, 'passed', False) if not isinstance(a, dict) else a.get('passed', False)}, score={score}, issues={len(issues)}"
    except Exception:
        pass
    return "节点完成"


# ============= 路由决策 =============


def _route_after_parse(state: Dict[str, Any]) -> str:
    parsed = state.get("parsed_task")
    if parsed is None:
        return "polish"
    needs_retrieval = getattr(parsed, "needs_retrieval", True) if not isinstance(parsed, dict) else parsed.get("needs_retrieval", True)
    degrade = state.get("degrade_mode", False)
    if needs_retrieval and not degrade:
        return "retrieve"
    return "polish"


# ============= 对外主入口 =============


def run_workflow(raw_input: str, user_id: str = "guest",
                 session_id: Optional[str] = None) -> FinalResponse:
    """阻塞式：运行完整工作流，返回 FinalResponse。"""
    return run_workflow_with_events(raw_input, user_id, None, session_id)


def run_workflow_with_events(
    raw_input: str,
    user_id: str = "guest",
    event_callback: Optional[Callable[[dict], None]] = None,
    session_id: Optional[str] = None,
) -> FinalResponse:
    """流式：运行工作流，通过 event_callback 逐事件通知调用方（前端 SSE）。"""
    t0 = time.time()
    ctx: Dict[str, Any] = {"event_callback": event_callback}

    # 记忆模块：读取会话
    sm = get_session_manager()
    session = sm.before_workflow(session_id, user_id, raw_input)
    context_text = session.context_summary()
    profile_text = session.profile.summary_text()
    has_context = bool(context_text)
    logger.info(
        f"[Graph] 会话 {session.session_id} 启动：user_id={user_id}, "
        f"历史消息数={len(session.messages)}, 画像_preference={session.profile.preferred_style}"
    )

    _emit(ctx, {"type": "workflow_start", "ts": t0, "input": raw_input[:200],
                "session_id": session.session_id, "has_context": has_context})

    # 初始化 state
    state: Dict[str, Any] = {
        "raw_input": raw_input,
        "user_id": user_id,
        "session_id": session.session_id,
        "session_context": context_text,
        "user_profile_text": profile_text,
        "parsed_task": None,
        "retrieved_materials": [],
        "draft_content": "",
        "audit_result": None,
        "iteration_count": 0,
        "max_iteration": int(settings.audit_max_iteration or 3),
        "degrade_mode": False,
        "error_info": "",
        "need_more_retrieval": False,
        "node_failed": "",
    }

    # 手动调度
    try:
        state = _manual_fallback(ctx, state)
    except Exception as e:
        logger.exception(f"工作流异常：{e}")
        state["error_info"] = str(e)
        state["degrade_mode"] = True
        _emit(ctx, {"type": "workflow_error", "error": str(e)})

    dt_ms = (time.time() - t0) * 1000
    _emit(ctx, {"type": "workflow_done", "ts": time.time(),
                "elapsed_ms": round(dt_ms, 1),
                "session_id": session.session_id})

    # 构造响应
    has_error = bool(state.get("error_info"))
    parsed_task_obj: Optional[ParsedTask] = None
    pt = state.get("parsed_task")
    if isinstance(pt, ParsedTask):
        parsed_task_obj = pt
    content = state.get("draft_content") or ""
    audit_result = state.get("audit_result")

    # 写入反思日志 & 持久化会话
    try:
        reflection_entry: Optional[ReflectionEntry] = None
        iteration = int(state.get("iteration_count") or 0)
        if audit_result is not None and iteration > 1:
            issues_found: List[str] = []
            try:
                issues = getattr(audit_result, "issues", []) if not isinstance(audit_result, dict) else audit_result.get("issues", [])
                for issue in issues:
                    if isinstance(issue, dict):
                        issues_found.append(f"[{issue.get('level','')}] {issue.get('suggestion','')[:40]}")
                    else:
                        issues_found.append(str(issue)[:80])
            except Exception:
                pass
            reflection_entry = ReflectionEntry(
                session_id=session.session_id,
                original_content=content[:300],
                revision_content=content[:300],
                audit_score_before=0.0,
                audit_score_after=float(getattr(audit_result, "score", 0) if not isinstance(audit_result, dict) else audit_result.get("score", 0)),
                issues_found=issues_found,
                iteration=iteration,
            )

        sm.after_workflow(
            session=session,
            content=content,
            audit_result=audit_result,
            parsed_task=parsed_task_obj,
            reflection_entry=reflection_entry,
        )
        logger.info(f"[Memory] 会话已持久化：reflection={reflection_entry is not None}")
    except Exception as e:
        logger.warning(f"[Memory] 会话持久化失败：{e}")

    return FinalResponse(
        success=bool(content) and not has_error,
        task_type=(
            getattr(parsed_task_obj, "task_type", None)
            if parsed_task_obj is not None
            else None
        ),
        content=content,
        audit_result=audit_result,
        iteration_count=int(state.get("iteration_count") or 0),
        degrade_mode=bool(state.get("degrade_mode")),
        error=state.get("error_info") or None,
        elapsed_ms=dt_ms,
        session_id=session.session_id,
        has_context=has_context,
        user_profile_summary=profile_text if profile_text else None,
    )


# ============= 手动调度路径 =============


def _manual_fallback(ctx: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """纯手动调度：parse → (retrieve?) → polish → audit → (iter polish? )。

    关键：节点输出通过 dict.update 合并，但 draft_content / audit_result / parsed_task
    等字段必须是"覆盖"语义 —— 这里直接用 update 即可（因为每节点只会写自己负责的字段）。
    """
    logger.info("[Graph] 使用手动调度路径（不依赖 LangGraph）")
    new_state = dict(state)

    # 1. parse
    parse_out = _safe_node(ctx, run_parse, "parse_node", new_state)
    for k, v in (parse_out or {}).items():
        new_state[k] = v

    # 2. 可选 retrieve
    if _route_after_parse(new_state) == "retrieve":
        retrieve_out = _safe_node(ctx, run_retrieve, "retrieve_node", new_state)
        # retrieved_materials 保持覆盖（单次检索结果）
        for k, v in (retrieve_out or {}).items():
            new_state[k] = v

    # 3. polish → audit 迭代
    max_iter = int(new_state.get("max_iteration") or 3)
    for i in range(max_iter):
        polish_out = _safe_node(ctx, run_polish, "polish_node", new_state)
        for k, v in (polish_out or {}).items():
            new_state[k] = v

        audit_out = _safe_node(ctx, run_audit, "audit_node", new_state)
        for k, v in (audit_out or {}).items():
            new_state[k] = v

        audit = new_state.get("audit_result")
        passed = getattr(audit, "passed", False) if audit is not None and not isinstance(audit, dict) else (audit or {}).get("passed", False) if isinstance(audit, dict) else False
        # 第一次迭代若通过或达到上限，退出
        if passed or (i + 1) >= max_iter:
            break
        # 若 audit 未通过，iteration_count 会在 run_audit 里自增；继续下一轮 polish
        logger.info(f"[Graph] 第 {i+1} 轮审核未通过，继续重写")

    return new_state


# ============= 辅助：列出已注册工具 =============


def list_tools() -> List[str]:
    from .tools import registry
    return registry.list_tools()
