"""LangGraph 工作流调度。

使用 LangGraph StateGraph 编排：parse → (retrieve?) → polish ↔ audit 反思迭代。
通过 Annotated overwrite reducer 保证 draft_content / audit_result 等字段为覆盖语义，
避免不同版本 state merge 导致内容丢失。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Literal, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from .agents.audit_agent import run_audit
from .agents.parser_agent import run_parse
from .agents.polish_agent import run_polish
from .agents.retriever_agent import run_retrieve
from .config import settings
from .exceptions import DramaAgentError
from .logging_setup import get_logger
from .memory import get_session_manager
from .models import AuditResult, FinalResponse, ParsedTask, ReflectionEntry

logger = get_logger("graph")


# ============= LangGraph State =============


def _overwrite(_old: Any, new: Any) -> Any:
    """Reducer：节点返回值覆盖旧值（非 append/merge）。"""
    return new


class WorkflowGraphState(TypedDict, total=False):
    """LangGraph 共享 state。"""

    raw_input: str
    user_id: str
    session_id: Optional[str]
    session_context: str
    user_profile_text: str
    parsed_task: Annotated[Optional[ParsedTask], _overwrite]
    retrieved_materials: Annotated[List[Any], _overwrite]
    draft_content: Annotated[str, _overwrite]
    audit_result: Annotated[Optional[AuditResult], _overwrite]
    iteration_count: Annotated[int, _overwrite]
    max_iteration: int
    degrade_mode: Annotated[bool, _overwrite]
    error_info: Annotated[str, _overwrite]
    need_more_retrieval: bool
    node_failed: Annotated[str, _overwrite]


# ============= 事件发射（前端可视化用）=============


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
    """把一个 Agent 函数包装为 LangGraph 节点；异常时写入 degrade_mode，不中断工作流。"""
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
        return {
            "degrade_mode": True,
            "error_info": f"{type(e).__name__}: {e}",
            "node_failed": name,
        }


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
                passed = getattr(a, "passed", False) if not isinstance(a, dict) else a.get("passed", False)
                return f"passed={passed}, score={score}, issues={len(issues)}"
    except Exception:
        pass
    return "节点完成"


# ============= 路由决策 =============


def _route_after_parse(state: WorkflowGraphState) -> Literal["retrieve_node", "polish_node"]:
    parsed = state.get("parsed_task")
    if parsed is None:
        return "polish_node"
    needs_retrieval = (
        getattr(parsed, "needs_retrieval", True)
        if not isinstance(parsed, dict)
        else parsed.get("needs_retrieval", True)
    )
    degrade = state.get("degrade_mode", False)
    if needs_retrieval and not degrade:
        return "retrieve_node"
    return "polish_node"


def _route_after_audit(state: WorkflowGraphState) -> Literal["polish_node", "__end__"]:
    audit = state.get("audit_result")
    passed = False
    if audit is not None:
        passed = (
            getattr(audit, "passed", False)
            if not isinstance(audit, dict)
            else audit.get("passed", False)
        )
    max_iter = int(state.get("max_iteration") or 3)
    iteration_count = int(state.get("iteration_count") or 0)
    if passed or iteration_count >= max_iter:
        return END
    logger.info(f"[Graph] 第 {iteration_count} 轮审核未通过，继续重写")
    return "polish_node"


# ============= LangGraph 构建 =============


def _build_workflow_graph(ctx: Dict[str, Any]):
    """构建并编译 LangGraph 工作流（节点闭包捕获 ctx 以发射 SSE 事件）。"""

    def parse_node(state: WorkflowGraphState) -> Dict[str, Any]:
        return _safe_node(ctx, run_parse, "parse_node", dict(state))

    def retrieve_node(state: WorkflowGraphState) -> Dict[str, Any]:
        return _safe_node(ctx, run_retrieve, "retrieve_node", dict(state))

    def polish_node(state: WorkflowGraphState) -> Dict[str, Any]:
        return _safe_node(ctx, run_polish, "polish_node", dict(state))

    def audit_node(state: WorkflowGraphState) -> Dict[str, Any]:
        return _safe_node(ctx, run_audit, "audit_node", dict(state))

    builder = StateGraph(WorkflowGraphState)
    builder.add_node("parse_node", parse_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("polish_node", polish_node)
    builder.add_node("audit_node", audit_node)

    builder.add_edge(START, "parse_node")
    builder.add_conditional_edges("parse_node", _route_after_parse)
    builder.add_edge("retrieve_node", "polish_node")
    builder.add_edge("polish_node", "audit_node")
    builder.add_conditional_edges("audit_node", _route_after_audit)

    return builder.compile()


def _run_langgraph(ctx: Dict[str, Any], state: WorkflowGraphState) -> WorkflowGraphState:
    """执行 LangGraph 工作流。"""
    logger.info("[Graph] 使用 LangGraph StateGraph 调度")
    graph = _build_workflow_graph(ctx)
    return graph.invoke(state)


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

    sm = get_session_manager()
    session = sm.before_workflow(session_id, user_id, raw_input)
    context_text = session.context_summary()
    profile_text = session.profile.summary_text()
    has_context = bool(context_text)
    logger.info(
        f"[Graph] 会话 {session.session_id} 启动：user_id={user_id}, "
        f"历史消息数={len(session.messages)}, 画像_preference={session.profile.preferred_style}"
    )

    _emit(ctx, {
        "type": "workflow_start",
        "ts": t0,
        "input": raw_input[:200],
        "session_id": session.session_id,
        "has_context": has_context,
    })

    state: WorkflowGraphState = {
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

    try:
        state = _run_langgraph(ctx, state)
    except Exception as e:
        logger.exception(f"工作流异常：{e}")
        state["error_info"] = str(e)
        state["degrade_mode"] = True
        _emit(ctx, {"type": "workflow_error", "error": str(e)})

    dt_ms = (time.time() - t0) * 1000
    _emit(ctx, {
        "type": "workflow_done",
        "ts": time.time(),
        "elapsed_ms": round(dt_ms, 1),
        "session_id": session.session_id,
    })

    has_error = bool(state.get("error_info"))
    parsed_task_obj: Optional[ParsedTask] = None
    pt = state.get("parsed_task")
    if isinstance(pt, ParsedTask):
        parsed_task_obj = pt
    content = state.get("draft_content") or ""
    audit_result = state.get("audit_result")

    try:
        reflection_entry: Optional[ReflectionEntry] = None
        iteration = int(state.get("iteration_count") or 0)
        if audit_result is not None and iteration > 1:
            issues_found: List[str] = []
            try:
                issues = (
                    getattr(audit_result, "issues", [])
                    if not isinstance(audit_result, dict)
                    else audit_result.get("issues", [])
                )
                for issue in issues:
                    if isinstance(issue, dict):
                        issues_found.append(
                            f"[{issue.get('level', '')}] {issue.get('suggestion', '')[:40]}"
                        )
                    else:
                        issues_found.append(str(issue)[:80])
            except Exception:
                pass
            reflection_entry = ReflectionEntry(
                session_id=session.session_id,
                original_content=content[:300],
                revision_content=content[:300],
                audit_score_before=0.0,
                audit_score_after=float(
                    getattr(audit_result, "score", 0)
                    if not isinstance(audit_result, dict)
                    else audit_result.get("score", 0)
                ),
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


# ============= 辅助：列出已注册工具 =============


def list_tools() -> List[str]:
    from .tools import registry
    return registry.list_tools()
