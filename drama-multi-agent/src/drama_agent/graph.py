"""
LangGraph 工作流调度：
  - start -> parse_node -> (需要检索?) retrieve_node? -> polish_node -> audit_node
  - audit 不通过且迭代<上限 -> 回流 polish 实现反思迭代
  - 任何节点异常 -> degrade_mode=True 并尝试最小功能输出

并发安全：
  - _graph_lock 控制 LangGraph 实例的访问（LangGraph 非线程安全）
  - event_queue 通过 thread-local 传递到节点（不依赖 state dict 序列化）

记忆模块（新增）：
  - 工作流开始前：从 SessionManager 读取会话上下文和用户画像，注入 state
  - 工作流结束后：写入反思日志（记录 audit→polish 的修改轨迹）、更新用户画像
"""
from __future__ import annotations

import threading
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
from .models import FinalResponse, ReflectionEntry

logger = get_logger("graph")

_graph = None
_graph_lock = threading.Lock()

# thread-local：让 _safe_node 在子线程（或同线程）里拿到 event_callback
_tls = threading.local()


# =============================================================================
# 事件发射（前端可视化）
# =============================================================================


def _set_event_callback(cb: Optional[Callable[[dict], None]]):
    """在当前线程注册事件回调。"""
    _tls.callback = cb


def _clear_event_callback():
    try:
        del _tls.callback
    except AttributeError:
        pass


def _emit(event: dict):
    """发出事件（只有注册了回调时才会触发）。"""
    cb = getattr(_tls, "callback", None)
    if cb is not None:
        try:
            cb(event)
        except Exception:
            pass


# =============================================================================
# 节点包装：异常捕获 + 事件发射
# =============================================================================


def _safe_node(func: Callable[..., Optional[Dict[str, Any]]], name: str) -> Callable[[dict], dict]:
    """把一个 Agent 函数包装为节点；异常时返回 degrade_mode=True。"""

    def _run(state: dict) -> dict:
        _emit({"type": "node_start", "node": name, "ts": time.time()})
        t0 = time.time()
        try:
            result = func(state) or {}
            dt_ms = (time.time() - t0) * 1000
            logger.info(f"node[{name}] ok 耗时={dt_ms:.0f}ms")
            _emit({
                "type": "node_done",
                "node": name,
                "ts": time.time(),
                "duration_ms": round(dt_ms, 1),
                "summary": _node_output_summary(name, result),
            })
            return result
        except DramaAgentError as e:
            logger.warning(f"node[{name}] DramaAgentError: {e}")
            _emit({"type": "node_error", "node": name, "error": str(e)})
            return {"degrade_mode": True, "error_info": str(e), "node_failed": name}
        except Exception as e:
            logger.exception(f"node[{name}] 未预期异常: {e}")
            _emit({"type": "node_error", "node": name, "error": f"{type(e).__name__}: {e}"})
            return {"degrade_mode": True, "error_info": f"{type(e).__name__}: {e}", "node_failed": name}

    return _run


def _node_output_summary(name: str, result: dict) -> str:
    try:
        if name == "parse_node":
            pt = result.get("parsed_task")
            if pt is not None:
                return f"类型={getattr(pt, 'task_type', '?')}, 主题={getattr(pt, 'topic', '')[:30]}"
        elif name == "retrieve_node":
            mats = result.get("retrieved_materials") or []
            titles: List[str] = []
            for m in mats[:2]:
                t = getattr(m, "title", None)
                if isinstance(m, dict):
                    t = m.get("title")
                titles.append(str(t)[:30] if t else "untitled")
            return f"召回 {len(mats)} 条：" + ", ".join(titles)
        elif name == "polish_node":
            return f"生成内容 {len(result.get('draft_content', ''))} 字"
        elif name == "audit_node":
            a = result.get("audit_result")
            if a is not None:
                score = getattr(a, "score", 0)
                issues = getattr(a, "issues", [])
                return f"passed={getattr(a, 'passed', False)}, score={score}, issues={len(issues)}"
    except Exception:
        pass
    return "节点完成"


# =============================================================================
# LangGraph 条件路由
# =============================================================================


def _route_after_parse(state: dict) -> str:
    parsed = state.get("parsed_task")
    if parsed is None:
        return "polish"
    needs_retrieval = getattr(parsed, "needs_retrieval", True)
    degrade = state.get("degrade_mode", False)
    if needs_retrieval and not degrade:
        return "retrieve"
    return "polish"


def _route_after_polish(state: dict) -> str:
    if state.get("need_more_retrieval") and (state.get("iteration_count") or 0) < 1:
        return "retrieve"
    return "audit"


def _route_after_audit(state: dict) -> str:
    audit = state.get("audit_result")
    if audit is None:
        return "end"
    iteration = state.get("iteration_count") or 0
    max_iter = state.get("max_iteration") or settings.audit_max_iteration
    if not audit.passed and iteration < max_iter:
        return "polish"
    return "end"


# =============================================================================
# LangGraph 构建
# =============================================================================


def _build_graph():
    """在全局锁内构建一次 LangGraph StateGraph。"""
    try:
        from langgraph.graph import StateGraph
        graph_builder = StateGraph(dict)
    except Exception as e:
        logger.warning(f"LangGraph 不可用（{e}），切换为手动调度器")
        return None

    graph_builder.add_node("parse_node", _safe_node(run_parse, "parse_node"))
    graph_builder.add_node("retrieve_node", _safe_node(run_retrieve, "retrieve_node"))
    graph_builder.add_node("polish_node", _safe_node(run_polish, "polish_node"))
    graph_builder.add_node("audit_node", _safe_node(run_audit, "audit_node"))
    graph_builder.set_entry_point("parse_node")
    graph_builder.add_conditional_edges("parse_node", _route_after_parse,
                                       {"retrieve": "retrieve_node", "polish": "polish_node"})
    graph_builder.add_edge("retrieve_node", "polish_node")
    graph_builder.add_conditional_edges("polish_node", _route_after_polish,
                                       {"retrieve": "retrieve_node", "audit": "audit_node"})
    graph_builder.add_conditional_edges("audit_node", _route_after_audit,
                                       {"polish": "polish_node", "end": "__end__"})
    return graph_builder.compile()


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = _build_graph()
    return _graph


# =============================================================================
# 对外主入口
# =============================================================================


def run_workflow(raw_input: str, user_id: str = "guest", session_id: Optional[str] = None) -> FinalResponse:
    return run_workflow_with_events(raw_input, user_id, None, session_id)


def run_workflow_with_events(
    raw_input: str,
    user_id: str = "guest",
    event_callback: Optional[Callable[[dict], None]] = None,
    session_id: Optional[str] = None,
) -> FinalResponse:
    """
    主入口：执行完整工作流。

    记忆模块集成点：
    1. 工作流开始前：从 SessionManager 读取会话上下文和用户画像 → 注入 state
    2. 工作流结束后：把内容、审核结果、迭代次数写回会话 → 写入反思日志
    """
    t0 = time.time()
    if event_callback is not None:
        _set_event_callback(event_callback)

    # -------- 记忆模块：读取会话 --------
    sm = get_session_manager()
    session = sm.before_workflow(session_id, user_id, raw_input)
    context_text = session.context_summary()
    profile_text = session.profile.summary_text()
    has_context = bool(context_text)
    logger.info(
        f"[Graph] 会话 {session.session_id} 启动：user_id={user_id}, "
        f"历史消息数={len(session.messages)}, 画像_preference={session.profile.preferred_style}"
    )

    try:
        _emit({"type": "workflow_start", "ts": t0, "input": raw_input[:200],
               "session_id": session.session_id, "has_context": has_context})

        initial_state: Dict[str, Any] = {
            "raw_input": raw_input,
            "user_id": user_id,
            "session_id": session.session_id,
            "session_context": context_text,          # 注入：最近几轮对话摘要
            "user_profile_text": profile_text,        # 注入：用户偏好画像
            "parsed_task": None,
            "retrieved_materials": [],
            "draft_content": "",
            "audit_result": None,
            "iteration_count": 0,
            "max_iteration": settings.audit_max_iteration,
            "degrade_mode": False,
            "error_info": "",
            "need_more_retrieval": False,
            "node_failed": "",
        }

        final = initial_state
        pre_polish_audit_score: Optional[float] = None  # 用于反思日志
        try:
            graph = _get_graph()
            if graph is not None:
                final = graph.invoke(initial_state)
            else:
                final = _manual_fallback(initial_state)
        except Exception as e:
            logger.exception(f"工作流异常：{e}")
            final = dict(initial_state)
            final["error_info"] = str(e)
            final["degrade_mode"] = True
            _emit({"type": "workflow_error", "error": str(e)})

        dt_ms = (time.time() - t0) * 1000
        _emit({
            "type": "workflow_done",
            "ts": time.time(),
            "elapsed_ms": round(dt_ms, 1),
            "session_id": session.session_id,
        })

        # 构建响应
        has_error = bool(final.get("error_info"))
        pt = final.get("parsed_task")
        content = final.get("draft_content") or ""
        audit_result = final.get("audit_result")

        # -------- 记忆模块：写入反思日志 + 更新画像 + 持久化会话 --------
        try:
            reflection_entry: Optional[ReflectionEntry] = None
            iteration = final.get("iteration_count") or 0
            if audit_result is not None and iteration > 0:
                # 只有发生了反思迭代（审核→修改）才记录一条反思日志
                issues_found = []
                try:
                    for issue in getattr(audit_result, "issues", []):
                        level = getattr(issue, "level", "")
                        suggestion = getattr(issue, "suggestion", "")
                        issues_found.append(f"[{level}] {suggestion[:40]}")
                except Exception:
                    pass
                reflection_entry = ReflectionEntry(
                    session_id=session.session_id,
                    original_content=content[:300],
                    revision_content=content[:300],
                    audit_score_before=pre_polish_audit_score or 0.0,
                    audit_score_after=getattr(audit_result, "score", 0.0),
                    issues_found=issues_found,
                    iteration=iteration,
                )

            sm.after_workflow(
                session=session,
                content=content,
                audit_result=audit_result,
                parsed_task=pt,
                reflection_entry=reflection_entry,
            )
            logger.info(f"[Memory] 会话已持久化：reflection={reflection_entry is not None}")
        except Exception as e:
            logger.warning(f"[Memory] 会话持久化失败：{e}")

        return FinalResponse(
            success=not has_error,
            task_type=getattr(pt, "task_type", None) if pt else None,
            content=content,
            audit_result=audit_result,
            iteration_count=final.get("iteration_count") or 0,
            degrade_mode=bool(final.get("degrade_mode")),
            error=final.get("error_info") or None,
            elapsed_ms=dt_ms,
            session_id=session.session_id,
            has_context=has_context,
            user_profile_summary=profile_text if profile_text else None,
        )
    finally:
        if event_callback is not None:
            _clear_event_callback()


# =============================================================================
# 手动调度回退（LangGraph 不可用时也能跑通）
# =============================================================================


def _manual_fallback(state: dict) -> dict:
    logger.info("[Graph] 使用手动调度回退路径")
    new_state = dict(state)
    new_state.update(_safe_node(run_parse, "parse_node")(new_state) or {})
    if _route_after_parse(new_state) == "retrieve":
        new_state.update(_safe_node(run_retrieve, "retrieve_node")(new_state) or {})
    for _ in range(settings.audit_max_iteration + 1):
        new_state.update(_safe_node(run_polish, "polish_node")(new_state) or {})
        if new_state.get("need_more_retrieval") and (new_state.get("iteration_count") or 0) < 1:
            new_state.update(_safe_node(run_retrieve, "retrieve_node")(new_state) or {})
        new_state.update(_safe_node(run_audit, "audit_node")(new_state) or {})
        audit = new_state.get("audit_result")
        iteration = new_state.get("iteration_count") or 0
        if audit and (audit.passed or iteration >= settings.audit_max_iteration):
            break
    return new_state


# =============================================================================
# 辅助：列出已注册工具（与 tools.registry 对齐）
# =============================================================================


def list_tools() -> list:
    from .tools import registry
    return registry.list_tools()
