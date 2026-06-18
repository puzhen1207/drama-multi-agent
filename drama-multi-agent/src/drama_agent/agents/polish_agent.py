"""
内容润色 Agent —— 基于素材 + 草稿 + 审核反馈 + 会话上下文，生成/重写内容
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import LLMServiceError, with_retry
from ..logging_setup import get_logger
from ..llm import chat, llm_available
from ..models import RetrievedMaterial
from ..tools.text_processor import tool_normalize_text
from .prompts import POLISH_SYSTEM_PROMPT, build_polish_user_prompt

logger = get_logger("polish_agent")


@with_retry
def run_polish(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph 节点：接收 state，返回部分更新的 state。"""
    parsed = state.get("parsed_task")
    materials = state.get("retrieved_materials") or []
    draft = state.get("draft_content") or ""
    audit = state.get("audit_result")
    iteration = state.get("iteration_count") or 0

    # 新增：从 state 读取会话上下文和用户画像摘要（memory 模块写入）
    session_context = state.get("session_context") or ""
    user_profile_text = state.get("user_profile_text") or ""

    topic = parsed.topic if parsed and hasattr(parsed, "topic") else ""
    style = parsed.style if parsed and hasattr(parsed, "style") else "爽文"
    target_length = parsed.target_length if parsed and hasattr(parsed, "target_length") else 500
    requirements = parsed.requirements if parsed and hasattr(parsed, "requirements") else ""
    task_type = parsed.task_type if parsed and hasattr(parsed, "task_type") else "copywriting"

    # 组装素材
    materials_text = _format_materials(materials) if isinstance(materials, list) else ""

    # 审核反馈
    audit_text = ""
    if audit and not audit.passed:
        lines = []
        for issue in audit.issues:
            level = getattr(issue, "level", "")
            cat = getattr(issue, "category", "")
            pos = getattr(issue, "position", "")
            sugg = getattr(issue, "suggestion", "")
            lines.append(f"- 【{level}】{cat}：{pos} -> {sugg}")
        if getattr(audit, "summary", ""):
            lines.append(f"- 整体结论：{audit.summary}")
        audit_text = "\n".join(lines) if lines else audit.summary

    # 组装 LLM prompt（包含会话上下文 + 用户画像）
    user_prompt = build_polish_user_prompt(
        task_type=task_type,
        topic=topic,
        style=style,
        target_length=target_length,
        requirements=requirements,
        materials=materials_text,
        draft=draft,
        audit_feedback=audit_text,
        session_context=session_context,
        user_profile_text=user_profile_text,
    )

    logger.info(
        f"[Polish] iter={iteration}, materials={len(materials) if isinstance(materials, list) else 0}, "
        f"audit_feedback={'有' if audit_text else '无'}, has_context={'有' if session_context else '无'}"
    )

    content: str
    if llm_available():
        content = chat(user_prompt=user_prompt, system_prompt=POLISH_SYSTEM_PROMPT)
    else:
        content = _stub_polish(task_type, topic, style, target_length, materials_text)

    content = tool_normalize_text(content)

    # 素材不足时触发反向检索（仅第一次迭代时）
    need_more = False
    if not materials and iteration == 0:
        need_more = False

    logger.info(f"[Polish] 生成内容 {len(content)} 字")
    return {"draft_content": content, "need_more_retrieval": need_more}


def _format_materials(materials: list) -> str:
    items = []
    for i, m in enumerate(materials):
        if isinstance(m, RetrievedMaterial):
            title, category, content = m.title, m.category, m.content
        elif isinstance(m, dict):
            title = m.get("title", "")
            category = m.get("category", "")
            content = m.get("content", "")
        else:
            continue
        snippet = content[:300]
        items.append(
            f"--- 素材 {i+1}（分类：{category}，标题：{title}）\n{snippet}\n"
        )
    return "\n".join(items)


def _stub_polish(task_type: str, topic: str, style: str, length: int, materials: str) -> str:
    body = (
        f"【《{topic}》{style}风格短剧（STUB 模式）】\n\n"
        f"【剧情亮点】\n"
        f"1. 开场三幕式结构，第 1 集就抛出核心冲突，制造悬念。\n"
        f"2. 节奏密集，每 300 字一个反转或情绪钩子，强化叙事张力。\n"
        f"3. 台词口语化、对话驱动叙事，贴合短剧观众阅读习惯。\n\n"
        f"【分集概览】\n"
        f"第 1 集：核心冲突登场，建立主角动机与情感基调。\n"
        f"第 2 集：矛盾升级，引入关键配角。\n"
        f"第 3 集：高潮反转，结尾留下强烈钩子。\n\n"
        f"（字数目标 {length} 字，task={task_type}）"
    )
    if materials:
        body += f"\n\n【参考素材摘要】\n基于素材摘要：{materials[:300]}"
    return body
