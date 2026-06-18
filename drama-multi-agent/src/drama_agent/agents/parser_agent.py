"""
任务解析 Agent —— 把用户原始输入解析为结构化任务
"""
from __future__ import annotations

from typing import Any

from ..exceptions import ValidationError, with_retry
from ..logging_setup import get_logger
from ..llm import chat_structured, llm_available
from ..models import ParsedTask
from .prompts import PARSER_FEW_SHOTS, PARSER_SYSTEM_PROMPT

logger = get_logger("parser_agent")


@with_retry
def run_parse(state: dict) -> dict:
    raw_input = state.get("raw_input") or ""
    logger.info(f"[Parser] 解析原始输入（{len(raw_input)} 字）")

    task: ParsedTask
    if llm_available():
        user_prompt = f"请解析以下用户输入：\n{raw_input}"
        few_shots_as_text = "\n\n".join(
            f"示例输入 {i+1}：{q}\n示例输出 {i+1}：{a}"
            for i, (q, a) in enumerate(PARSER_FEW_SHOTS)
        )
        task = chat_structured(
            pydantic_cls=ParsedTask,
            user_prompt=user_prompt,
            system_prompt=PARSER_SYSTEM_PROMPT + "\n\n" + few_shots_as_text,
        )
    else:
        # 降级：走规则解析
        task = _rule_based_parse(raw_input)
    logger.info(f"[Parser] 解析完成：task_type={task.task_type}, topic={task.topic}")
    return {"parsed_task": task}


def _rule_based_parse(text: str) -> ParsedTask:
    """无 LLM 时的降级解析，基于关键词启发式。"""
    text_lower = text.lower()
    task_type = "copywriting"
    needs_retrieval = True
    style = "爽文"

    if any(k in text for k in ("整理", "大纲", "章节", "人设", "结构")):
        task_type = "content_organize"
    elif any(k in text for k in ("合规", "审核", "检查", "是否违规", "审查")):
        task_type = "audit"
        needs_retrieval = False
    elif any(k in text for k in ("答疑", "问答", "?", "？", "要求", "规则", "怎么")):
        task_type = "qa"
    elif any(k in text for k in ("推广", "文案", "标题", "海报")):
        task_type = "copywriting"

    if any(k in text for k in ("虐", "哭", "悲剧")):
        style = "虐恋"
    elif any(k in text for k in ("悬疑", "推理", "破案", "密室")):
        style = "悬疑"
    elif any(k in text for k in ("甜", "宠", "恋爱", "浪漫")):
        style = "甜宠"

    # 关键词：中文每 4~6 字取一段，用 set 去重
    tokens = [t for t in text.replace("，", " ").replace("。", " ").split() if t]
    keywords = list({t for t in tokens if 1 < len(t) <= 8})[:8] or ["短剧"]

    # 字数估计
    target_length = 500
    import re

    m = re.search(r"(\d{2,5})\s*(?:字|词)", text)
    if m:
        target_length = int(m.group(1))

    return ParsedTask(
        task_type=task_type,
        topic=text[:40] or "短剧",
        style=style,
        target_length=target_length,
        keywords=keywords,
        needs_retrieval=needs_retrieval,
        requirements=text,
        raw_explanation="[降级] 基于规则引擎的启发式解析",
    )
