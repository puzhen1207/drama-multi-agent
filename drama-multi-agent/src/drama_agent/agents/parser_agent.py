"""任务解析 Agent —— 把用户原始输入解析为结构化任务（ParsedTask）。"""
from __future__ import annotations

import re
from typing import Any, Dict

from ..exceptions import with_retry
from ..llm import chat_structured, llm_available
from ..logging_setup import get_logger
from ..models import ParsedTask
from .prompts import PARSER_FEW_SHOTS, PARSER_SYSTEM_PROMPT

logger = get_logger("parser_agent")


@with_retry
def run_parse(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph 节点：解析用户原始输入 → 返回含 parsed_task 的 dict。"""
    raw_input = state.get("raw_input") or ""
    logger.info(f"[Parser] 解析原始输入（{len(raw_input)} 字）")

    if llm_available():
        try:
            task = chat_structured(
                pydantic_cls=ParsedTask,
                user_prompt=f"请解析以下用户输入：\n{raw_input}",
                system_prompt=PARSER_SYSTEM_PROMPT,
                few_shots=PARSER_FEW_SHOTS,
            )
            logger.info(f"[Parser] 解析完成（LLM）：task_type={task.task_type}, topic={task.topic}")
            return {"parsed_task": task}
        except Exception as e:
            logger.warning(f"[Parser] LLM 解析失败：{e}，走规则解析")

    task = _rule_based_parse(raw_input)
    logger.info(f"[Parser] 解析完成（规则）：task_type={task.task_type}, topic={task.topic}")
    return {"parsed_task": task}


def _rule_based_parse(text: str) -> ParsedTask:
    text_lower = (text or "").lower()
    task_type = "copywriting"
    needs_retrieval = True
    style = "爽文"

    if any(k in text for k in ("整理", "大纲", "章节", "人设", "结构")):
        task_type = "content_organize"
    elif any(k in text for k in ("合规", "审核", "检查", "是否违规", "审查")):
        task_type = "audit"
        needs_retrieval = False
    elif any(k in text for k in ("答疑", "问答", "?", "？", "规则", "怎么", "如何")):
        task_type = "qa"
    elif any(k in text for k in ("推广", "文案", "标题", "海报", "营销")):
        task_type = "copywriting"

    if any(k in text for k in ("虐", "哭", "悲剧")):
        style = "虐恋"
    elif any(k in text for k in ("悬疑", "推理", "破案", "密室")):
        style = "悬疑"
    elif any(k in text for k in ("甜", "宠", "恋爱", "浪漫")):
        style = "甜宠"
    elif any(k in text for k in ("都市", "职场")):
        style = "都市"
    elif any(k in text for k in ("古装", "穿越", "重生")):
        style = "古装"
    elif any(k in text for k in ("科幻", "未来", "外星")):
        style = "科幻"

    tokens = [t for t in re.sub(r"[，。,.!?！？\s]+", " ", text).split() if t]
    keywords = list({t for t in tokens if 1 < len(t) <= 8})[:8] or ["短剧"]

    target_length = 500
    m = re.search(r"(\d{2,5})\s*(?:字|词)", text)
    if m:
        try:
            target_length = int(m.group(1))
        except Exception:
            target_length = 500

    topic = text[:40] or "短剧"
    return ParsedTask(
        task_type=task_type,
        topic=topic,
        style=style,
        target_length=target_length,
        keywords=keywords,
        needs_retrieval=needs_retrieval,
        requirements=text,
        raw_explanation="[降级] 基于规则引擎的启发式解析",
    )
