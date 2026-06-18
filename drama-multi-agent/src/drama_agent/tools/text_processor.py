"""文本格式处理工具：清洗、规范化、长度截断等。"""
from __future__ import annotations

import re
from typing import List

from . import registry


def tool_normalize_text(text: str) -> str:
    """去除多余空白、合并空行。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tool_truncate_text(text: str, max_chars: int = 2000, keep_tail: bool = False) -> str:
    """按字符数截断文本，防止 LLM 上下文爆炸。"""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    if keep_tail:
        return text[-max_chars:]
    return text[:max_chars] + f"\n...（已截断，原文共 {len(text)} 字）"


def tool_split_paragraphs(text: str) -> List[str]:
    """按段落切分。"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


registry.register(
    "normalize_text",
    tool_normalize_text,
    description="规范化文本空白与换行",
    input_schema={"type": "object", "properties": {
        "text": {"type": "string", "description": "原始文本"},
    }},
)

registry.register(
    "truncate_text",
    tool_truncate_text,
    description="按字符数截断文本，避免上下文过大",
    input_schema={"type": "object", "properties": {
        "text": {"type": "string"},
        "max_chars": {"type": "integer", "default": 2000},
        "keep_tail": {"type": "boolean", "default": False},
    }},
)

registry.register(
    "split_paragraphs",
    tool_split_paragraphs,
    description="把文本按段落切分",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
)
