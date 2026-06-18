"""
合规规则引擎：敏感词检测 + 规则校验
- 三级分级：forbidden（硬违规，直接拦截）/ warning（需修改）/ suggestion（优化）
- 敏感词库支持从 txt 文件加载；内置中文短剧行业常用敏感词兜底
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import settings
from ..logging_setup import get_logger
from . import registry

logger = get_logger("compliance")


# =============================================================================
# 内置敏感词与正则规则
# =============================================================================

_CATEGORIES: Dict[str, List[str]] = {
    "政治敏感": [
        "领导人姓名", "敏感政治术语", "反动", "台独", "港独", "法轮功",
    ],
    "暴力血腥": ["血腥", "砍杀", "虐杀", "爆头", "分尸", "自杀", "自残", "烧杀"],
    "色情低俗": ["色情", "低俗", "床戏", "裸", "嫖娼", "强奸", "意淫"],
    "毒品": ["毒品", "吸毒", "大麻", "海洛因", "冰毒"],
    "赌博": ["赌博", "博彩", "赌场", "百家乐", "老虎机"],
    "恐怖惊悚": ["恐怖", "惊悚", "鬼", "诅咒", "邪恶仪式"],
    "歧视": ["种族歧视", "地域歧视", "性别歧视", "辱骂"],
}

# 硬违规（命中即不通过）
_FORBIDDEN_KEYWORDS = [
    "反动", "台独", "港独", "法轮功", "色情", "毒品", "吸毒",
    "赌博", "砍杀", "虐杀", "血腥", "分尸", "烧杀", "嫖娼", "强奸",
]

# 警告级（需要改写）
_WARNING_KEYWORDS = [
    "暴力", "裸", "床戏", "惊悚", "赌场", "老虎机", "意淫", "自残",
    "自杀", "爆头", "鬼片",
]

# 正则规则（邮箱、电话、极端数字等）
_REGEX_RULES: List[Tuple[str, str]] = [
    ("personal_phone", r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    ("personal_idcard", r"(?<!\d)(\d{17}[\dXx])(?!\d)"),
    ("extreme_number", r"(^|\s|，|。)(\d{15,})(\s|，|。|$)"),
]


# =============================================================================
# 规则引擎
# =============================================================================

class RuleEngine:
    """三级分级规则引擎。"""

    def __init__(self, custom_words_path: Optional[Path] = None):
        self.forbidden: List[str] = list(_FORBIDDEN_KEYWORDS)
        self.warning: List[str] = list(_WARNING_KEYWORDS)
        self.suggestion: List[str] = []
        self._load_custom(custom_words_path or settings.absolute_sensitive_words_path)
        # 正则
        self.regex_rules = _REGEX_RULES
        logger.info(
            f"[Compliance] 规则引擎就绪：forbidden={len(self.forbidden)}, "
            f"warning={len(self.warning)}, regex={len(self.regex_rules)}"
        )

    def _load_custom(self, path: Optional[Path]) -> None:
        if not path or not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # 支持 level:word 的写法
                    if ":" in line:
                        level, word = line.split(":", 1)
                    else:
                        level, word = "warning", line
                    level = level.strip().lower()
                    word = word.strip()
                    if not word:
                        continue
                    if level in ("forbidden", "f", "forbid"):
                        self.forbidden.append(word)
                    elif level in ("warning", "w"):
                        self.warning.append(word)
                    else:
                        self.suggestion.append(word)
        except Exception as e:
            logger.warning(f"[Compliance] 自定义敏感词加载失败：{e}")

    def check(self, text: str) -> Dict:
        """
        返回：{
            "forbidden": list[(category, keyword, snippet)],
            "warning":   list[(category, keyword, snippet)],
            "suggestion":list[(category, keyword, snippet)],
            "regex":     list[(rule_name, matched, position)],
            "passed_rule": bool,
        }
        """
        text = text or ""
        forbidden: List[dict] = []
        warning: List[dict] = []
        suggestion: List[dict] = []

        def _match(words: List[str], bucket: List[dict], level_name: str) -> None:
            for kw in words:
                if kw and kw in text:
                    # 取命中片段
                    idx = text.find(kw)
                    start = max(0, idx - 8)
                    end = min(len(text), idx + 8)
                    bucket.append({
                        "level": level_name,
                        "category": _detect_category(kw),
                        "keyword": kw,
                        "snippet": text[start:end],
                    })

        _match(self.forbidden, forbidden, "forbidden")
        _match(self.warning, warning, "warning")
        _match(self.suggestion, suggestion, "suggestion")

        regex_hits = []
        for name, pattern in self.regex_rules:
            for m in re.finditer(pattern, text):
                regex_hits.append({
                    "level": "forbidden",
                    "category": "个人信息",
                    "keyword": m.group(0)[:16],
                    "snippet": text[max(0, m.start() - 8):m.end() + 8],
                })
        passed_rule = len(forbidden) == 0 and len(regex_hits) == 0
        return {
            "forbidden": forbidden + regex_hits,
            "warning": warning,
            "suggestion": suggestion,
            "passed_rule": passed_rule,
        }


def _detect_category(kw: str) -> str:
    for cat, words in _CATEGORIES.items():
        for w in words:
            if w in kw or kw in w:
                return cat
    return "其他"


_rule_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine


# =============================================================================
# MCP 工具：敏感词检测
# =============================================================================

def tool_sensitive_check(text: str) -> dict:
    """[MCP Tool] 敏感词检测 + 规则校验，返回命中详情。"""
    return get_rule_engine().check(text)


registry.register(
    "sensitive_check",
    tool_sensitive_check,
    description="对输入文本做敏感词与规则校验，返回三级问题清单",
    input_schema={"type": "object", "properties": {
        "text": {"type": "string", "description": "待检测文本"},
    }},
)
