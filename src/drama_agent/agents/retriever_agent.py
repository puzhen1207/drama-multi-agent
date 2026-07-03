"""素材检索 Agent。

核心改进（相比 drama-multi-agent 原版）：
- 空知识库 / 0 命中 不再 degrade_mode = True；
- 避免因为没有素材而让整个系统"降级"，从而让 polish 继续正常生成。
- 合并公共素材库 + 个人记忆库检索结果。
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import settings
from ..exceptions import EmptyMaterialError, RetrievalError
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from ..tools.user_memory import search_user_memory
from ..tools.vector_retriever import tool_retrieve_materials

logger = get_logger("retriever_agent")


def _merge_materials(user_items: List[dict], public_items: List[dict], limit: int = 5) -> List[dict]:
    """个人记忆优先，去重后截断。"""
    seen: set = set()
    merged: List[dict] = []
    for item in user_items + public_items:
        key = (item.get("title", ""), (item.get("content") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def run_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    parsed = state.get("parsed_task")
    raw_input = state.get("raw_input") or ""
    user_id = state.get("user_id") or "guest"

    if parsed is not None and not getattr(parsed, "needs_retrieval", True):
        logger.info("[Retriever] 任务不需要检索，跳过")
        return {"retrieved_materials": []}

    if parsed is None:
        query = raw_input[:200]
    else:
        topic = getattr(parsed, "topic", "") or ""
        keywords = " ".join(getattr(parsed, "keywords", []) or [])
        query = f"{topic} {keywords} {raw_input[:80]}".strip()

    logger.info(f"[Retriever] query={query[:80]}")

    user_materials: List[dict] = []
    if settings.enable_user_memory and settings.user_memory_top_k > 0:
        try:
            user_materials = search_user_memory(user_id, query, top_k=settings.user_memory_top_k)
        except Exception as e:
            logger.warning(f"[Retriever] 个人记忆检索失败：{e}")

    public_materials: List[dict] = []
    try:
        public_materials = tool_retrieve_materials(query=query, top_k=3)
    except EmptyMaterialError:
        logger.info("[Retriever] 公共知识库为空")
    except RetrievalError as e:
        logger.error(f"[Retriever] 公共素材检索异常：{e}")
    except Exception as e:
        logger.error(f"[Retriever] 公共素材未预期异常：{e}")

    merged = _merge_materials(user_materials, public_materials, limit=5)
    if not merged:
        logger.info("[Retriever] 无命中素材，走纯生成模式（不降级）")
        return {"retrieved_materials": []}

    materials = [RetrievedMaterial(**r) for r in merged]
    logger.info(
        f"[Retriever] 命中 {len(materials)} 条（个人记忆 {len(user_materials)}，公共 {len(public_materials)}）"
    )
    return {"retrieved_materials": [m.model_dump() for m in materials]}
