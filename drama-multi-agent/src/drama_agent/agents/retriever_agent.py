"""素材检索 Agent。

核心改进（相比 drama-multi-agent 原版）：
- 空知识库 / 0 命中 不再 degrade_mode = True；
- 避免因为没有素材而让整个系统"降级"，从而让 polish 继续正常生成。
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..exceptions import EmptyMaterialError, RetrievalError
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from ..tools.vector_retriever import tool_retrieve_materials

logger = get_logger("retriever_agent")


def run_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    parsed = state.get("parsed_task")
    raw_input = state.get("raw_input") or ""

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
    try:
        results = tool_retrieve_materials(query=query, top_k=3)
        materials = [RetrievedMaterial(**r) for r in results]
        if not materials:
            logger.info("[Retriever] 无命中素材，走纯生成模式（不降级）")
            return {"retrieved_materials": []}
        logger.info(f"[Retriever] 命中 {len(materials)} 条素材")
        return {"retrieved_materials": [m.model_dump() for m in materials]}
    except EmptyMaterialError:
        logger.info("[Retriever] 知识库为空，走纯生成模式（不降级）")
        return {"retrieved_materials": []}
    except RetrievalError as e:
        logger.error(f"[Retriever] 检索异常：{e}；继续走纯生成模式（不降级）")
        return {"retrieved_materials": []}
    except Exception as e:
        logger.error(f"[Retriever] 未预期异常：{e}；继续走纯生成模式")
        return {"retrieved_materials": []}
