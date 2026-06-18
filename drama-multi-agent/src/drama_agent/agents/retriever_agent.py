"""
素材检索 Agent —— 调用 MCP vector_retriever 工具，收集父块素材。
"""
from __future__ import annotations

from ..exceptions import EmptyMaterialError, RetrievalError
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from ..tools.vector_retriever import tool_retrieve_materials

logger = get_logger("retriever_agent")


def run_retrieve(state: dict) -> dict:
    parsed = state.get("parsed_task")
    raw_input = state.get("raw_input") or ""
    need_more = state.get("need_more_retrieval", False)

    if parsed is None:
        query = raw_input[:200]
    else:
        topic = parsed.topic or ""
        keywords = " ".join(parsed.keywords or [])
        query = f"{topic} {keywords} {raw_input[:80]}".strip()

    logger.info(f"[Retriever] need_more={need_more}, query={query[:60]}")
    try:
        results = tool_retrieve_materials(query=query, top_k=3)
        materials = [RetrievedMaterial(**r) for r in results]
        if not materials:
            logger.warning("[Retriever] 未命中任何素材，设置 degrade_mode=true")
            return {
                "retrieved_materials": [],
                "degrade_mode": True,
                "error_info": "素材库无匹配内容（已走纯生成模式）",
            }
        logger.info(f"[Retriever] 命中 {len(materials)} 条素材")
        return {"retrieved_materials": materials}
    except EmptyMaterialError:
        logger.warning("[Retriever] 素材库为空，降级")
        return {"retrieved_materials": [], "degrade_mode": True, "error_info": "素材库为空"}
    except RetrievalError as e:
        logger.error(f"[Retriever] 检索异常：{e}")
        return {
            "retrieved_materials": [],
            "degrade_mode": True,
            "node_failed": "retrieve_node",
            "error_info": str(e),
        }
