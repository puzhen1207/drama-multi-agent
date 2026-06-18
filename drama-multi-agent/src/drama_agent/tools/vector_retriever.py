"""
分层 RAG 向量检索工具
- 父文档 (剧本/文案章节) 持久化到本地
- 子块 (200-300字) 做 Embedding，存 FAISS
- 检索：Top-K 子块 -> 映射父块 -> Reranker 重排 -> 返回 Top-N 父块

依赖可选：本地 sentence-transformers 做 embedding；若不可用则走随机向量的降级路径。
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import settings
from ..exceptions import EmptyMaterialError, RetrievalError, with_retry
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from . import registry

logger = get_logger("retriever")


# =============================================================================
# Embedding 适配层
# =============================================================================

class EmbeddingProvider:
    """统一 embedding 接口：优先本地 sentence-transformers，失败走随机。"""

    def __init__(self, dim: int = 384):
        self.model = None
        self.dim = dim
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )  # 体积小、下载快；中文可切 BAAI/bge-m3
            self.dim = self.model.get_sentence_embedding_dimension() or dim
            logger.info(f"[Embedding] 已加载 sentence-transformers, dim={self.dim}")
        except Exception as e:
            logger.warning(f"[Embedding] sentence-transformers 不可用（{e}），走随机向量降级")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            try:
                vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Embedding] encode 失败：{e}，回退随机向量")
        # 降级：随机向量（保证可运行）
        rs = np.random.default_rng(seed=42)
        vecs = rs.standard_normal((len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# =============================================================================
# FAISS 向量索引 + 父块映射
# =============================================================================

class HierarchicalVectorStore:
    """分层 RAG：父文档（剧本/章节）+ 子块（200-300字语义块）。"""

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir or settings.absolute_vector_index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.pkl"
        self.embedding = EmbeddingProvider()
        self.faiss_index = None
        # 元数据：sub_id -> {"parent_id", "text", "category", "title"}
        self.sub_meta: List[dict] = []
        # parent_id -> {"title", "content", "category"}
        self.parents: dict = {}
        self._load_or_init()

    # ---------- IO ----------

    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                import faiss

                self.faiss_index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    data = pickle.load(f)
                self.sub_meta = data.get("sub_meta", [])
                self.parents = data.get("parents", {})
                logger.info(f"[FAISS] 从磁盘加载：子块 {len(self.sub_meta)}，父块 {len(self.parents)}")
                return
            except Exception as e:
                logger.warning(f"[FAISS] 加载失败，重建索引：{e}")
        # 初始化空索引
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(self.embedding.dim)
        except Exception:
            # 若 faiss 不可用，退化到一个简单的 NumPy 向量表
            self.faiss_index = None
        logger.info("[FAISS] 新建空索引")

    def save(self) -> None:
        if self.faiss_index is not None:
            try:
                import faiss

                faiss.write_index(self.faiss_index, str(self.index_path))
            except Exception:
                # IndexFlatIP 可能无 write_index 适配；用 pickle 兜底
                pass
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {"sub_meta": self.sub_meta, "parents": self.parents, "dim": self.embedding.dim},
                f,
            )
        logger.info("[FAISS] 索引已保存")

    # ---------- 构建 ----------

    def add_documents(
        self,
        documents: List[dict],
        chunk_size: int = 250,
        chunk_overlap: int = 30,
    ) -> None:
        """
        documents: [{
            "title": str,
            "content": str,           # 父块全文
            "category": "剧本|文案|人设|规则",
        }]
        """
        new_subs = []
        new_parents = {}
        for doc in documents:
            parent_id = "P_" + uuid.uuid4().hex[:8]
            title = doc.get("title", "未命名")
            content = doc.get("content", "")
            category = doc.get("category", "unknown")
            self.parents[parent_id] = {"title": title, "content": content, "category": category}
            new_parents[parent_id] = True
            # 语义切块：按 chunk_size 切 + overlap
            chunks = _split_text(content, chunk_size, chunk_overlap) or [content]
            for i, chunk in enumerate(chunks):
                self.sub_meta.append({
                    "parent_id": parent_id,
                    "text": chunk,
                    "title": title,
                    "category": category,
                    "index": i,
                })
                new_subs.append(chunk)
        if not new_subs:
            logger.info("[FAISS] 无新文档")
            return
        # 编码并写入向量索引
        vecs = self.embedding.encode(new_subs)
        self._add_vectors(vecs)
        logger.info(f"[FAISS] 新增 {len(new_subs)} 子块，{len(new_parents)} 父块")

    def _add_vectors(self, vecs: np.ndarray) -> None:
        if self.faiss_index is not None:
            try:
                import faiss

                if not isinstance(self.faiss_index, faiss.Index):
                    self.faiss_index = None
                else:
                    self.faiss_index.add(vecs.astype(np.float32))
                    return
            except Exception:
                self.faiss_index = None
        # 降级：numpy 线性扫描
        if not hasattr(self, "_numpy_matrix") or self._numpy_matrix is None:
            self._numpy_matrix = vecs.astype(np.float32)
        else:
            self._numpy_matrix = np.vstack([self._numpy_matrix, vecs.astype(np.float32)])

    # ---------- 检索 ----------

    @with_retry
    def search(
        self,
        query: str,
        top_k_sub: Optional[int] = None,
        top_k_parent: Optional[int] = None,
    ) -> List[RetrievedMaterial]:
        """查询 → 子块 TopK → 映射父块 → 按分数去重合并 → Reranker 重排。"""
        if not self.sub_meta:
            raise EmptyMaterialError("知识库为空，请先调用 build_knowledge.py 构建素材库")
        top_k_sub = top_k_sub or settings.retrieve_top_k
        top_k_parent = top_k_parent or settings.rerank_top_k
        top_k_sub = min(top_k_sub, len(self.sub_meta))
        qvec = self.embedding.encode_one(query).astype(np.float32).reshape(1, -1)
        scores, indices = self._search_impl(qvec, top_k_sub)
        # 映射 + 合并
        parent_scores: dict = {}
        parent_hits: dict = {}
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.sub_meta):
                continue
            meta = self.sub_meta[int(idx)]
            pid = meta["parent_id"]
            # 归一化分数（cos 相似度范围 -1~1，转为 0~1）
            norm_s = float(max(0.0, min(1.0, (float(s) + 1.0) / 2.0)))
            parent_scores[pid] = parent_scores.get(pid, 0.0) + norm_s
            parent_hits[pid] = parent_hits.get(pid, 0) + 1
        # 按平均分排序
        ranked = sorted(
            parent_scores.items(),
            key=lambda kv: kv[1] / max(1, parent_hits[kv[0]]),
            reverse=True,
        )
        # Reranker（轻量实现：基于 query 与父块内容的 bm25-like 覆盖度，避免额外依赖）
        results: List[RetrievedMaterial] = []
        for pid, _score in ranked[: max(top_k_parent * 3, 10)]:
            parent = self.parents[pid]
            rerank_score = _light_rerank(query, parent["content"])
            results.append(RetrievedMaterial(
                material_id=pid,
                title=parent.get("title", ""),
                content=parent.get("content", ""),
                category=parent.get("category", "unknown"),
                score=float(rerank_score),
                source="faiss+rerank",
            ))
        results.sort(key=lambda m: m.score, reverse=True)
        final = results[:top_k_parent]
        logger.info(f"[FAISS] 召回 {len(final)} 条素材（子块 top_k={top_k_sub}）")
        return final

    def _search_impl(self, qvec, k: int):
        if self.faiss_index is not None:
            try:
                scores, indices = self.faiss_index.search(qvec, k)
                return scores, indices
            except Exception:
                pass
        matrix = getattr(self, "_numpy_matrix", None)
        if matrix is not None:
            sims = (matrix @ qvec.T).flatten()
            top_idx = np.argsort(-sims)[:k]
            return sims[top_idx].reshape(1, -1), top_idx.reshape(1, -1)
        return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=np.int64)

    def count(self) -> dict:
        return {"sub_blocks": len(self.sub_meta), "parents": len(self.parents)}


# =============================================================================
# 文本切分 / 轻量重排
# =============================================================================

def _split_text(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if len(chunk) < 50 and chunks:
            # 过小的尾段合并到上一块
            chunks[-1] = (chunks[-1] + chunk)
            break
        chunks.append(chunk)
    return chunks


def _light_rerank(query: str, doc: str) -> float:
    """基于关键词覆盖度的轻量重排（0~1）。"""
    try:
        import jieba

        q_tokens = set(jieba.lcut(query.lower()))
        d_tokens = set(jieba.lcut(doc.lower()))
    except Exception:
        q_tokens = set(query.lower().split())
        d_tokens = set(doc.lower().split())
    q_tokens = {t for t in q_tokens if t and t.strip() and len(t.strip()) >= 1}
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & d_tokens)
    return round(overlap / len(q_tokens), 3)


# =============================================================================
# 单例 + MCP 工具注册
# =============================================================================

_vector_store: Optional[HierarchicalVectorStore] = None


def get_vector_store() -> HierarchicalVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = HierarchicalVectorStore()
    return _vector_store


def tool_retrieve_materials(query: str, top_k: int = 3) -> List[dict]:
    """[MCP Tool] 短剧素材检索工具：输入查询，返回最相关的 TopK 素材。"""
    try:
        materials = get_vector_store().search(query=query, top_k_parent=top_k)
        return [m.model_dump() for m in materials]
    except EmptyMaterialError:
        logger.warning("[Retriever] 素材库为空，返回空列表")
        return []
    except Exception as e:
        logger.error(f"[Retriever] 检索异常：{e}")
        raise RetrievalError(str(e))


registry.register(
    "retrieve_materials",
    tool_retrieve_materials,
    description="根据关键词/主题从短剧素材库（剧本、爆款文案、人设、合规规则）中检索最相关素材",
    input_schema={"type": "object", "properties": {
        "query": {"type": "string", "description": "检索关键词或自然语言描述"},
        "top_k": {"type": "integer", "default": 3, "description": "返回素材数量"},
    }},
)
