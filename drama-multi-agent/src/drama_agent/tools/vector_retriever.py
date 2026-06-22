"""分层 RAG 向量检索工具。
- 父文档（剧本/文案章节）持久化到本地；
- 子块（200-300 字）做 Embedding → FAISS；
- 检索：Top-K 子块 → 映射父块 → 轻量重排 → 返回 Top-N 父块。

依赖可选：本地 sentence-transformers 做 embedding；若不可用会走随机向量的降级路径。
"""
from __future__ import annotations

import json
import pickle
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import settings
from ..exceptions import EmptyMaterialError, RetrievalError, with_retry
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from . import registry

logger = get_logger("retriever")


# ============= Embedding 适配层 =============


class EmbeddingProvider:
    """统一 embedding 接口：优先本地 sentence-transformers，失败走随机。"""

    def __init__(self, dim: int = 384):
        self.model = None
        self.dim = dim
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = int(self.model.get_sentence_embedding_dimension() or dim)
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
        # 降级：稳定随机向量（hash 文本以确保相同文本得到相同向量）
        rs = np.random.default_rng(seed=42)
        vecs = rs.standard_normal((len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ============= FAISS 索引保存辅助（兼容 Windows 中文路径）=============


def _faiss_save_index(index: Any, path: Path) -> None:
    """使用 faiss.serialize_index + numpy + 文件 IO，避免 C++ 层的路径编码问题。"""
    import faiss  # type: ignore

    buf = faiss.serialize_index(index)
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with open(str(tmp), "wb") as f:
        np.save(f, buf)
    tmp.replace(str(path))


def _faiss_load_index(path: Path) -> Any:
    """使用 numpy + faiss.deserialize_index 加载。"""
    import faiss  # type: ignore

    buf = np.load(str(path), allow_pickle=False)
    return faiss.deserialize_index(buf)


# ============= FAISS 向量索引 + 父块映射 =============


class HierarchicalVectorStore:
    """分层 RAG：父文档 + 子块。"""

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = Path(index_dir or settings.absolute_vector_index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.pkl"
        self.embedding = EmbeddingProvider()
        self.faiss_index = None
        self.sub_meta: List[dict] = []
        self.parents: Dict[str, dict] = {}
        self._load_or_init()

    # ---------- IO ----------

    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.faiss_index = _faiss_load_index(self.index_path)
                with open(str(self.meta_path), "rb") as f:
                    data = pickle.load(f)
                self.sub_meta = data.get("sub_meta", [])
                self.parents = data.get("parents", {})
                logger.info(f"[FAISS] 从磁盘加载：子块 {len(self.sub_meta)}，父块 {len(self.parents)}")
                return
            except Exception as e:
                logger.warning(f"[FAISS] 加载失败，重建索引：{e}")
        try:
            import faiss as _faiss  # type: ignore
            del _faiss  # 只检查可用性
        except Exception:
            logger.warning("[FAISS] faiss-cpu 不可用，将走纯 numpy 线性扫描")
        logger.info("[FAISS] 新建空索引")

    def save(self) -> None:
        if self.faiss_index is not None:
            try:
                _faiss_save_index(self.faiss_index, self.index_path)
            except Exception as e:
                logger.warning(f"[FAISS] 保存索引失败：{e}")
        try:
            tmp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
            with open(str(tmp_meta), "wb") as f:
                pickle.dump(
                    {"sub_meta": self.sub_meta, "parents": self.parents, "dim": self.embedding.dim},
                    f,
                )
            tmp_meta.replace(str(self.meta_path))
        except Exception as e:
            logger.warning(f"[FAISS] 保存元数据失败：{e}")
        logger.info("[FAISS] 索引已保存")

    # ---------- 构建 ----------

    def add_documents(
        self,
        documents: List[dict],
        chunk_size: int = 250,
        chunk_overlap: int = 30,
    ) -> None:
        new_subs: List[str] = []
        for doc in documents:
            parent_id = "P_" + uuid.uuid4().hex[:8]
            title = doc.get("title", "未命名")
            content = doc.get("content", "")
            category = doc.get("category", "unknown")
            self.parents[parent_id] = {"title": title, "content": content, "category": category}
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
        vecs = self.embedding.encode(new_subs)
        self._add_vectors(vecs)
        logger.info(f"[FAISS] 新增 {len(new_subs)} 子块，{len(documents)} 父块")

    def _add_vectors(self, vecs: np.ndarray) -> None:
        try:
            import faiss  # type: ignore
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(int(self.embedding.dim))
            self.faiss_index.add(vecs.astype(np.float32))
            return
        except Exception as e:
            logger.warning(f"[FAISS] faiss 添加失败：{e}，走 numpy 线性扫描")
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
        if not self.sub_meta:
            raise EmptyMaterialError("知识库为空，请先构建素材库")
        top_k_sub = top_k_sub or settings.retrieve_top_k
        top_k_parent = top_k_parent or settings.rerank_top_k
        top_k_sub = min(top_k_sub, len(self.sub_meta))
        qvec = self.embedding.encode_one(query).astype(np.float32).reshape(1, -1)
        scores, indices = self._search_impl(qvec, top_k_sub)
        # 聚合父块得分
        parent_scores: Dict[str, float] = {}
        parent_hits: Dict[str, int] = {}
        for s, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.sub_meta):
                continue
            pid = self.sub_meta[int(idx)]["parent_id"]
            norm_s = float(max(0.0, min(1.0, (float(s) + 1.0) / 2.0)))
            parent_scores[pid] = parent_scores.get(pid, 0.0) + norm_s
            parent_hits[pid] = parent_hits.get(pid, 0) + 1
        ranked = sorted(
            parent_scores.items(),
            key=lambda kv: kv[1] / max(1, parent_hits[kv[0]]),
            reverse=True,
        )
        # 轻量重排：基于 query 与父块内容的词重叠
        results: List[RetrievedMaterial] = []
        for pid, _ in ranked[: max(top_k_parent * 3, 10)]:
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

    def _search_impl(self, qvec: np.ndarray, k: int):
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
        # 空索引：返回 -1
        return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=np.int64)

    def count(self) -> dict:
        return {"sub_blocks": len(self.sub_meta), "parents": len(self.parents)}

    def is_empty(self) -> bool:
        return len(self.sub_meta) == 0


# ============= 文本切分 / 轻量重排 =============


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        if len(chunk) < 50 and chunks:
            chunks[-1] = chunks[-1] + chunk
            break
        chunks.append(chunk)
    return chunks


def _light_rerank(query: str, doc: str) -> float:
    """基于关键词覆盖度的轻量重排（0~1）。"""
    try:
        import jieba  # type: ignore
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


# ============= 单例 + MCP 工具注册 =============


_vector_store: Optional[HierarchicalVectorStore] = None


def get_vector_store() -> HierarchicalVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = HierarchicalVectorStore()
    return _vector_store


def tool_retrieve_materials(query: str, top_k: int = 3) -> List[dict]:
    """[MCP Tool] 短剧素材检索工具：输入查询，返回最相关的 TopK 素材（dict 列表）。"""
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
)


# 提供一个便捷的 JSON 兼容导出函数，方便 api.py 使用
def ensure_builtin_knowledge() -> None:
    """启动时自动构建基础知识库（空索引时才会写入）。"""
    vs = get_vector_store()
    if not vs.is_empty():
        return
    builtin_docs = [
        {
            "title": "短剧爽文剧本结构模板",
            "category": "剧本",
            "content": (
                "第一幕：开场冲突。用 300 字交代主角身份、所处困境，制造强烈情绪钩子。"
                "第二幕：反转升级。引入关键配角或外力，让局势反复反转，保持高密度节奏。"
                "第三幕：高潮+钩子。冲突达到顶点，以巨大悬念结尾，吸引读者看下一集。"
                "核心技巧：爽点前置、对话驱动叙事、每 500 字一个情绪钩子。"
            ),
        },
        {
            "title": "都市爆款文案样例",
            "category": "文案",
            "content": (
                "标题公式：【强烈反差】她被豪门抛弃三年，归来时身价十亿。"
                "正文公式：林晚从来没想过，离婚后第一次见顾言，会是在他的订婚宴上。"
                "她端着酒杯，笑着走上前：「祝你们幸福。」顾言却一把攥住她的手腕——"
            ),
        },
        {
            "title": "短剧行业合规红线",
            "category": "规则",
            "content": (
                "1. 禁止政治敏感内容、国家领导人姓名与相关符号。"
                "2. 禁止色情低俗、淫秽暗示、床戏赤裸描写。"
                "3. 禁止血腥暴力、砍杀虐杀、自残自杀的详细描写。"
                "4. 禁止毒品、赌博、邪教、恐怖主义相关内容。"
                "5. 禁止歧视性言论（种族、地域、性别、残障等）。"
                "6. 禁止暴露个人信息（身份证、手机号、地址）。"
                "7. 禁止未成年人出现成人化、低俗化剧情。"
            ),
        },
        {
            "title": "霸总追妻人设参考",
            "category": "人设",
            "content": (
                "身份：30 岁左右的企业总裁 / CEO / 家族继承人。"
                "外貌：身材挺拔，五官深邃，气场压人。"
                "性格：外冷内热，控制欲强，对女主专一，有不可告人的过去。"
                "动机：弥补当年对女主的伤害 / 复仇 / 守护家庭。"
                "核心张力：霸道的控制 vs 隐藏的深情；强势 vs 脆弱。"
            ),
        },
    ]
    try:
        vs.add_documents(builtin_docs)
        vs.save()
        logger.info(f"[API] 已自动构建基础知识库：{len(builtin_docs)} 份文档")
    except Exception as e:
        logger.warning(f"[API] 自动构建知识库失败：{e}")
