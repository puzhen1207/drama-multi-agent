"""统一 Embedding 提供层：优先本地 sentence-transformers（中文模型），失败走哈希稳定向量降级。"""
from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import settings
from ..logging_setup import get_logger

logger = get_logger("embedding")

_MODEL_ALIASES = {
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
    "bge-large-zh": "BAAI/bge-large-zh-v1.5",
    "bge-base-zh": "BAAI/bge-base-zh-v1.5",
}


def _resolve_model_name(name: str) -> str:
    raw = (name or "BAAI/bge-small-zh-v1.5").strip()
    if "/" in raw:
        return raw
    return _MODEL_ALIASES.get(raw, raw)

_provider: Optional["EmbeddingProvider"] = None
_provider_lock = threading.Lock()


class EmbeddingProvider:
    """本地 embedding：配置项 EMBEDDING_MODEL，默认 BAAI/bge-small-zh-v1.5。"""

    def __init__(self) -> None:
        self.model = None
        self.mode = "hash_fallback"
        self.model_name = _resolve_model_name(settings.embedding_model or "BAAI/bge-small-zh-v1.5")
        self.dim = int(settings.embedding_dim or 512)
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(self.model_name)
            self.dim = int(self.model.get_sentence_embedding_dimension() or self.dim)
            self.mode = "sentence_transformers"
            logger.info(f"[Embedding] 已加载 {self.model_name}, dim={self.dim}")
        except Exception as e:
            self.model = None
            self.mode = "hash_fallback"
            logger.warning(
                f"[Embedding] sentence-transformers 不可用（{e}），"
                f"走哈希向量降级（dim={self.dim}）。"
                f"建议: pip install sentence-transformers && python scripts/build_knowledge.py --rebuild"
            )

    def is_real(self) -> bool:
        return self.mode == "sentence_transformers"

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            try:
                vecs = self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False,
                )
                return np.asarray(vecs, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Embedding] encode 失败：{e}，回退哈希向量")
        return np.vstack([self._hash_embed(t) for t in texts]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        """文本哈希确定性向量（比固定 seed 随机更适合无模型降级）。"""
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec)) + 1e-8
        return vec / norm

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.is_real(),
            "mode": self.mode,
            "model": self.model_name if self.is_real() else None,
            "dim": self.dim,
            "degraded": not self.is_real(),
            "hint": (
                None if self.is_real()
                else "请安装 sentence-transformers 并执行 python scripts/build_knowledge.py --rebuild"
            ),
        }


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = EmbeddingProvider()
    return _provider


def embedding_status() -> Dict[str, Any]:
    return get_embedding_provider().status()
