"""个人记忆库 —— 存储用户确认的 Q&A，按问题相似度召回供生成参考。"""
from __future__ import annotations

import pickle
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import settings
from ..logging_setup import get_logger
from ..models import RetrievedMaterial
from .vector_retriever import _faiss_load_index, _faiss_save_index, _light_rerank
from .embedding import get_embedding_provider

logger = get_logger("user_memory")


class UserMemoryStore:
    """按 user_id 隔离的个人 Q&A 向量记忆库。"""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = Path(store_dir or settings.absolute_user_memory_path)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "faiss.index"
        self.meta_path = self.store_dir / "metadata.pkl"
        self.embedding = get_embedding_provider()
        self.faiss_index = None
        self.entries: List[dict] = []
        self._lock = threading.Lock()
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.faiss_index = _faiss_load_index(self.index_path)
                with open(str(self.meta_path), "rb") as f:
                    data = pickle.load(f)
                self.entries = data.get("entries", [])
                logger.info(f"[UserMemory] 已加载 {len(self.entries)} 条个人记忆")
                return
            except Exception as e:
                logger.warning(f"[UserMemory] 加载失败，重建：{e}")
        self.entries = []
        self.faiss_index = None
        if not hasattr(self, "_numpy_matrix"):
            self._numpy_matrix = None

    def save(self) -> None:
        if self.faiss_index is not None:
            try:
                _faiss_save_index(self.faiss_index, self.index_path)
            except Exception as e:
                logger.warning(f"[UserMemory] 保存索引失败：{e}")
        tmp = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        with open(str(tmp), "wb") as f:
            pickle.dump({"entries": self.entries, "dim": self.embedding.dim}, f)
        tmp.replace(str(self.meta_path))

    def add(
        self,
        user_id: str,
        question: str,
        answer: str,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if len(question) < 2:
            raise ValueError("问题过短，无法保存")
        if len(answer) < 20:
            raise ValueError("回答过短，无法保存（至少 20 字）")

        memory_id = "M_" + uuid.uuid4().hex[:10]
        entry = {
            "memory_id": memory_id,
            "user_id": user_id or "guest",
            "question": question,
            "answer": answer,
            "title": (title or question[:40]).strip(),
            "session_id": session_id or "",
            "created_ts": time.time(),
            "updated_ts": time.time(),
        }
        vec = self.embedding.encode_one(question).astype(np.float32).reshape(1, -1)

        with self._lock:
            self.entries.append(entry)
            self._add_vector(vec)
            self.save()

        logger.info(f"[UserMemory] 已保存 memory_id={memory_id}, user={user_id}")
        return memory_id

    def _add_vector(self, vec: np.ndarray) -> None:
        try:
            import faiss  # type: ignore

            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(int(self.embedding.dim))
            self.faiss_index.add(vec.astype(np.float32))
            return
        except Exception as e:
            logger.warning(f"[UserMemory] faiss 添加失败：{e}")
            self.faiss_index = None

        matrix = getattr(self, "_numpy_matrix", None)
        if matrix is None:
            self._numpy_matrix = vec.astype(np.float32)
        else:
            self._numpy_matrix = np.vstack([matrix, vec.astype(np.float32)])

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 2,
        min_score: float = 0.15,
    ) -> List[RetrievedMaterial]:
        if not self.entries:
            return []

        user_entries = [
            (i, e) for i, e in enumerate(self.entries)
            if e.get("user_id") == (user_id or "guest")
        ]
        if not user_entries:
            return []

        qvec = self.embedding.encode_one(query).astype(np.float32).reshape(1, -1)
        k_search = min(len(self.entries), max(top_k * 8, 16))
        scores, indices = self._search_impl(qvec, k_search)

        results: List[RetrievedMaterial] = []
        user_index_set = {idx for idx, _ in user_entries}

        for s, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.entries) or idx not in user_index_set:
                continue
            entry = self.entries[idx]
            rerank = _light_rerank(query, entry["question"] + " " + entry["answer"])
            norm_s = float(max(0.0, min(1.0, (float(s) + 1.0) / 2.0)))
            score = round(max(norm_s * 0.4 + rerank * 0.6, rerank), 3)
            if score < min_score:
                continue
            content = (
                f"【历史提问】{entry['question']}\n"
                f"【历史回答】{entry['answer']}"
            )
            results.append(RetrievedMaterial(
                material_id=entry["memory_id"],
                title=f"个人记忆 · {entry.get('title', '')[:30]}",
                content=content,
                category="个人记忆",
                score=score,
                source="user_memory",
            ))

        results.sort(key=lambda m: m.score, reverse=True)
        final = results[:top_k]
        if final:
            logger.info(f"[UserMemory] 召回 {len(final)} 条个人记忆（user={user_id}）")
        return final

    def _search_impl(self, qvec: np.ndarray, k: int):
        total = len(self.entries)
        k = min(k, total)
        if self.faiss_index is not None:
            try:
                return self.faiss_index.search(qvec, k)
            except Exception:
                pass
        matrix = getattr(self, "_numpy_matrix", None)
        if matrix is not None and len(matrix) > 0:
            sims = (matrix @ qvec.T).flatten()
            top_idx = np.argsort(-sims)[:k]
            return sims[top_idx].reshape(1, -1), top_idx.reshape(1, -1)
        return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=np.int64)

    def _entry_to_dict(self, e: dict, full: bool = True) -> Dict[str, Any]:
        item = {
            "memory_id": e["memory_id"],
            "title": e.get("title", ""),
            "question": e.get("question", ""),
            "answer": e.get("answer", "") if full else e.get("answer", "")[:120],
            "created_ts": e.get("created_ts"),
            "updated_ts": e.get("updated_ts", e.get("created_ts")),
            "session_id": e.get("session_id", ""),
        }
        if not full:
            item["question"] = item["question"][:200]
            item["answer_preview"] = item["answer"]
            del item["answer"]
        return item

    def get(self, memory_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        uid = user_id or "guest"
        for e in self.entries:
            if e.get("memory_id") == memory_id and e.get("user_id") == uid:
                return self._entry_to_dict(e, full=True)
        return None

    def list_memories(self, user_id: str, limit: int = 50, full: bool = True) -> List[Dict[str, Any]]:
        uid = user_id or "guest"
        items = [e for e in self.entries if e.get("user_id") == uid]
        items.sort(key=lambda x: x.get("updated_ts", x.get("created_ts", 0)), reverse=True)
        return [self._entry_to_dict(e, full=full) for e in items[:limit]]

    def update(
        self,
        memory_id: str,
        user_id: str,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        title: Optional[str] = None,
    ) -> bool:
        uid = user_id or "guest"
        question_changed = False
        with self._lock:
            target = None
            for e in self.entries:
                if e.get("memory_id") == memory_id and e.get("user_id") == uid:
                    target = e
                    break
            if target is None:
                return False

            if question is not None:
                q = question.strip()
                if len(q) < 2:
                    raise ValueError("问题过短，无法保存")
                if q != target.get("question"):
                    question_changed = True
                target["question"] = q
            if answer is not None:
                a = answer.strip()
                if len(a) < 20:
                    raise ValueError("回答过短，无法保存（至少 20 字）")
                target["answer"] = a
            if title is not None:
                target["title"] = title.strip() or target["question"][:40]
            elif question is not None and not target.get("title"):
                target["title"] = target["question"][:40]

            target["updated_ts"] = time.time()

            if question_changed:
                self._rebuild_index()
            self.save()

        logger.info(f"[UserMemory] 已更新 memory_id={memory_id}")
        return True

    def delete(self, memory_id: str, user_id: str) -> bool:
        uid = user_id or "guest"
        with self._lock:
            idx_to_remove = None
            for i, e in enumerate(self.entries):
                if e.get("memory_id") == memory_id and e.get("user_id") == uid:
                    idx_to_remove = i
                    break
            if idx_to_remove is None:
                return False
            self.entries.pop(idx_to_remove)
            self._rebuild_index()
            self.save()
        logger.info(f"[UserMemory] 已删除 memory_id={memory_id}")
        return True

    def _rebuild_index(self) -> None:
        """删除条目后重建向量索引。"""
        self.faiss_index = None
        self._numpy_matrix = None
        if not self.entries:
            return
        questions = [e["question"] for e in self.entries]
        vecs = self.embedding.encode(questions)
        for i in range(len(vecs)):
            self._add_vector(vecs[i : i + 1])

    def export_all(self, user_id: str) -> List[Dict[str, Any]]:
        uid = user_id or "guest"
        return [
            dict(e) for e in self.entries
            if e.get("user_id") == uid
        ]

    def import_entries(
        self,
        user_id: str,
        items: List[Dict[str, Any]],
        skip_duplicates: bool = True,
    ) -> Dict[str, int]:
        uid = user_id or "guest"
        imported = 0
        skipped = 0
        with self._lock:
            existing_ids = {e.get("memory_id") for e in self.entries if e.get("user_id") == uid}
            for raw in items:
                if not isinstance(raw, dict):
                    skipped += 1
                    continue
                question = (raw.get("question") or "").strip()
                answer = (raw.get("answer") or "").strip()
                if len(question) < 2 or len(answer) < 20:
                    skipped += 1
                    continue
                mid = raw.get("memory_id") or ("M_" + uuid.uuid4().hex[:10])
                if skip_duplicates and mid in existing_ids:
                    skipped += 1
                    continue
                entry = {
                    "memory_id": mid,
                    "user_id": uid,
                    "question": question,
                    "answer": answer,
                    "title": (raw.get("title") or question[:40]).strip(),
                    "session_id": raw.get("session_id") or "",
                    "created_ts": float(raw.get("created_ts") or time.time()),
                    "updated_ts": float(raw.get("updated_ts") or time.time()),
                }
                self.entries.append(entry)
                existing_ids.add(mid)
                imported += 1
            if imported:
                self._rebuild_index()
                self.save()
        logger.info(f"[UserMemory] 导入完成 user={uid}, imported={imported}, skipped={skipped}")
        return {"imported": imported, "skipped": skipped, "total": self.count(uid)}

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            return sum(1 for e in self.entries if e.get("user_id") == user_id)
        return len(self.entries)


_user_memory_store: Optional[UserMemoryStore] = None
_store_lock = threading.Lock()


def get_user_memory_store() -> UserMemoryStore:
    global _user_memory_store
    if _user_memory_store is None:
        with _store_lock:
            if _user_memory_store is None:
                _user_memory_store = UserMemoryStore()
    return _user_memory_store


def search_user_memory(user_id: str, query: str, top_k: Optional[int] = None) -> List[dict]:
    k = top_k or settings.user_memory_top_k
    materials = get_user_memory_store().search(user_id, query, top_k=k)
    return [m.model_dump() for m in materials]
