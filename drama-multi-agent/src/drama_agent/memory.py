"""记忆管理核心模块：会话记忆 + 用户画像学习 + 反思日志持久化。

核心设计：
- SessionManager：会话级运行时仓库（内存 + JSON 文件持久化）。
- ProfileLearner：把 ParsedTask 的结构化信息叠加到 UserProfile。
- ReflectionEntry：记录每次 polish → audit → polish 的修改轨迹。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings
from .logging_setup import get_logger
from .models import AuditResult, ParsedTask, ReflectionEntry, SessionState, UserProfile

logger = get_logger("memory")


def _memory_dir() -> Path:
    base = Path(settings.absolute_vector_index_path).parent
    d = base / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(session_id: str) -> Path:
    return _memory_dir() / f"{session_id}.json"


# ============= 用户画像学习 =============


class ProfileLearner:
    """轻量统计式画像学习器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def update_from_task(self, profile: UserProfile, task: Optional[ParsedTask]) -> UserProfile:
        if task is None:
            return profile
        with self._lock:
            profile.total_interactions += 1
            profile.last_interaction_ts = time.time()

            tt = task.task_type or "unknown"
            profile.task_type_distribution[tt] = profile.task_type_distribution.get(tt, 0) + 1

            if task.style and task.style.strip():
                if not profile.preferred_style:
                    profile.preferred_style = task.style
                elif task.style != profile.preferred_style and task.style not in profile.preferred_style:
                    parts = [p.strip() for p in profile.preferred_style.split("/") if p.strip()]
                    if len(parts) < 3:
                        parts.append(task.style)
                        profile.preferred_style = "/".join(parts)

            if task.keywords:
                merged = list(dict.fromkeys(list(profile.preferred_topics) + list(task.keywords)))
                profile.preferred_topics = merged[-12:]

            tl = int(task.target_length or 0)
            if tl > 0:
                if profile.target_length_range is None:
                    profile.target_length_range = [tl, tl]
                else:
                    lo, hi = profile.target_length_range
                    profile.target_length_range = [min(lo, tl), max(hi, tl)]

            if task.topic and task.topic.strip() and task.topic not in profile.preferred_topics:
                if len(task.topic) <= 20:
                    profile.preferred_topics.append(task.topic)
                    if len(profile.preferred_topics) > 12:
                        profile.preferred_topics = profile.preferred_topics[-12:]

            return profile

    def reset(self, profile: UserProfile) -> UserProfile:
        profile.preferred_style = ""
        profile.preferred_topics = []
        profile.target_length_range = None
        profile.task_type_distribution = {}
        profile.notes = ""
        return profile


# ============= 会话管理 =============


class SessionManager:
    """会话的内存仓库 + JSON 文件持久化。线程安全。"""

    def __init__(self, max_sessions_per_user: int = 20, max_turns: int = 10) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, SessionState] = {}
        self._max_sessions = max_sessions_per_user
        self._max_turns = max_turns
        self._profile_learner = ProfileLearner()

    def get_or_create(self, session_id: Optional[str], user_id: str = "guest") -> SessionState:
        sid = session_id or ("S_" + uuid.uuid4().hex[:10])
        with self._lock:
            if sid in self._cache:
                return self._cache[sid]
            fp = _session_path(sid)
            if fp.exists():
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    state = SessionState.model_validate(data)
                    self._cache[sid] = state
                    logger.info(f"[Memory] 从磁盘加载会话: {sid}")
                    return state
                except Exception as e:
                    logger.warning(f"[Memory] 会话文件损坏，重建: {e}")
            state = SessionState(session_id=sid, user_id=user_id)
            self._cache[sid] = state
            self._persist(state)
            logger.info(f"[Memory] 新建会话: {sid}")
            return state

    def upsert(self, session: SessionState) -> None:
        with self._lock:
            session.updated_ts = time.time()
            self._cache[session.session_id] = session
            self._persist(session)

    def _persist(self, session: SessionState) -> None:
        try:
            fp = _session_path(session.session_id)
            data = session.model_dump(mode="json")
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Memory] 会话持久化失败: {e}")

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = False
            if session_id in self._cache:
                del self._cache[session_id]
                existed = True
            fp = _session_path(session_id)
            if fp.exists():
                try:
                    fp.unlink()
                    existed = True
                except Exception:
                    pass
            return existed

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        summaries: List[Dict] = []
        md = _memory_dir()
        for fp in md.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if user_id and data.get("user_id") != user_id:
                    continue
                summaries.append({
                    "session_id": data.get("session_id"),
                    "user_id": data.get("user_id", "guest"),
                    "message_count": len(data.get("messages", [])),
                    "reflection_count": len(data.get("reflections", [])),
                    "created_ts": data.get("created_ts"),
                    "updated_ts": data.get("updated_ts"),
                    "profile_summary": data.get("profile", {}),
                })
            except Exception:
                continue
        summaries.sort(key=lambda x: x.get("updated_ts", 0), reverse=True)
        return summaries

    # --------- 工作流生命周期钩子 ---------

    def before_workflow(
        self, session_id: Optional[str], user_id: str, raw_input: str
    ) -> SessionState:
        session = self.get_or_create(session_id, user_id)
        session.push_user(raw_input, max_turns=self._max_turns)
        return session

    def after_workflow(
        self,
        session: SessionState,
        content: str,
        audit_result: Optional[AuditResult],
        parsed_task: Optional[ParsedTask],
        reflection_entry: Optional[ReflectionEntry] = None,
    ) -> None:
        session.push_assistant(content[:2000] if len(content) > 2000 else content,
                                max_turns=self._max_turns)
        if reflection_entry is not None:
            session.reflections.append(reflection_entry)
            if len(session.reflections) > 50:
                session.reflections = session.reflections[-50:]
        if parsed_task is not None:
            self._profile_learner.update_from_task(session.profile, parsed_task)
        self.upsert(session)


# ============= 单例 =============


_session_manager: Optional[SessionManager] = None
_sm_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        with _sm_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
                logger.info("[Memory] SessionManager 已初始化")
    return _session_manager
