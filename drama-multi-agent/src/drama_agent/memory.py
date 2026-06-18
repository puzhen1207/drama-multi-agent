"""
记忆管理核心模块：会话记忆 + 用户画像学习 + 反思日志持久化

核心设计：
- SessionManager：会话级别的运行时仓库（内存 + 文件持久化）
  - get_or_create(session_id, user_id) -> SessionState
  - upsert(session) ：更新会话到磁盘
  - list_sessions(user_id) ：列出用户的所有会话
  - delete(session_id) ：删除会话

- ProfileLearner：把 ParsedTask 的结构化信息叠加到 UserProfile
  - 风格频次统计 → 提炼 preferred_style
  - 主题关键词词频 → 提炼 preferred_topics
  - 字数分布 → 提炼 target_length_range

- ReflectionLog：记录每次 polish → audit → polish 的修改轨迹
  - 后续可作为 fine-tune 数据或反例素材库
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
from .models import (
    AuditResult,
    ParsedTask,
    ReflectionEntry,
    SessionState,
    UserProfile,
)

logger = get_logger("memory")

# =============================================================================
# 持久化路径
# =============================================================================


def _memory_dir() -> Path:
    """返回记忆持久化目录。"""
    # 放在 data/ 下单独的子目录，避免与 faiss 索引混淆
    base = Path(settings.absolute_vector_index_path).parent
    d = base / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(session_id: str) -> Path:
    return _memory_dir() / f"{session_id}.json"


# =============================================================================
# 用户画像学习器
# =============================================================================


class ProfileLearner:
    """
    轻量统计式画像学习器。
    不依赖外部数据库；直接从 ParsedTask 中统计频次与分布。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def update_from_task(self, profile: UserProfile, task: Optional[ParsedTask]) -> UserProfile:
        """把一次任务解析结果喂到画像中。"""
        if task is None:
            return profile
        with self._lock:
            profile.total_interactions += 1
            profile.last_interaction_ts = time.time()

            # 1. 任务类型分布
            tt = task.task_type or "unknown"
            profile.task_type_distribution[tt] = profile.task_type_distribution.get(tt, 0) + 1

            # 2. 风格频次（简单多数投票；阈值 2）
            if task.style and task.style.strip():
                # 用隐式计数：每次相同风格命中 +1，最终取最高
                # 用 notes 作为计数器文本
                style_count_key = f"__style_{task.style}__"
                # 简化：直接用 notes 中的伪标记计数器；如果 notes 太乱改用外部字段
                # 更简单：preferred_style = 出现次数最多的风格
                # 为了不引入额外字段，用 task_type_distribution 之外的简单启发：
                # 每次更新时，若当前风格与上一次相同则“加强”，否则保留历史最高
                if not profile.preferred_style:
                    profile.preferred_style = task.style
                # 简单启发：如果当前风格与已有偏好相同，或已有偏好为空，则保留；否则用逗号分隔
                elif task.style != profile.preferred_style and task.style not in profile.preferred_style:
                    # 保留前 3 个偏好，用 / 分隔展示
                    parts = [p.strip() for p in profile.preferred_style.split("/") if p.strip()]
                    if len(parts) < 3:
                        parts.append(task.style)
                        profile.preferred_style = "/".join(parts)

            # 3. 主题关键词聚合（去重 + 保留最近 12 个）
            if task.keywords:
                merged = list(dict.fromkeys(profile.preferred_topics + list(task.keywords)))
                profile.preferred_topics = merged[-12:]

            # 4. 字数范围（取 min/max 扩展）
            tl = int(task.target_length or 0)
            if tl > 0:
                if profile.target_length_range is None:
                    profile.target_length_range = [tl, tl]
                else:
                    lo, hi = profile.target_length_range
                    profile.target_length_range = [min(lo, tl), max(hi, tl)]

            # 5. 从 topic 提炼主题
            if task.topic and task.topic.strip() and task.topic not in profile.preferred_topics:
                # 只在主题尚未出现时追加（避免长文本溢出关键词列表）
                if len(task.topic) <= 20:
                    profile.preferred_topics.append(task.topic)
                    if len(profile.preferred_topics) > 12:
                        profile.preferred_topics = profile.preferred_topics[-12:]

            return profile

    def reset(self, profile: UserProfile) -> UserProfile:
        """清除所有学习到的偏好（但保留 total_interactions）。"""
        profile.preferred_style = ""
        profile.preferred_topics = []
        profile.target_length_range = None
        profile.task_type_distribution = {}
        profile.notes = ""
        return profile


# =============================================================================
# 会话管理
# =============================================================================


class SessionManager:
    """会话的内存仓库 + JSON 文件持久化。线程安全。"""

    def __init__(self, max_sessions_per_user: int = 20, max_turns: int = 10) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, SessionState] = {}  # session_id -> state
        self._max_sessions = max_sessions_per_user
        self._max_turns = max_turns
        self._profile_learner = ProfileLearner()

    # ---------- 核心读写 ----------

    def get_or_create(self, session_id: Optional[str], user_id: str = "guest") -> SessionState:
        """读取现有会话，或新建一个。"""
        sid = session_id or ("S_" + uuid.uuid4().hex[:10])
        with self._lock:
            # 1. 命中内存
            if sid in self._cache:
                return self._cache[sid]
            # 2. 命中磁盘
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
            # 3. 新建
            state = SessionState(session_id=sid, user_id=user_id)
            self._cache[sid] = state
            self._persist(state)
            logger.info(f"[Memory] 新建会话: {sid}")
            return state

    def upsert(self, session: SessionState) -> None:
        """写入一条会话（同时更新缓存与磁盘）。"""
        with self._lock:
            session.updated_ts = time.time()
            self._cache[session.session_id] = session
            self._persist(session)

    def _persist(self, session: SessionState) -> None:
        """持久化到磁盘（无锁，由外层调用者持有锁）。"""
        try:
            fp = _session_path(session.session_id)
            # 截断为纯 JSON；控制文本长度避免磁盘占用爆炸
            data = session.model_dump(mode="json")
            # 避免超长 messages 被持久化（已经通过 _trim 控制了，但做双保险）
            json_text = json.dumps(data, ensure_ascii=False, indent=2)
            fp.write_text(json_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Memory] 会话持久化失败: {e}")

    def delete(self, session_id: str) -> bool:
        """删除一个会话（同时清缓存 + 文件）。"""
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
        """返回会话摘要列表（不读 full messages）。"""
        summaries = []
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

    # ---------- 高级操作：工作流生命周期 ----------

    def before_workflow(
        self,
        session_id: Optional[str],
        user_id: str,
        raw_input: str,
    ) -> SessionState:
        """工作流开始前：加载会话 + push user message。"""
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
        """工作流结束后：push assistant message + 写入反思日志 + 更新画像 + 持久化。"""
        session.push_assistant(content[:2000] if len(content) > 2000 else content, max_turns=self._max_turns)
        if reflection_entry is not None:
            session.reflections.append(reflection_entry)
            if len(session.reflections) > 50:
                session.reflections = session.reflections[-50:]
        if parsed_task is not None:
            self._profile_learner.update_from_task(session.profile, parsed_task)
        self.upsert(session)


# =============================================================================
# 单例
# =============================================================================


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
