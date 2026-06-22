"""全局共享 State 与数据模型。
定义短剧多智能体系统的核心数据结构，跨 Agent 流转字段均在此集中管理。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


TaskType = Literal["content_organize", "copywriting", "qa", "audit"]


class ParsedTask(BaseModel):
    """经过任务解析 Agent 处理后的结构化任务。"""

    task_type: TaskType = Field(default="copywriting")
    topic: str = ""
    style: str = "爽文"
    target_length: int = 500
    keywords: List[str] = Field(default_factory=list)
    needs_retrieval: bool = True
    requirements: str = ""
    raw_explanation: str = ""


class RetrievedMaterial(BaseModel):
    """单次召回素材。"""

    material_id: str = ""
    title: str = ""
    content: str = ""
    category: str = "unknown"
    score: float = 0.0
    source: str = "faiss"


class AuditIssue(BaseModel):
    level: Literal["forbidden", "warning", "suggestion"] = "warning"
    category: str = ""
    position: str = ""
    suggestion: str = ""


class AuditResult(BaseModel):
    passed: bool = False
    score: float = 0.0
    issues: List[AuditIssue] = Field(default_factory=list)
    summary: str = ""
    rule_engine_hit: bool = False
    degrade_mode: bool = False

    @field_validator("score", mode="before")
    @classmethod
    def _clip_score(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


# ===== 会话 / 用户画像（记忆模块）=====


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""
    ts: float = Field(default_factory=lambda: __import__("time").time())


class UserProfile(BaseModel):
    preferred_style: str = ""
    preferred_topics: List[str] = Field(default_factory=list)
    target_length_range: Optional[List[int]] = None
    task_type_distribution: Dict[str, int] = Field(default_factory=dict)
    total_interactions: int = 0
    last_interaction_ts: Optional[float] = None
    notes: str = ""

    def summary_text(self) -> str:
        parts: List[str] = []
        if self.preferred_style:
            parts.append(f"- 常用风格：{self.preferred_style}")
        if self.preferred_topics:
            parts.append(f"- 关注主题：{', '.join(self.preferred_topics[:8])}")
        if self.target_length_range:
            parts.append(f"- 偏好字数：{self.target_length_range[0]}~{self.target_length_range[1]}")
        if self.task_type_distribution:
            top = sorted(self.task_type_distribution.items(), key=lambda kv: kv[1], reverse=True)[:3]
            parts.append(f"- 历史任务分布：{', '.join([f'{k}={v}' for k, v in top])}")
        if self.notes:
            parts.append(f"- 笔记：{self.notes}")
        if not parts:
            return "新用户，无明显偏好"
        return "\n".join(parts)


class ReflectionEntry(BaseModel):
    session_id: str = ""
    original_content: str = ""
    revision_content: str = ""
    audit_score_before: float = 0.0
    audit_score_after: float = 0.0
    issues_found: List[str] = Field(default_factory=list)
    iteration: int = 0
    ts: float = Field(default_factory=lambda: __import__("time").time())

    def short_repr(self) -> str:
        return (f"[iter={self.iteration}, score {self.audit_score_before:.2f}→"
                f"{self.audit_score_after:.2f}] 问题数={len(self.issues_found)}")


class SessionState(BaseModel):
    """会话级状态：包含多轮对话 + 用户画像 + 反思日志。"""

    session_id: str = ""
    user_id: str = "guest"
    messages: List[ChatMessage] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)
    reflections: List[ReflectionEntry] = Field(default_factory=list)
    created_ts: float = Field(default_factory=lambda: __import__("time").time())
    updated_ts: float = Field(default_factory=lambda: __import__("time").time())

    def push_user(self, content: str, max_turns: int = 10) -> None:
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim(max_turns)

    def push_assistant(self, content: str, max_turns: int = 10) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))
        self._trim(max_turns)

    def _trim(self, max_turns: int) -> None:
        if len(self.messages) <= max_turns:
            return
        rest = self.messages[-max_turns:]
        self.messages = rest

    def last_assistant(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def context_summary(self) -> str:
        if not self.messages:
            return ""
        lines = []
        recent = self.messages[-6:]
        for msg in recent:
            if msg.role == "system":
                continue
            prefix = "用户" if msg.role == "user" else "助理"
            snippet = msg.content[:200] if len(msg.content) <= 200 else msg.content[:200] + "…"
            lines.append(f"- [{prefix}] {snippet}")
        return "\n".join(lines)


class FinalResponse(BaseModel):
    """对外统一返回结构。"""

    success: bool = False
    task_type: Optional[str] = None
    content: str = ""
    audit_result: Optional[AuditResult] = None
    iteration_count: int = 0
    degrade_mode: bool = False
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    session_id: Optional[str] = None
    has_context: bool = False
    user_profile_summary: Optional[str] = None


class WorkflowState(BaseModel):
    """全链路共享 state（不使用 LangGraph 的 Annotated 机制，便于纯手动调度）。"""

    model_config = {"arbitrary_types_allowed": True}

    raw_input: str = ""
    user_id: str = "guest"
    session_id: Optional[str] = None

    # 记忆
    session_context: str = ""
    user_profile_text: str = ""

    # 解析
    parsed_task: Optional[ParsedTask] = None

    # 素材
    retrieved_materials: List[Dict[str, Any]] = Field(default_factory=list)

    # 内容
    draft_content: str = ""

    # 审核
    audit_result: Optional[AuditResult] = None
    iteration_count: int = 0
    max_iteration: int = 3

    # 异常与降级
    error_info: str = ""
    degrade_mode: bool = False
    node_failed: str = ""
