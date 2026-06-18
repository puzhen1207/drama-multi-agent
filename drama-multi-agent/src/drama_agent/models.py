"""
全局共享 State 与数据模型
定义短剧多智能体系统的核心数据结构，跨 Agent 流转字段均在此集中管理。
"""
from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

# =============================================================================
# 任务类型与子结构
# =============================================================================

TaskType = Literal["content_organize", "copywriting", "qa", "audit"]


class ParsedTask(BaseModel):
    """经过任务解析 Agent 处理后的结构化任务。"""

    task_type: TaskType = Field(..., description="任务分类：content_organize|copywriting|qa|audit")
    topic: str = Field("", description="短剧核心主题 / 人物 / 剧情关键词")
    style: str = Field("爽文", description="风格偏好：爽文 / 虐恋 / 悬疑 / 甜宠 等")
    target_length: int = Field(500, ge=100, le=5000, description="期望输出字数")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    needs_retrieval: bool = Field(True, description="是否需要素材检索")
    requirements: str = Field("", description="用户额外要求（自由文本）")
    raw_explanation: str = Field("", description="模型解析说明，便于定位问题")


class RetrievedMaterial(BaseModel):
    """单次召回素材。"""

    material_id: str = Field(..., description="素材唯一 ID（父块 ID）")
    title: str = Field("", description="素材标题 / 章节名")
    content: str = Field(..., description="素材正文片段（父块全文，或上下文扩展后的内容）")
    category: str = Field("unknown", description="素材分类：剧本 / 文案 / 人设 / 规则")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Reranker 打分（0-1）")
    source: str = Field("faiss", description="来源：faiss / fallback / manual")


class AuditIssue(BaseModel):
    """单次审核问题。"""

    level: Literal["forbidden", "warning", "suggestion"] = Field(
        ..., description="问题等级：forbidden=禁止级 | warning=警告级 | suggestion=建议级"
    )
    category: str = Field("", description="违规类型：敏感词 / 政治 / 暴力 / 色情 / 其他")
    position: str = Field("", description="问题所在段落或关键词原文")
    suggestion: str = Field("", description="具体修改建议")


class AuditResult(BaseModel):
    """合规审核结果。"""

    passed: bool = Field(False, description="是否通过审核")
    score: float = Field(0.0, ge=0.0, le=1.0, description="整体合规分")
    issues: List[AuditIssue] = Field(default_factory=list, description="问题列表")
    summary: str = Field("", description="审核结论摘要，供润色 Agent 读取做定向修改")
    rule_engine_hit: bool = Field(False, description="规则引擎是否命中（硬违规）")
    degrade_mode: bool = Field(False, description="是否走降级审核（仅规则层）")

    @field_validator("score", mode="before")
    @classmethod
    def _clip_score(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0


# =============================================================================
# 记忆模块：会话 / 用户画像 / 反思日志
# =============================================================================


class ChatMessage(BaseModel):
    """多轮对话中的单条消息（与 OpenAI 兼容格式）。"""

    role: Literal["system", "user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    ts: float = Field(default_factory=lambda: __import__("time").time(), description="时间戳")


class UserProfile(BaseModel):
    """用户画像：从历史对话中提炼的稳定偏好。"""

    preferred_style: str = Field("", description="常用风格偏好（爽文/虐恋/悬疑/甜宠等）")
    preferred_topics: List[str] = Field(default_factory=list, description="高频主题")
    target_length_range: Optional[List[int]] = Field(default=None, description="典型字数范围")
    task_type_distribution: Dict[str, int] = Field(default_factory=dict, description="任务类型分布")
    total_interactions: int = Field(0, description="总对话次数")
    last_interaction_ts: Optional[float] = Field(default=None, description="最后一次互动时间")
    notes: str = Field("", description="自由文本的观察笔记（LLM 可读取）")

    def summary_text(self) -> str:
        """生成面向 LLM 的用户画像摘要。"""
        parts = []
        if self.preferred_style:
            parts.append(f"- 常用风格：{self.preferred_style}")
        if self.preferred_topics:
            parts.append(f"- 关注主题：{', '.join(self.preferred_topics[:8])}")
        if self.target_length_range:
            parts.append(f"- 偏好字数：{self.target_length_range[0]}~{self.target_length_range[1]}")
        if self.task_type_distribution:
            top_types = sorted(self.task_type_distribution.items(), key=lambda kv: kv[1], reverse=True)[:3]
            parts.append(f"- 历史任务分布：{', '.join([f'{k}={v}' for k, v in top_types])}")
        if self.notes:
            parts.append(f"- 笔记：{self.notes}")
        if not parts:
            return "新用户，无明显偏好"
        return "\n".join(parts)


class ReflectionEntry(BaseModel):
    """反思日志条目：记录一次 audit→polish 的修改轨迹。"""

    session_id: str = Field(..., description="所属会话")
    original_content: str = Field("", description="修改前的草稿（截断前 300 字）")
    revision_content: str = Field("", description="修改后的内容（截断前 300 字）")
    audit_score_before: float = Field(0.0, description="修改前审核分")
    audit_score_after: float = Field(0.0, description="修改后审核分")
    issues_found: List[str] = Field(default_factory=list, description="被修复的问题摘要")
    iteration: int = Field(0, description="迭代轮次")
    ts: float = Field(default_factory=lambda: __import__("time").time())

    def short_repr(self) -> str:
        return (f"[iter={self.iteration}, score {self.audit_score_before:.2f}→"
                f"{self.audit_score_after:.2f}] 问题数={len(self.issues_found)}")


class SessionState(BaseModel):
    """
    会话级别的运行时记忆：包含多轮对话上下文 + 用户画像 + 反思日志。
    每次请求都会读取当前会话的上下文，注入到工作流；结束后再回写。
    """

    session_id: str = Field(..., description="会话唯一 ID（外部传入或自动生成）")
    user_id: str = Field("guest", description="用户标识")
    messages: List[ChatMessage] = Field(default_factory=list, description="多轮对话上下文（最近 N 轮）")
    profile: UserProfile = Field(default_factory=UserProfile, description="用户画像（从历史中提炼）")
    reflections: List[ReflectionEntry] = Field(default_factory=list, description="反思日志")
    created_ts: float = Field(default_factory=lambda: __import__("time").time())
    updated_ts: float = Field(default_factory=lambda: __import__("time").time())

    # ---------- 便捷方法 ----------

    def push_user(self, content: str, max_turns: int = 10) -> None:
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim(max_turns)

    def push_assistant(self, content: str, max_turns: int = 10) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))
        self._trim(max_turns)

    def _trim(self, max_turns: int) -> None:
        """保持最近 N 条（system message 不动，user/assistant 成对保留）。"""
        if len(self.messages) <= max_turns:
            return
        # 保留 system（若首条是 system）
        system_msg: Optional[ChatMessage] = None
        rest = self.messages
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
            rest = self.messages[1:]
        # 只保留尾部 max_turns
        if len(rest) > max_turns:
            rest = rest[-max_turns:]
        self.messages = ([system_msg] if system_msg else []) + rest

    def last_assistant(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def context_summary(self) -> str:
        """给润色 Agent 读取的上下文摘要（仅取最近几轮对话）。"""
        if not self.messages:
            return ""
        lines = []
        recent = self.messages[-6:]  # 最近 6 条作为上下文注入
        for msg in recent:
            if msg.role == "system":
                continue
            prefix = "用户" if msg.role == "user" else "助理"
            snippet = msg.content[:200] if len(msg.content) <= 200 else msg.content[:200] + "..."
            lines.append(f"- [{prefix}] {snippet}")
        return "\n".join(lines)


# =============================================================================
# 全链路共享 State
# =============================================================================


def _concat_materials(
    existing: Optional[List[RetrievedMaterial]], new: Optional[List[RetrievedMaterial]]
) -> List[RetrievedMaterial]:
    """Runs on node output：两次检索结果合并去重。"""
    existing = existing or []
    new = new or []
    seen = {m.material_id for m in existing}
    merged = list(existing)
    for m in new:
        if m.material_id not in seen:
            merged.append(m)
            seen.add(m.material_id)
    return merged


ConcatMaterials = Annotated[List[RetrievedMaterial], _concat_materials]


class OverallState(TypedDict, total=False):
    """
    短剧多智能体系统全链路共享状态池。
    作为 LangGraph StateGraph 的统一 State Schema。
    """

    # 原始输入
    raw_input: str
    user_id: str
    session_id: Optional[str]           # 新增：会话 ID（来自 API）

    # 记忆（由 graph.py 读取/回写）
    session_context: str                # 上下文摘要字符串，注入润色 prompt
    user_profile_text: str              # 用户画像摘要

    # 解析
    parsed_task: ParsedTask

    # 素材
    retrieved_materials: ConcatMaterials  # 可追加写入
    need_more_retrieval: bool             # 润色 Agent 反向触发补充检索

    # 内容
    draft_content: str

    # 审核
    audit_result: AuditResult
    iteration_count: int
    max_iteration: int

    # 异常与降级
    error_info: str
    degrade_mode: bool
    node_failed: str                       # 最近失败的节点名

    # 元数据
    start_ts: float
    end_ts: float


class FinalResponse(BaseModel):
    """对外统一返回结构。"""

    success: bool
    task_type: Optional[str] = None
    content: str = ""
    audit_result: Optional[AuditResult] = None
    iteration_count: int = 0
    degrade_mode: bool = False
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    # 记忆扩展字段
    session_id: Optional[str] = None
    has_context: bool = False
    user_profile_summary: Optional[str] = None
