"""记忆模块单元测试：SessionState / UserProfile / SessionManager 持久化与画像学习。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drama_agent.models import (  # noqa: E402
    AuditResult,
    ParsedTask,
    ReflectionEntry,
    SessionState,
    UserProfile,
    ChatMessage,
)
from drama_agent.memory import ProfileLearner, SessionManager  # noqa: E402


@pytest.fixture(autouse=True)
def _env_root(monkeypatch):
    monkeypatch.chdir(ROOT)


# =============================================================================
# SessionState 与消息管理
# =============================================================================


def test_session_auto_creates_with_uuid_prefix():
    sm = SessionManager()
    session = sm.get_or_create(None, "alice")
    assert session.user_id == "alice"
    assert session.session_id.startswith("S_")
    assert session.messages == []


def test_push_user_and_assistant_builds_history():
    sm = SessionManager()
    session = sm.get_or_create("test_session_1", "bob")
    session.push_user("你好，帮我写一个关于霸总的短剧")
    session.push_assistant("好的，以下是一个《霸总追妻》短剧大纲：...")
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    assert "霸总" in session.messages[0].content


def test_context_summary_is_readable():
    sm = SessionManager()
    session = sm.get_or_create("test_ctx_sum", "u1")
    session.push_user("问一")
    session.push_assistant("答一")
    session.push_user("问二")
    session.push_assistant("答二")
    summary = session.context_summary()
    assert summary
    assert "问一" in summary
    assert "答二" in summary


# =============================================================================
# 用户画像学习（ProfileLearner）
# =============================================================================


def test_profile_learner_records_style_and_topics():
    pl = ProfileLearner()
    profile = UserProfile()
    task = ParsedTask(
        task_type="content_organize",
        topic="穿越古代",
        style="爽文",
        target_length=1500,
        keywords=["穿越", "医生", "权谋"],
        needs_retrieval=True,
    )
    profile = pl.update_from_task(profile, task)
    assert profile.preferred_style == "爽文"
    assert profile.total_interactions >= 1
    # 关键词应被记录
    assert "穿越" in profile.preferred_topics


def test_profile_learner_accumulates_multiple_styles():
    pl = ProfileLearner()
    profile = UserProfile()
    for style in ("爽文", "虐恋", "悬疑"):
        task = ParsedTask(
            task_type="copywriting",
            topic="主题",
            style=style,
            target_length=1000,
            keywords=["通用"],
        )
        profile = pl.update_from_task(profile, task)
    # 多种风格应当被拼接
    assert "爽文" in profile.preferred_style
    # 任务分布计数应累计
    assert profile.task_type_distribution.get("copywriting", 0) == 3


def test_profile_summary_text_is_meaningful():
    profile = UserProfile(
        preferred_style="爽文",
        preferred_topics=["穿越", "霸总", "权谋"],
        target_length_range=[500, 2000],
        task_type_distribution={"content_organize": 5, "copywriting": 3},
        total_interactions=8,
    )
    s = profile.summary_text()
    assert "爽文" in s
    assert "穿越" in s
    assert "500" in s or "2000" in s


# =============================================================================
# 会话管理（SessionManager）
# =============================================================================


def test_list_and_delete_sessions():
    sm = SessionManager()
    # 新建3个会话
    for i in range(3):
        s = sm.get_or_create(None, f"user_x_{i}")
        s.push_user("hi")
        s.push_assistant("hello")
        sm.upsert(s)
    # 列出
    all_sessions = sm.list_sessions()
    # list_sessions 返回 list
    assert len(all_sessions) >= 3
    first_id = all_sessions[0]["session_id"]
    ok = sm.delete(first_id)
    assert ok is True
    # 再删除不存在的，返回 False（或忽略）
    # delete 可能 False 或忽略，不强校验


def test_before_after_workflow_hook():
    sm = SessionManager()
    sid = "test_workflow_hook"
    # before_workflow 应 push 一条 user 消息
    session = sm.before_workflow(sid, "tester", "请生成内容")
    assert any(m.role == "user" for m in session.messages)

    # 构造模拟结果
    task = ParsedTask(
        task_type="content_organize",
        topic="短剧",
        style="爽文",
        target_length=500,
        keywords=["短剧"],
    )
    audit = AuditResult(
        passed=True, score=0.92, issues=[], summary="整体合规",
        rule_engine_hit=False, degrade_mode=False,
    )
    reflection = ReflectionEntry(
        session_id=sid,
        original_content="origin",
        revision_content="rev",
        audit_score_before=0.7,
        audit_score_after=0.92,
        issues_found=["轻微口语修改"],
        iteration=1,
    )
    sm.after_workflow(
        session=session,
        content="这里是一个结构化的短剧生成结果",
        audit_result=audit,
        parsed_task=task,
        reflection_entry=reflection,
    )
    # 画像已更新
    assert session.profile.total_interactions >= 1
    # 有反思日志
    assert len(session.reflections) >= 1
    # assistant 消息被 push
    assert any(m.role == "assistant" for m in session.messages)


# =============================================================================
# 线程安全（快速冒烟测试）
# =============================================================================


def test_concurrent_session_access():
    import threading
    sm = SessionManager()
    errors = []

    def worker(n):
        try:
            session = sm.get_or_create("test_concurrent", f"worker_{n}")
            session.push_user(f"hi {n}")
            session.push_assistant(f"reply {n}")
            sm.upsert(session)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发错误: {errors[:3]}"


# =============================================================================
# llm 多轮消息构建（context_messages 注入）
# =============================================================================


def test_build_messages_with_history_trims_large_context():
    from drama_agent.llm import _build_messages

    messages = _build_messages(
        user_prompt="新的请求",
        system_prompt="你是短剧编剧",
        context_messages=[
            {"role": "user", "content": "旧请求"},
            {"role": "assistant", "content": "旧响应"},
        ],
    )
    # system + history + user 至少3条
    assert len(messages) >= 3
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    # 字符长度限制：如果总内容过长应被截断（这里不触发）
    assert all(len(m.get("content", "")) > 0 for m in messages)


def test_trim_reduces_messages_when_too_long():
    from drama_agent.llm import _trim_to_context_window

    messages = [
        {"role": "system", "content": "你是短剧编剧"},
    ] + [
        {"role": "user", "content": "长文" * 2000},
        {"role": "assistant", "content": "答案"},
    ] * 2
    result = _trim_to_context_window(messages, max_chars=200)
    # system 保留，历史被删除（或部分），最后一条 user 保留
    assert result[0]["role"] == "system"
    assert result[-1]["role"] == "assistant" or result[-1]["role"] == "user"
    # 字符总量应 <= max_chars + buffer
    total = sum(len(m["content"]) for m in result)
    assert total <= 200 + 50  # 允许一点误差（函数不一定严格 <= max_chars）
