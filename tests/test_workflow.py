"""工作流集成测试（Stub 模式，无需 LLM API Key）。"""
from __future__ import annotations

from drama_agent.graph import run_workflow, run_workflow_with_events
from drama_agent.models import FinalResponse


def test_run_workflow_returns_content():
    resp = run_workflow("写一段关于重生80年代当首富的剧情，300字", user_id="pytest")
    assert isinstance(resp, FinalResponse)
    assert resp.content
    assert len(resp.content) > 50


def test_run_workflow_with_events():
    events: list = []
    resp = run_workflow_with_events(
        "帮我整理霸总追妻短剧大纲",
        user_id="pytest",
        event_callback=lambda ev: events.append(ev),
    )
    assert resp.content
    types = {e.get("type") for e in events}
    assert "node_start" in types or "workflow_start" in types
    assert any(e.get("type") == "node_done" for e in events)


def test_session_persistence():
    resp1 = run_workflow("第一集：开场冲突", user_id="session_test")
    sid = resp1.session_id
    assert sid

    resp2 = run_workflow(
        "继续写第二集",
        user_id="session_test",
        session_id=sid,
    )
    assert resp2.session_id == sid
    assert resp2.has_context is True
