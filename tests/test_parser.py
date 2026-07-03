"""任务解析 Agent 单元测试（不依赖 LLM）。"""
from __future__ import annotations

from drama_agent.agents.parser_agent import _rule_based_parse, run_parse


def test_rule_based_parse_copywriting():
    task = _rule_based_parse("写2版推广文案，推广都市新剧《错位人生》")
    assert task.task_type == "copywriting"
    assert task.needs_retrieval is True


def test_rule_based_parse_audit():
    task = _rule_based_parse("帮我检查这段剧本是否有违规内容")
    assert task.task_type == "audit"
    assert task.needs_retrieval is False


def test_rule_based_parse_qa():
    task = _rule_based_parse("短剧创作中哪些内容是红线？")
    assert task.task_type == "qa"


def test_rule_based_parse_target_length():
    task = _rule_based_parse("写一段300字的短剧开头")
    assert task.target_length == 300


def test_run_parse_stub_mode():
    state = {"raw_input": "给我整理一个关于霸总追妻的短剧大纲，分3集"}
    result = run_parse(state)
    assert "parsed_task" in result
    assert result["parsed_task"].task_type == "content_organize"
