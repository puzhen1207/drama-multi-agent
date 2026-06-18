"""核心路由逻辑单元测试（不依赖 LLM）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drama_agent.graph import _route_after_parse, _route_after_audit, _route_after_polish  # noqa: E402
from drama_agent.agents.parser_agent import ParsedTask  # noqa: E402
from drama_agent.agents.audit_agent import AuditResult, AuditIssue  # noqa: E402


@pytest.fixture(autouse=True)
def _inject_root_env(monkeypatch):
    """确保 config.py 能解析到项目根目录。"""
    monkeypatch.chdir(ROOT)


def test_route_after_parse_needs_retrieval():
    state = {
        "parsed_task": ParsedTask(
            task_type="content_organize",
            topic="霸总追妻",
            keywords=["霸总", "追妻"],
            needs_retrieval=True,
        ),
        "degrade_mode": False,
    }
    assert _route_after_parse(state) == "retrieve"


def test_route_after_parse_no_retrieval():
    state = {
        "parsed_task": ParsedTask(
            task_type="qa",
            topic="合规要求",
            keywords=["合规"],
            needs_retrieval=False,
        ),
        "degrade_mode": False,
    }
    assert _route_after_parse(state) == "polish"


def test_route_after_parse_degrade_mode_skips_retrieve():
    state = {
        "parsed_task": ParsedTask(
            task_type="content_organize", topic="x", keywords=[],
            needs_retrieval=True,
        ),
        "degrade_mode": True,
    }
    assert _route_after_parse(state) == "polish"


def test_route_after_parse_none_task_is_polish():
    # 解析失败时，降级为纯生成
    assert _route_after_parse({"parsed_task": None}) == "polish"


def test_route_after_audit_passed_is_end():
    state = {
        "audit_result": AuditResult(passed=True, score=0.9, issues=[], degrade_mode=False),
        "iteration_count": 0,
        "max_iteration": 3,
    }
    assert _route_after_audit(state) == "end"


def test_route_after_audit_failed_can_iterate():
    state = {
        "audit_result": AuditResult(
            passed=False,
            score=0.4,
            issues=[
                AuditIssue(level="warning", category="敏感词", position="需要修改", suggestion="替换为合规词汇"),
            ],
            degrade_mode=False,
        ),
        "iteration_count": 0,
        "max_iteration": 3,
    }
    assert _route_after_audit(state) == "polish"


def test_route_after_audit_failed_but_at_limit_ends():
    state = {
        "audit_result": AuditResult(passed=False, score=0.2, issues=[], degrade_mode=False),
        "iteration_count": 3,
        "max_iteration": 3,
    }
    assert _route_after_audit(state) == "end"


def test_route_after_audit_none_result_is_end():
    assert _route_after_audit({"audit_result": None, "iteration_count": 0, "max_iteration": 3}) == "end"


def test_route_after_polish_no_retrieval_goes_to_audit():
    state = {"need_more_retrieval": False, "iteration_count": 0}
    assert _route_after_polish(state) == "audit"


def test_route_after_polish_with_retrieval_in_first_iteration():
    state = {"need_more_retrieval": True, "iteration_count": 0}
    assert _route_after_polish(state) == "retrieve"


def test_route_after_polish_with_retrieval_later_goes_to_audit():
    # 只有第 1 次迭代允许反向检索，之后走审核
    state = {"need_more_retrieval": True, "iteration_count": 1}
    assert _route_after_polish(state) == "audit"
