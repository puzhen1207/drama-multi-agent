"""合规规则引擎单元测试。"""
from __future__ import annotations

from drama_agent.tools.compliance_engine import tool_sensitive_check


def test_clean_text_passes():
    result = tool_sensitive_check("这是一段正常的短剧文案，讲述都市爱情故事。")
    assert result["passed_rule"] is True
    assert len(result["forbidden"]) == 0


def test_forbidden_keyword_detected():
    result = tool_sensitive_check("这段内容涉及色情低俗描写。")
    assert result["passed_rule"] is False
    assert any(h["level"] == "forbidden" for h in result["forbidden"])


def test_phone_number_detected():
    result = tool_sensitive_check("请联系我：13800138000")
    assert result["passed_rule"] is False
