"""
端到端 smoke test：验证 4 类核心场景是否可运行。
- `python tests/test_workflow.py` 运行命令行 smoke test
- `python -m pytest tests/test_workflow.py -v` 运行 pytest 测试
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drama_agent.graph import run_workflow  # noqa: E402
from drama_agent.logging_setup import setup_logging  # noqa: E402
from drama_agent.tools.vector_retriever import tool_retrieve_materials  # noqa: E402
from drama_agent.tools.compliance_engine import tool_sensitive_check  # noqa: E402

setup_logging()


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = {
    "content_organize": "给我整理一段关于「霸总追妻」的小说大纲，分 3 集，每集 400 字",
    "copywriting": "写 2 版不同风格的推广文案，用来推广我们的都市新剧《错位人生》",
    "qa": "短剧内容中是否允许出现血腥暴力场景？有哪些合规要求？",
    "qa_2": "生成一个关于「高考状元穿越古代」的爆款剧本，风格爽文，500 字",
}


# =============================================================================
# pytest 风格测试
# =============================================================================

def test_parser_content_organize():
    r = run_workflow(TEST_CASES["content_organize"], "pytest-org")
    assert r.content and len(r.content) > 50
    assert r.audit_result is not None


def test_parser_copywriting():
    r = run_workflow(TEST_CASES["copywriting"], "pytest-copy")
    assert r.content and len(r.content) > 50
    assert r.audit_result is not None


def test_qa_compliance():
    r = run_workflow(TEST_CASES["qa"], "pytest-qa")
    assert r.content and len(r.content) > 50
    assert r.audit_result is not None


def test_retrieve_materials_tool():
    results = tool_retrieve_materials("霸总追妻 小说 大纲", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "content" in item or "title" in item


def test_compliance_engine_clean():
    result = tool_sensitive_check("这是一段完全合规的内容，没有任何敏感词。")
    assert result["passed_rule"] is True


def test_compliance_engine_hit():
    result = tool_sensitive_check("这段内容包含赌博、色情、血腥、暴力等敏感词")
    assert len(result.get("forbidden", [])) > 0 or len(result.get("warning", [])) > 0


# =============================================================================
# 命令行 smoke test
# =============================================================================

def run_all() -> int:
    print("=" * 70)
    print("短剧多智能体系统 · 端到端 smoke test")
    print("=" * 70)

    results = []
    for name, case in TEST_CASES.items():
        print(f"\n▶ [{name}] {case}")
        resp = run_workflow(case, user_id="smoke-test")
        passed = bool(resp.content) and resp.elapsed_ms < 60_000
        print(
            f"  ✓ task_type={resp.task_type}, content_len={len(resp.content)}字, "
            f"耗时={resp.elapsed_ms:.0f}ms, degrade={resp.degrade_mode}"
        )
        if resp.audit_result:
            a = resp.audit_result
            print(
                f"  ✓ audit passed={a.passed}, score={a.score}, issues={len(a.issues)}, "
                f"degrade={a.degrade_mode}"
            )
        if resp.error:
            print(f"  ⚠ err: {resp.error}")
        results.append(passed)

    total_pass = sum(results)
    print("\n" + "=" * 70)
    print(f"结果: {total_pass}/{len(results)} 用例通过")
    print("=" * 70)
    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_all())
