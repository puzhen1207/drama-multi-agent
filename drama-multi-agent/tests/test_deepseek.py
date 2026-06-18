"""DeepSeek API 连通性 & 完整工作流测试"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

import urllib.request, urllib.error

from drama_agent.config import settings
from drama_agent.graph import run_workflow
from drama_agent.logging_setup import setup_logging

setup_logging()


def test_llm_http():
    """测试 DeepSeek 直接 HTTP 调用"""
    print("=" * 70)
    print("1. DeepSeek API 连通性测试")
    print("=" * 70)
    print(f"   模型:     {settings.llm_model}")
    print(f"   Base URL: {settings.llm_base_url}")
    print(f"   API Key:  {'✓ 已配置' if settings.llm_api_key else '✗ 未配置'}")
    print()

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是一位短剧创作助手，用简洁中文回答。"},
            {"role": "user", "content": "用 30 字以内介绍一下什么是短剧爽文。"},
        ],
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as resp:
            dt = time.time() - t0
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            print(f"   ✅ 连接成功！耗时 {dt:.1f}s")
            print(f"   模型回答: {content[:200]}")
            return True
    except urllib.error.HTTPError as e:
        print(f"   ❌ HTTP {e.code}: {e.read().decode()[:500]}")
        return False
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False


def test_full_workflow():
    """测试完整工作流：解析 → 检索 → 润色 → 审核 → 迭代"""
    print()
    print("=" * 70)
    print("2. 完整多智能体工作流测试")
    print("=" * 70)

    cases = [
        ("内容整理", "给我整理一段关于「霸总追妻」的小说大纲，分 3 集，每集 400 字"),
        ("文案生成", "写 2 版不同风格的推广文案，推广都市新剧《错位人生》"),
        ("资料答疑", "短剧内容中不允许出现哪些内容？请列出主要合规要求"),
    ]
    results = []
    for i, (label, case) in enumerate(cases, 1):
        print(f"\n▶ [{i}] {label}: {case}")
        t0 = time.time()
        r = run_workflow(case, user_id="deepseek-test")
        dt = time.time() - t0
        print(f"   任务类型: {r.task_type}")
        print(f"   内容长度: {len(r.content)} 字")
        if r.audit_result:
            a = r.audit_result
            status = "✅ PASS" if a.passed else "⚠ 已修正"
            print(f"   审核结果: {status} score={a.score}, issues={len(a.issues)}, degrade={a.degrade_mode}")
        print(f"   总迭代: {r.iteration_count} 次, 总耗时: {dt:.1f}s")
        print(f"   降级模式: {'是' if r.degrade_mode else '否'}")
        preview = r.content[:80].replace("\n", " ")
        print(f"   内容预览: {preview}...")
        results.append(r.success or len(r.content) > 100)

    print()
    print("=" * 70)
    print(f"总结: {sum(results)}/{len(results)} 个场景成功生成有意义的内容")
    print("=" * 70)
    return all(results)


def main():
    ok = test_llm_http()
    if ok:
        test_full_workflow()
    else:
        print("\n⚠ LLM 连接失败，请检查 API Key 或网络")
        sys.exit(1)


if __name__ == "__main__":
    main()
