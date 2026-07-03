#!/usr/bin/env python3
"""API 快速测试脚本。

用法：
    python scripts/api_test.py
    python scripts/api_test.py http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def _post(base: str, path: str, payload: dict) -> dict:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base: str, path: str) -> dict:
    url = f"{base.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

    print(f"[1/4] GET /health @ {base}")
    health = _get(base, "/health")
    print(f"      status={health.get('status')} version={health.get('version')}")

    print("[2/4] GET /v1/tools")
    tools = _get(base, "/v1/tools")
    print(f"      tools={tools.get('tools')}")

    print("[3/4] POST /v1/generate（同步）")
    result = _post(base, "/v1/generate", {
        "raw_input": "写一段关于霸总追妻的短剧开头，300字",
        "user_id": "api_test",
    })
    data = result.get("data") or {}
    content = data.get("content", "")
    print(f"      status={result.get('status')} task_type={data.get('task_type')}")
    print(f"      content_len={len(content)} degrade={data.get('degrade_mode')}")
    if content:
        print(f"      preview: {content[:120]}...")

    print("[4/4] POST /v1/async/generate（异步）")
    async_resp = _post(base, "/v1/async/generate", {
        "raw_input": "短剧创作有哪些合规红线？",
        "user_id": "api_test",
    })
    task_id = async_resp.get("task_id")
    print(f"      task_id={task_id} status={async_resp.get('status')}")

    if task_id:
        import time
        for _ in range(60):
            status_resp = _get(base, f"/v1/async/{task_id}")
            if status_resp.get("status") != "pending":
                async_data = status_resp.get("data") or {}
                print(f"      async done: status={status_resp.get('status')} "
                      f"content_len={len(async_data.get('content', ''))}")
                break
            time.sleep(2)
        else:
            print("      [WARN] 异步任务超时")

    print("\n[OK] API 测试完成")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"\n[ERROR] 无法连接服务：{e}")
        print("请先启动服务：uvicorn drama_agent.api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
