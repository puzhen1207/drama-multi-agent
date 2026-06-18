"""短剧多智能体系统 — API 快速测试脚本"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

API_BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"

TEST_CASES = [
    ("内容整理", "给我整理一段关于「霸总追妻」的短剧大纲，分 3 集，每集 400 字"),
    ("文案生成", "写 2 版不同风格的推广文案，推广都市新剧《错位人生》"),
    ("资料答疑", "短剧创作中不允许出现哪些内容？列出主要合规要求"),
]


def health_check():
    print("=" * 70)
    print("1. 健康检查:", API_BASE + "/health")
    try:
        with urllib.request.urlopen(API_BASE + "/health", timeout=5) as r:
            print("   OK:", json.loads(r.read().decode()))
            return True
    except Exception as e:
        print("   FAIL:", e)
        return False


def list_tools():
    print("=" * 70)
    print("2. 已注册 MCP 工具:")
    try:
        with urllib.request.urlopen(API_BASE + "/v1/tools", timeout=5) as r:
            tools = json.loads(r.read().decode())["tools"]
            for t in tools:
                print(f"   • {t}")
    except Exception as e:
        print("   FAIL:", e)


def run_sync(raw_input: str, user_id: str) -> dict:
    body = json.dumps({"raw_input": raw_input, "user_id": user_id}).encode()
    req = urllib.request.Request(
        API_BASE + "/v1/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def run_stream(raw_input: str, user_id: str):
    print("=" * 70)
    print("3. 流式生成 (SSE, 可视化事件):", raw_input[:60])
    body = json.dumps({"raw_input": raw_input, "user_id": user_id}).encode()
    req = urllib.request.Request(
        API_BASE + "/v1/stream", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    event_count = 0
    nodes_done = []
    content = ""
    with urllib.request.urlopen(req, timeout=180) as r:
        reader = __import__("io").TextIOWrapper(r, encoding="utf-8")
        buffer = ""
        for chunk in reader:
            buffer += chunk
            if "\n\n" in buffer:
                parts = buffer.split("\n\n")
                buffer = parts.pop()
                for part in parts:
                    if not part.strip():
                        continue
                    lines = part.strip().split("\n")
                    event_type = None
                    data_str = ""
                    for line in lines:
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str += line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        print("   [raw]", data_str[:80])
                        continue
                    event_count += 1
                    et = event_type or event.get("type", "?")
                    if et == "node_start":
                        print(f"   ▶ 开始 {event.get('node')}")
                    elif et == "node_done":
                        node = event.get("node")
                        nodes_done.append(node)
                        print(f"   ✅ {node} 完成 · {event.get('duration_ms')}ms"
                              + ((" · " + event.get("summary", "")[:60]) if event.get("summary") else ""))
                    elif et == "node_error":
                        print(f"   ❌ {event.get('node')} ERROR · {event.get('error', '')[:80]}")
                    elif et == "final":
                        data = event.get("data", {})
                        content = data.get("content", "")
                        audit = data.get("audit_result") or {}
                        print(f"   🏁 最终结果: {len(content)}字, "
                              f"task={data.get('task_type')}, "
                              f"audit={audit.get('passed')}, "
                              f"score={audit.get('score')}, "
                              f"iter={data.get('iteration_count')}")
                    elif et in ("start", "workflow_start"):
                        print(f"   🚀 工作流启动: {event.get('input', '')[:50]}")
                    elif et in ("done", "workflow_done"):
                        print(f"   ⏱ 工作流完成: {event.get('elapsed_ms')}ms")
                    elif et == "error":
                        print(f"   ⚠ ERROR: {event.get('message', event.get('error', ''))[:80]}")
                    else:
                        print(f"   [{et}]", json.dumps(event, ensure_ascii=False)[:80])

    dt = time.time() - t0
    print(f"\n   总计 {event_count} 个事件，总耗时 {dt:.1f}s，完成节点: {', '.join(nodes_done)}")
    if content:
        preview = content[:200].replace("\n", "\n      ")
        print(f"   内容预览:\n      {preview}...")


def main():
    print("🎬 短剧多智能体内容生产系统 — API 快速测试")
    print("   服务地址:", API_BASE)

    if not health_check():
        return

    list_tools()

    for label, prompt in TEST_CASES:
        print()
        print("=" * 70)
        print(f"场景 [{label}]:", prompt)
        run_stream(prompt, label)


if __name__ == "__main__":
    main()
