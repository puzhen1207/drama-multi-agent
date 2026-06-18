"""简单的 API 冒烟测试：启动 FastAPI 服务并测试端点"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
PORT = 8765


def main() -> int:
    print("=" * 60)
    print("FastAPI 服务测试")
    print("=" * 60)

    # 启动服务
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": ""}
    import os
    for k, v in os.environ.items():
        env.setdefault(k, v)

    cmd = [
        sys.executable,
        "-c",
        "import uvicorn; uvicorn.run('drama_agent.api:app',host='127.0.0.1',port=%d,log_level='warning')" % PORT,
    ]
    print(f"▶ 启动服务...", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(4)

    try:
        # 测试健康检查
        print("▶ GET /health")
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print("  ↓", data)
            assert data.get("status") == "ok"

        # 测试工具列表
        print("▶ GET /v1/tools")
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/tools")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print("  ↓ tools:", len(data.get("tools", [])), "个")
            assert len(data.get("tools", [])) > 0

        # 测试内容生成
        print("▶ POST /v1/generate")
        body = json.dumps({"raw_input": "给我整理一段关于霸总追妻的小说大纲", "user_id": "api-tester"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            print(f"  ↓ status={data.get('status')}, content_len={len(data.get('data',{}).get('content',''))}")
            assert data.get("status") == "ok"
            assert len(data.get("data", {}).get("content", "")) > 0

        print("\n" + "=" * 60)
        print("✓ 所有 API 测试通过")
        print("=" * 60)
        return 0
    finally:
        try:
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
