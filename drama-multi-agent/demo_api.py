"""快速 API 测试脚本"""
import json
import sys
import urllib.request

API = "http://127.0.0.1:8001"

def step1():
    """健康检查"""
    print("▶ 1. 健康检查")
    try:
        with urllib.request.urlopen(API + "/health", timeout=5) as r:
            data = json.loads(r.read().decode())
            print(f"   ✅ status={data.get('status')}, port={data.get('api_port')}")
            return True
    except Exception as e:
        print(f"   ❌ {e}")
        return False


def step2():
    """工具列表"""
    print("▶ 2. 已注册工具")
    try:
        with urllib.request.urlopen(API + "/v1/tools", timeout=5) as r:
            data = json.loads(r.read().decode())
            for t in data.get("tools", []):
                print(f"   • {t}")
            return True
    except Exception as e:
        print(f"   ❌ {e}")
        return False


def step3(case: str):
    """内容生成"""
    print(f"▶ 3. 内容生成: {case[:40]}...")
    body = json.dumps({"raw_input": case, "user_id": "live-demo"}).encode()
    req = urllib.request.Request(
        API + "/v1/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode())
            d = data["data"]
            print(f"   task_type: {d['task_type']}")
            print(f"   内容长度: {len(d['content'])} 字")
            if d.get("audit_result"):
                a = d["audit_result"]
                print(f"   审核: passed={a['passed']}, score={a['score']}, issues={len(a.get('issues', []))}")
            print(f"   迭代: {d['iteration_count']} 次, 耗时: {d['elapsed_ms']/1000:.1f}s")
            print(f"   降级: {'是' if d.get('degrade_mode') else '否'}")
            print()
            print("   --- 内容预览 ---")
            preview = d["content"][:400].replace("\n", "\n   ")
            print("   " + preview)
            print("   ...")
            return True
    except Exception as e:
        print(f"   ❌ {e}")
        return False


def main():
    print("=" * 70)
    print("短剧多智能体系统 · API 实时演示")
    print(f"服务地址: {API}")
    print(f"Swagger 文档: {API}/docs")
    print("=" * 70)
    print()

    if not step1():
        print("\n⚠ 服务未启动，请先运行: python -m uvicorn drama_agent.api:app --port 8001")
        sys.exit(1)
    print()
    step2()
    print()

    cases = [
        "给我整理一段关于「霸总追妻」的短剧大纲，分 3 集",
        "写 2 版不同风格的推广文案，推广都市新剧《错位人生》",
    ]
    for case in cases:
        step3(case)
        print()

    print("=" * 70)
    print(f"✅ 演示完成。完整 API 文档：{API}/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
