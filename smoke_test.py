"""项目冒烟测试（零依赖核心模块导入 + 工作流跑一次）"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "src"
sys.path.insert(0, str(src))

# 1. 检查模块导入
from drama_agent.agents.parser_agent import run_parse
from drama_agent.agents.retriever_agent import run_retrieve
from drama_agent.agents.polish_agent import run_polish
from drama_agent.agents.audit_agent import run_audit
from drama_agent.graph import run_workflow, run_workflow_with_events
from drama_agent.models import FinalResponse, ParsedTask, AuditResult
print("[OK] 所有模块导入成功")

# 2. 跑一次完整工作流（不配置 LLM -> stub 模式）
print("\n===== 冒烟测试：run_workflow =====")
resp = run_workflow("帮我整理一个关于霸总追妻的短剧大纲，分 3 集",
                    user_id="pytest")
print("success:", resp.success)
print("task_type:", resp.task_type)
print("content 字数:", len(resp.content))
print("degrade_mode:", resp.degrade_mode)
print("audit_result.score:",
      resp.audit_result.score if resp.audit_result else None)
if resp.content:
    print("---- 内容预览（前 200 字）----")
    print(resp.content[:200])
    print("--------------------------------")

# 3. 检查事件流（SSE 模式）
print("\n===== 冒烟测试：run_workflow_with_events =====")
events = []
resp2 = run_workflow_with_events(
    "写一段关于重生 80 年代当首富的剧情，300 字",
    user_id="pytest",
    event_callback=lambda ev: events.append(ev),
)
print("event 数量:", len(events))
for ev in events[:6]:
    print(" ", ev.get("type"), "|",
          ev.get("node") or ev.get("input") or ev.get("summary") or "")
print("final content 字数:", len(resp2.content))
assert resp.content and len(resp.content) > 50, "生成内容过短！"
assert resp2.content and len(resp2.content) > 50, "SSE 生成内容过短！"

# 4. 验证 SSE 格式
from drama_agent.api import _format_sse
b = _format_sse("final", {"content": "你好，世界", "score": 0.95})
print("\n===== 验证 SSE 编码 =====")
print(repr(b[:200]))
assert b.startswith(b"event: final\n"), "SSE event 前缀缺失"
assert b"data:" in b, "SSE data 前缀缺失"
# 确认是真正的 UTF-8 字节（\xe4\xbd\xa0 是 "你" 的真正 UTF-8 字节，不是 \\\\u 的形式）
assert b'\\u4f60' not in b, "错误：中文被 unicode-escape 编码！"
assert "你好".encode("utf-8") in b, "中文未正确出现在 SSE 字节流"
# 再额外验证：如果 decode 一下，应该能正常解析
decoded = b.decode("utf-8")
assert "你好" in decoded, "UTF-8 解码后丢失中文"
print("[OK] SSE 编码正常（UTF-8，纯中文未被转义）")
print("UTF-8 decode 后:", decoded[:200])

print("\n[OK] 所有冒烟测试通过！")
