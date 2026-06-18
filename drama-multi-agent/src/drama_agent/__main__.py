"""命令行入口：`python -m drama_agent [subcommand] args。"""
from __future__ import annotations

import argparse
import json
import sys

from .config import PROJECT_ROOT
from .graph import list_tools, run_workflow
from .logging_setup import get_logger, setup_logging

setup_logging()
logger = get_logger("cli")


def _cmd_run(args: argparse.Namespace) -> int:
    text = args.text or input("请输入：")
    resp = run_workflow(text, user_id=args.user_id)
    print("\n" + "=" * 60)
    print(f"✓ 任务完成（耗时 {resp.elapsed_ms:.0f}ms，迭代 {resp.iteration_count} 次，降级={resp.degrade_mode}")
    if resp.error:
        print(f"⚠️  错误：{resp.error}")
    print(f"任务类型：{resp.task_type}")
    print("=" * 60)
    if resp.audit_result:
        audit = resp.audit_result
        print(f"审核结果：passed={audit.passed}, score={audit.score}, issues={len(audit.issues)}")
        print(f"审核摘要：{audit.summary}")
    print("-" * 60)
    print(resp.content)
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    print("已注册 MCP 工具：")
    for t in list_tools():
        print(f"  - {t}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cases = [
        "给我整理一段关于「霸总追妻」的小说大纲，分 3 集，每集 400 字",
        "写 2 版不同风格的推广文案，用来推广我们的都市新剧《错位人生》",
        "短剧内容中是否允许出现血腥暴力场景？有哪些合规要求？",
        "生成一个关于「高考状元穿越古代」的爆款剧本，风格是爽文，500 字",
    ]
    for i, case in enumerate(cases, 1):
        print("\n=== demo " + str(i) + "/4 === " + case)
        resp = run_workflow(case, user_id="demo")
        print(f"   type={resp.task_type}, passed={resp.audit_result.passed if resp.audit_result else None}, "
              f"耗时 {resp.elapsed_ms:.0f}ms")
    return 0


def main(argv: argparse.Namespace) -> int:
    parser = argparse.ArgumentParser(prog="drama-agent", description="短剧多智能体 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="单次运行")
    p_run.add_argument("text", nargs="?", default=None)
    p_run.add_argument("--user-id", default="cli")
    p_run.set_defaults(func=_cmd_run)

    p_tools = sub.add_parser("tools", help="列出已注册的 MCP 工具")
    p_tools.set_defaults(func=_cmd_tools)

    p_demo = sub.add_parser("demo", help="跑 4 个内置测试用例")
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
