"""命令行入口：启动服务或单次运行工作流。"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="drama-agent",
        description="短剧多智能体内容生产系统",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="启动 FastAPI 服务")
    serve_p.add_argument("--host", default=None, help="监听地址（默认读取 .env）")
    serve_p.add_argument("--port", type=int, default=None, help="监听端口（默认 8000）")
    serve_p.add_argument("--reload", action="store_true", help="开发模式热重载")

    run_p = sub.add_parser("run", help="单次运行工作流并打印结果")
    run_p.add_argument("prompt", help="用户输入")
    run_p.add_argument("--user-id", default="cli", help="用户标识")
    run_p.add_argument("--session-id", default=None, help="会话 ID（可选）")

    sub.add_parser("tools", help="列出已注册的 MCP 工具")

    sub.add_parser("mcp", help="启动标准 MCP Server（stdio，供 Cursor 等客户端连接）")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "tools":
        _cmd_tools()
    elif args.command == "mcp":
        _cmd_mcp()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .config import settings

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    uvicorn.run(
        "drama_agent.api:app",
        host=host,
        port=port,
        reload=args.reload,
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from .graph import run_workflow

    resp = run_workflow(args.prompt, args.user_id, args.session_id)
    print(f"success={resp.success} task_type={resp.task_type} "
          f"iter={resp.iteration_count} degrade={resp.degrade_mode}")
    if resp.audit_result:
        print(f"audit: passed={resp.audit_result.passed} score={resp.audit_result.score}")
    print("-" * 60)
    print(resp.content)
    if resp.error:
        print(f"\n[error] {resp.error}", file=sys.stderr)
        sys.exit(1)


def _cmd_tools() -> None:
    from .graph import list_tools

    for name in list_tools():
        print(name)


def _cmd_mcp() -> None:
    from .mcp_server import run_mcp_server

    run_mcp_server()


if __name__ == "__main__":
    main()
