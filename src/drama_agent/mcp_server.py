"""标准 MCP Server —— 将已注册工具暴露给 Cursor / Claude Desktop 等外部客户端。

启动：drama-agent mcp
传输：stdio（MCP 标准）
"""
from __future__ import annotations

from typing import Any, List


def run_mcp_server() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise SystemExit(
            "未安装 MCP SDK。请执行: pip install mcp\n"
            f"原始错误: {e}"
        ) from e

    from .tools import registry
    from .tools.compliance_engine import tool_sensitive_check
    from .tools.text_processor import (
        tool_normalize_text,
        tool_split_paragraphs,
        tool_truncate_text,
    )
    from .tools.user_memory import search_user_memory
    from .tools.vector_retriever import tool_retrieve_materials

    mcp = FastMCP("drama-agent")

    @mcp.tool()
    def retrieve_materials(query: str, top_k: int = 3) -> List[dict]:
        """从短剧公共素材库（剧本/文案/合规/人设）检索最相关素材。"""
        return tool_retrieve_materials(query=query, top_k=top_k)

    @mcp.tool()
    def search_personal_memory(user_id: str, query: str, top_k: int = 2) -> List[dict]:
        """从用户个人记忆库检索历史 Q&A。"""
        return search_user_memory(user_id=user_id, query=query, top_k=top_k)

    @mcp.tool()
    def sensitive_check(text: str) -> dict:
        """短剧内容合规规则检测（敏感词/个人信息等）。"""
        return tool_sensitive_check(text)

    @mcp.tool()
    def normalize_text(text: str) -> str:
        """规范化文本空白与换行。"""
        return tool_normalize_text(text)

    @mcp.tool()
    def truncate_text(text: str, max_chars: int = 500) -> str:
        """按字符数截断文本。"""
        return tool_truncate_text(text, max_chars=max_chars)

    @mcp.tool()
    def split_paragraphs(text: str) -> List[str]:
        """按段落切分文本。"""
        return tool_split_paragraphs(text)

    @mcp.tool()
    def list_registered_tools() -> List[str]:
        """列出进程内已注册的全部工具名。"""
        return registry.list_tools()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
