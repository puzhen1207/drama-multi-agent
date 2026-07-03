"""工具注册中心 —— MCP 风格的工具注册表（drama-multi-agent1 重写版）。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..logging_setup import get_logger

logger = get_logger("tools")


class ToolRegistry:
    """MCP 风格工具注册中心：一次注册，多 Agent 共用。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name_or_func: Any = None,
        func: Optional[Callable] = None,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """支持装饰器模式和直接注册两种用法。"""
        # 用法 1：作为装饰器 @register（只有一个参数，且是 callable）
        if callable(name_or_func) and func is None and description == "" and input_schema is None:
            fn = name_or_func
            self._do_register(fn.__name__, fn, "", None)
            return fn

        # 用法 2：作为装饰器 @register("name", ...) 或 @register("name")
        if not callable(name_or_func) and func is None:
            def decorator(f: Callable) -> Callable:
                self._do_register(name_or_func or f.__name__, f, description, input_schema)
                return f
            return decorator

        # 用法 3：直接 register("name", func, "desc", schema)
        self._do_register(name_or_func or (func.__name__ if func else "unnamed"),
                          func, description, input_schema)
        return None

    def _do_register(
        self, name: str, func: Optional[Callable], description: str,
        input_schema: Optional[Dict[str, Any]],
    ) -> None:
        if name in self._tools:
            logger.warning(f"工具 {name} 已存在，将被覆盖")
        self._tools[name] = {
            "name": name,
            "func": func,
            "description": description or "",
            "input_schema": input_schema or {"type": "object", "properties": {}},
        }
        logger.info(f"[MCP] 工具已注册: {name}")

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        tool = self._tools[name]
        logger.info(f"[MCP] 调用工具: {name}, args={list(kwargs.keys())}")
        if tool["func"] is None:
            raise RuntimeError(f"工具 {name} 未绑定函数")
        return tool["func"](**kwargs)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


registry = ToolRegistry()


def _ensure_tools_loaded() -> None:
    """导入所有工具模块以完成 MCP 注册。"""
    from . import compliance_engine, text_processor, vector_retriever  # noqa: F401


_ensure_tools_loaded()
