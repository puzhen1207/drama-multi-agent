"""工具注册中心 —— MCP 风格的工具注册表。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..logging_setup import get_logger

logger = get_logger("tools")


class ToolRegistry:
    """MCP 风格工具注册中心：一次注册，多 Agent 共用。

    用法 1：直接注册函数
        registry.register("my_tool", my_func, "description", {"input": str})

    用法 2：装饰器模式
        @registry.register("my_tool", input_schema={"input": str})
        def my_func(input: str = "") -> dict: ...

    用法 3：调用工具
        registry.call("my_tool", input="value")
        registry.invoke("my_tool", {"input": "value"})
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        func: Optional[Callable] = None,
        description: str = "",
        input_schema: Dict[str, Any] | None = None,
    ) -> Any:
        """注册工具。支持装饰器用法和直接传参用法。

        - 直接传参: register("name", func, "desc", schema)
        - 装饰器用法: @register("name", input_schema={...})
        - 装饰器简洁用法: @register("name", schema_dict)
        - 装饰器无参数: @register
        """
        # case 1: 作为装饰器但只有函数 (@registry.register)
        if callable(name) and func is None:
            func = name
            name = func.__name__
            self._do_register(name, func, description, input_schema)
            return func

        # case 2: 作为装饰器 (func 是 None 或是 dict 等非 callable)
        if func is None or not callable(func):
            # 如果 func 实际上是 input_schema（dict），把它转过去
            if isinstance(func, dict):
                input_schema = func
            # 3 位置参数的情况: register("name", func, schema_dict) - 检测第三个参数
            def decorator(f: Callable) -> Callable:
                self._do_register(name or f.__name__, f, description, input_schema)
                return f
            return decorator

        # case 3: 直接注册
        self._do_register(name or func.__name__, func, description, input_schema)
        return None

    def _do_register(
        self, name: str, func: Callable, description: str, input_schema: Dict[str, Any] | None
    ) -> None:
        if name in self._tools:
            logger.warning(f"工具 {name} 已存在，将被覆盖")
        self._tools[name] = {
            "name": name,
            "func": func,
            "description": description or getattr(func, "__doc__", "") or "",
            "input_schema": input_schema or {"type": "object", "properties": {}},
        }
        logger.info(f"[MCP] 工具已注册: {name}")

    def call(self, name: str, **kwargs) -> Any:
        """用关键字参数调用工具。"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        tool = self._tools[name]
        logger.info(f"[MCP] 调用工具: {name}, args={list(kwargs.keys())}")
        return tool["func"](**kwargs)

    def invoke(self, name: str, kwargs: Dict[str, Any]) -> Any:
        """用字典参数调用工具（别名，方便测试和外部 API 调用）。"""
        return self.call(name, **(kwargs or {}))

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def tool_meta(self, name: str) -> Dict[str, Any]:
        return {k: v for k, v in self._tools[name].items() if k != "func"}

    def has_tool(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()
