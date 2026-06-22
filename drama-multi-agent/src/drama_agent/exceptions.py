"""异常定义与重试装饰器。"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class DramaAgentError(Exception):
    """系统基类异常。"""

    code = "DRAMA_ERROR"

    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __repr__(self) -> str:
        return f"<{self.code}: {self.message}>"


class RetryableError(DramaAgentError):
    """可重试异常（超时、限流、网络波动）。"""

    code = "RETRYABLE"


class NonRetryableError(DramaAgentError):
    """不可重试异常（参数错误、数据非法、鉴权失败）。"""

    code = "NON_RETRYABLE"


class LLMTimeoutError(RetryableError):
    code = "LLM_TIMEOUT"


class LLMServiceError(RetryableError):
    code = "LLM_SERVICE"


class ValidationError(NonRetryableError):
    code = "VALIDATION"


class RetrievalError(RetryableError):
    code = "RETRIEVAL"


class EmptyMaterialError(NonRetryableError):
    code = "EMPTY_MATERIAL"


RETRYABLE_EXCEPTIONS = (
    LLMTimeoutError,
    LLMServiceError,
    RetrievalError,
    TimeoutError,
    ConnectionError,
)


DEFAULT_RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)


def with_retry(func: Callable | None = None, **kwargs) -> Callable:
    """节点级重试装饰器。

    用法：
        @with_retry
        def foo(...): ...

        @with_retry(stop=stop_after_attempt(5))
        def bar(...): ...
    """
    policy = {**DEFAULT_RETRY_POLICY, **kwargs}

    def _decorator(fn: Callable) -> Callable:
        @retry(**policy)
        @wraps(fn)
        def _wrapped(*args, **kw):
            return fn(*args, **kw)

        return _wrapped

    if func is None:
        return _decorator
    return _decorator(func)
