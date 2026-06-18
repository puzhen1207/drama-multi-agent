"""
大模型封装：统一调用豆包 / OpenAI 兼容接口，支持结构化输出 + 多轮对话。

设计：
- 每次 chat() 可接收额外的 context_messages（来自会话历史），构造成完整的 messages 发送给 LLM
- 提供 context_window 的粗略 token 控制：超过 max_chars 时自动截断最早的消息
- 若 API key 未配置 → 走本地 stub 模式（方便离线演示）
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

from .config import settings
from .exceptions import (
    LLMServiceError,
    LLMTimeoutError,
    LLMTokenLimitError,
    with_retry,
)
from .logging_setup import get_logger

logger = get_logger("llm")
T = TypeVar("T", bound=BaseModel)

# 粗略的字符上限（对应约 4k tokens 的中文），可通过 config 扩展
DEFAULT_MAX_CHARS = 8000


# =============================================================================
# 能力检测
# =============================================================================


def llm_available() -> bool:
    """LLM HTTP API 是否可用（有 base_url + 非空 api_key）。"""
    if not settings.llm_base_url:
        return False
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    return bool(key) and not key.startswith("sk-your-") and key != "sk-"


def llm_key_present() -> bool:
    """仅判断是否存在 API key（不校验值），用于初始化检查。"""
    return bool(settings.llm_api_key.get_secret_value()) if settings.llm_api_key else False


# =============================================================================
# messages 构建（支持多轮 + context window 管理）
# =============================================================================


def _build_messages(
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """
    构建 LLM messages。

    messages 顺序：[system] + [few_shots] + [context_messages(history)] + [最新 user_prompt]
    - context_messages：来自会话历史，格式与 OpenAI 兼容：[{"role":"user|assistant","content":str}]
    - 如果总字符过长，会从最老的 context 逐步删除，直到满足 max_chars
    """
    # 1. 放 system
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. 放 few_shots （示例对）
    if few_shots:
        for q, a in few_shots:
            messages.append({"role": "user", "content": str(q)})
            messages.append({"role": "assistant", "content": str(a)})

    # 3. 放历史上下文（最近几轮对话）
    history: List[Dict[str, Any]] = []
    if context_messages:
        for m in context_messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role in ("user", "assistant"):
                history.append({"role": role, "content": content})
    messages.extend(history)

    # 4. 放当前 user
    messages.append({"role": "user", "content": user_prompt})

    # 5. 截断：如果字符总量超 max_chars，从最老的 history message 开始删（保留 system + few_shots + 当前 user）
    return _trim_to_context_window(messages, max_chars)


def _trim_to_context_window(messages: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """保持首尾（system + 最后 user），删中间最老的历史消息，直到字符总量 <= max_chars。"""
    def total_chars() -> int:
        return sum(len(m.get("content", "")) for m in messages)

    if total_chars() <= max_chars:
        return messages

    # 找到可删除范围：第一个非 system 到倒数第二个（最后一个是当前 user）
    # 先尝试删除历史 message 中最早的
    while len(messages) > 2 and total_chars() > max_chars:
        # 找第一个 role != "system" 的 message 删除
        removed = False
        for i, m in enumerate(messages):
            if m.get("role") != "system" and i < len(messages) - 1:
                messages.pop(i)
                removed = True
                break
        if not removed:
            break
    return messages


# =============================================================================
# 基础 HTTP 调用（底层）
# =============================================================================


@with_retry
def _call_http_api(messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
    """用 httpx 直接调用 OpenAI 兼容接口（豆包、DeepSeek、OpenAI 等）。"""
    if not llm_available():
        raise LLMServiceError("LLM 未配置有效 API key，无法调用（可进入 stub 模式）")
    try:
        t0 = time.time()
        key = settings.llm_api_key.get_secret_value()
        url = settings.llm_base_url.rstrip("/") + "/chat/completions"

        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            dt_ms = (time.time() - t0) * 1000
            logger.info(f"LLM HTTP 调用完成 status={resp.status_code} 耗时 {dt_ms:.0f}ms, messages={len(messages)}")
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise LLMServiceError("LLM 返回空内容")
            return str(content)
    except httpx.TimeoutException as e:
        raise LLMTimeoutError(f"LLM 超时: {type(e).__name__}") from e
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        text = e.response.text[:300]
        if status in (429, 503, 502, 500):
            raise LLMTimeoutError(f"LLM {status} 限流/服务不可用") from e
        if status in (400,) and "token" in text.lower():
            raise LLMTokenLimitError(f"LLM token 超限（{status}）") from e
        if status in (401, 403):
            raise LLMServiceError(f"LLM 鉴权失败（{status}），请检查 API Key 是否正确") from e
        raise LLMServiceError(f"LLM 调用失败 status={status}: {text}") from e
    except Exception as e:
        raise LLMServiceError(f"LLM 调用失败: {type(e).__name__}") from e


# =============================================================================
# 高层 API
# =============================================================================


def chat(
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    普通文本对话。若提供 context_messages（历史 user/assistant 消息对），会完整注入到 LLM 上下文中。
    """
    if llm_available():
        messages = _build_messages(user_prompt, system_prompt, few_shots, context_messages)
        temp = temperature if temperature is not None else settings.llm_temperature
        return _call_http_api(messages, temperature=temp)
    logger.warning("LLM 不可用，进入本地 stub 模式")
    return _stub_chat(user_prompt, system_prompt)


def chat_structured(
    pydantic_cls: Type[T],
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> T:
    """结构化输出：要求 LLM 返回符合 Pydantic 模型的 JSON。"""
    schema = json.dumps(pydantic_cls.model_json_schema(), ensure_ascii=False, indent=2)
    full_system = (
        (f"{system_prompt}\n\n" if system_prompt else "") +
        "【输出格式要求】\n"
        "你必须严格返回一个 JSON 对象，字段与下面的 JSON Schema 一致：\n"
        f"{schema}\n"
        "不要返回任何解释性文字、markdown 代码块标记或额外内容，只返回一个合法 JSON 字符串。"
    )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            raw = chat(user_prompt=user_prompt, system_prompt=full_system,
                       few_shots=few_shots, context_messages=context_messages, **kwargs)
            raw_clean = _strip_json(raw)
            return pydantic_cls.model_validate_json(raw_clean)
        except Exception as e:
            last_err = str(e)
            logger.warning(f"结构化解析 attempt={attempt} 失败: {last_err}")
    raise LLMServiceError(f"结构化输出解析失败（多次重试后）: {last_err}")


def _strip_json(text: str) -> str:
    """清理 LLM 可能包裹的 ```json...``` 代码块，提取纯 JSON。"""
    if not text:
        return "{}"
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            t = t[start:end + 1]
    return t


# =============================================================================
# 本地 STUB（无 API key 时使用 — 便于离线演示）
# =============================================================================


def _stub_chat(user_prompt: str, system_prompt: str = "") -> str:
    """降级模式：当 LLM 未配置时走这里，保证系统始终可运行。"""
    text = (user_prompt + " " + system_prompt).strip()[:80]
    return (
        f"【《{text[:20]}》（STUB 模式 · 未接入真实 LLM）】\n\n"
        f"第 1 幕：开场冲突。主角在一次意外事件中身陷绝境，强烈情绪钩子吸引读者。\n"
        f"第 2 幕：反转升级。关键配角登场，局势反复反转，节奏紧凑。\n"
        f"第 3 幕：高潮与钩子。冲突达到顶点，以悬念结尾，吸引读者看下一集。\n\n"
        f"【人设】\n"
        f"- 主角：外柔内刚，心思缜密，关键时刻爆发。\n"
        f"- 配角：强势霸道，控制欲强，对主角专一。\n\n"
        f"【合规提示】内容中无政治敏感、色情低俗、血腥暴力元素。\n"
        f"\n提示：请在 .env 中配置 LLM_API_KEY 后可获得高质量生成。"
    )
