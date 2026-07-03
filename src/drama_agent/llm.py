"""大模型封装层：统一调用 OpenAI 兼容接口；未配置 API Key 时自动降级到 stub 模式。

目标：
- 每次 chat() 可接收额外 context_messages（来自会话历史），构造完整 messages 发给 LLM；
- 粗略 token 控制：超过 max_chars 时从最早的历史消息删除；
- 若未配置 API Key → 本地 stub 模式（方便离线演示）。
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Type

import httpx
from pydantic import BaseModel

from .config import settings
from .exceptions import LLMServiceError, LLMTimeoutError, with_retry
from .logging_setup import get_logger

logger = get_logger("llm")


DEFAULT_MAX_CHARS = 8000


def llm_available() -> bool:
    """LLM HTTP API 是否可用（有 base_url + 非空有效 api_key）。"""
    if not settings.llm_base_url:
        return False
    key = ""
    try:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    except Exception:
        key = ""
    if not key:
        return False
    # 占位 key（sk-your-...）视为未配置
    if key.startswith("sk-your-") or key == "sk-":
        return False
    return True


def llm_key_present() -> bool:
    try:
        return bool(settings.llm_api_key.get_secret_value()) if settings.llm_api_key else False
    except Exception:
        return False


# ============= messages 构建 =============


def _build_messages(
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shots:
        for q, a in few_shots:
            messages.append({"role": "user", "content": str(q)})
            messages.append({"role": "assistant", "content": str(a)})
    if context_messages:
        for m in context_messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return _trim_to_context_window(messages, max_chars)


def _trim_to_context_window(messages: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    def total() -> int:
        return sum(len(m.get("content", "")) for m in messages)

    if total() <= max_chars:
        return messages

    # 只删"历史 context"的非 system 消息（system 和最后一条 user 保留）
    while len(messages) > 2 and total() > max_chars:
        # 删除第一条非 system 的消息（通常是最旧的历史）
        removed = False
        for i, m in enumerate(messages):
            if m.get("role") != "system" and i < len(messages) - 1:
                messages.pop(i)
                removed = True
                break
        if not removed:
            break
    return messages


# ============= 底层 HTTP =============


@with_retry
def _call_http_api(messages: List[Dict[str, Any]], temperature: float = 0.7) -> str:
    """调用 OpenAI 兼容接口；未配置时抛出 LLMServiceError（走 stub 路径）。"""
    if not llm_available():
        raise LLMServiceError("LLM 未配置有效 API Key")
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
        if status in (401, 403):
            raise LLMServiceError(f"LLM 鉴权失败（{status}），请检查 API Key 是否正确") from e
        raise LLMServiceError(f"LLM 调用失败 status={status}: {text}") from e
    except Exception as e:
        raise LLMServiceError(f"LLM 调用失败: {type(e).__name__}: {e}") from e


# ============= 高层 API =============


def chat(
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
) -> str:
    """普通文本对话。LLM 未配置时自动返回 stub。"""
    if llm_available():
        messages = _build_messages(user_prompt, system_prompt, few_shots, context_messages)
        temp = temperature if temperature is not None else settings.llm_temperature
        try:
            return _call_http_api(messages, temperature=temp)
        except Exception as e:
            logger.warning(f"LLM 调用失败，进入 stub 模式: {e}")
    logger.warning("LLM 不可用（未配置 API Key），进入本地 stub 模式")
    return _stub_chat(user_prompt, system_prompt)


def chat_structured(
    pydantic_cls: Type[BaseModel],
    user_prompt: str,
    system_prompt: str = "",
    few_shots: Optional[List[tuple]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 2,
    **kwargs: Any,
) -> BaseModel:
    """结构化输出：要求 LLM 返回符合 Pydantic 模型的 JSON。"""
    schema = json.dumps(pydantic_cls.model_json_schema(), ensure_ascii=False, indent=2)
    full_system = (
        (f"{system_prompt}\n\n" if system_prompt else "")
        + "【输出格式要求】\n"
        + "你必须严格返回一个 JSON 对象，字段与下面的 JSON Schema 一致：\n"
        + f"{schema}\n"
        + "不要返回任何解释性文字、markdown 代码块标记或额外内容，只返回一个合法 JSON 字符串。"
    )
    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            raw = chat(
                user_prompt=user_prompt,
                system_prompt=full_system,
                few_shots=few_shots,
                context_messages=context_messages,
                **kwargs,
            )
            raw_clean = _strip_json(raw)
            obj = pydantic_cls.model_validate_json(raw_clean)
            return obj
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


def _stub_chat(user_prompt: str, system_prompt: str = "") -> str:
    """本地 stub 模式 —— 不依赖网络也能输出演示内容。"""
    summary = (user_prompt + " " + (system_prompt or "")).strip()[:60]
    return (
        f"【《{summary}》（STUB 模式 · 未接入真实 LLM）】\n\n"
        f"本内容由本地模板生成，用于演示系统可运行性。请在项目根目录的 .env 中"
        f"配置 LLM_API_KEY 后，可获得高质量的大模型生成内容。\n\n"
        f"第 1 幕：开场冲突。主角在一次意外事件中身陷绝境，强烈情绪钩子吸引读者。\n"
        f"第 2 幕：反转升级。关键配角登场，局势反复反转，节奏紧凑。\n"
        f"第 3 幕：高潮与钩子。冲突达到顶点，以悬念结尾，吸引读者看下一集。\n\n"
        f"【人设】\n"
        f"- 主角：外柔内刚，心思缜密，关键时刻爆发。\n"
        f"- 配角：强势霸道，控制欲强，对主角专一。\n\n"
        f"【合规提示】内容中无政治敏感、色情低俗、血腥暴力元素。\n"
    )
