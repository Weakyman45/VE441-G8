"""DashScope / Qwen OpenAI-compatible Chat Completions (Planner & Workers)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def qwen_configured() -> bool:
    return bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip())


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    timeout: float = 45.0,
) -> str:
    """
    Call Qwen via DashScope compatible-mode.
    Returns assistant text, or raises RuntimeError on failure.
    """
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    base = (
        os.environ.get("QWEN_CHAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    chat_model = (
        model
        or os.environ.get("QWEN_CHAT_MODEL")
        or "qwen-plus"
    ).strip()

    payload = {
        "model": chat_model,
        "temperature": temperature,
        "messages": messages,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen network error: {exc}") from exc

    text = (
        ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    ).strip()
    if not text:
        raise RuntimeError(f"Qwen empty response: {json.dumps(data)[:300]}")
    return text


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Ask for JSON object; strips markdown fences if present."""
    raw = chat_completion(messages, model=model, temperature=temperature)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
