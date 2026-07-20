"""Qwen VL image understanding for shopping references."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request


def describe_shopping_image(
    image_bytes: bytes,
    *,
    mime_type: str = "image/jpeg",
    filename: str = "reference.jpg",
    user_text: str = "",
    timeout: float = 45.0,
) -> dict:
    """Return compact shopping signals extracted from an uploaded image.

    Uses DashScope OpenAI-compatible multimodal chat when configured. The
    fallback is intentionally useful enough for the rest of the pipeline.
    """
    if not image_bytes:
        return _fallback(filename, "empty image")
    if not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        return _fallback(filename, "DASHSCOPE_API_KEY not set")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type or 'image/jpeg'};base64,{b64}"
    prompt = (
        "You are the image understanding worker for a shopping assistant. "
        "Look at the uploaded reference image and extract purchase-relevant "
        "signals for laptop recommendations. Return ONLY a JSON object with "
        "keys: summary, product_category, visual_preferences, hard_constraints, "
        "soft_preferences, search_keywords. Do not invent exact model names, "
        "prices, or specs unless clearly visible in the image."
    )
    if user_text.strip():
        prompt += f"\nUser text context: {user_text.strip()[:500]}"

    payload = {
        "model": (os.environ.get("QWEN_VL_MODEL") or "qwen-vl-plus").strip(),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": "Extract concise shopping signals from this image.",
                    },
                ],
            },
        ],
    }
    base = (
        os.environ.get("QWEN_CHAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {(os.environ.get('DASHSCOPE_API_KEY') or '').strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        parsed = _parse_json_object(raw)
        parsed.setdefault("summary", raw[:180] if raw else "Reference image attached")
        parsed.setdefault("product_category", "laptop")
        parsed.setdefault("visual_preferences", [])
        parsed.setdefault("hard_constraints", [])
        parsed.setdefault("soft_preferences", [])
        parsed.setdefault("search_keywords", [])
        parsed["provider"] = "qwen-vl"
        return parsed
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return _fallback(filename, str(exc)[:180])


def visual_context_text(analysis: dict) -> str:
    parts: list[str] = []
    summary = str(analysis.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    for key in ("visual_preferences", "soft_preferences", "hard_constraints", "search_keywords"):
        values = analysis.get(key) or []
        if isinstance(values, list):
            parts.extend(str(v).strip() for v in values if str(v).strip())
        elif isinstance(values, str) and values.strip():
            parts.append(values.strip())
    text = "; ".join(dict.fromkeys(parts))
    return text[:700] if text else "Reference image attached"


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _fallback(filename: str, reason: str) -> dict:
    return {
        "summary": f"Reference image attached: {filename}",
        "product_category": "laptop",
        "visual_preferences": ["use the uploaded image as a style and context reference"],
        "hard_constraints": [],
        "soft_preferences": ["visual reference provided"],
        "search_keywords": ["laptop"],
        "provider": "fallback",
        "warning": reason,
    }
