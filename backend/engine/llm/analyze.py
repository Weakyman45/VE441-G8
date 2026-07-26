"""LLM-based shopping-need analysis for text / transcribed-voice input.

Turns a free-text request into a structured brief (category, budget, use case,
platform, must-haves, nice-to-haves) using the same Qwen chat endpoint the
Planner/Recommend workers use. Falls back gracefully when the LLM is not
configured or the call fails, so the pipeline keeps working offline.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .qwen_client import chat_json, chat_stream, qwen_configured

# Appended to the system prompt when the user also attached reference image(s),
# so the model treats the image as first-class multimodal input (not a caption).
_IMAGE_HINT = (
    "\nNote: the user also attached product reference image(s). Treat the image "
    "as multimodal input. Analyze the visual details you can see, such as "
    "category, color, style, material, and usage context, together with the "
    "text description. Make search_keywords reflect visible product traits too."
)


def _analyze_model(images: list[str] | None) -> str | None:
    """VL model when images are present; None → caller's default text model."""
    if images:
        return (os.environ.get("QWEN_VL_MODEL") or "qwen-vl-plus").strip()
    return None


def _user_message(user_content: str, images: list[str] | None) -> dict:
    """Build the user turn. With images, emit OpenAI-compatible multimodal
    content parts (text + image_url) so the whole image reaches the LLM."""
    if not images:
        return {"role": "user", "content": user_content}
    parts: list[dict] = [
        {"type": "text", "text": user_content or "Analyze my shopping need using the attached image."}
    ]
    for url in images:
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": parts}

_SYSTEM_PROMPT = (
    "You are a shopping need analysis assistant. The user may describe any "
    "product category in Chinese or English, such as laptops, phones, earbuds, "
    "cameras, clothing, home goods, or sports equipment. Output ONLY one JSON "
    "object with no extra text or explanation. Field guide:\n"
    '{"category":"short product category matching the user input, e.g. laptop, phone, running shoes, coffee machine",'
    '"budget":numeric budget in RMB, or null if not mentioned,'
    '"use_case":"short main use case in English, e.g. student daily use, gaming, design/video editing",'
    '"platform":"No preference, Windows, or macOS",'
    '"must_haves":["3-5 short English phrases for required conditions"],'
    '"nice_to_haves":["2-4 short English phrases for optional preferences"],'
    '"search_keywords":["3-6 English catalog search keywords, translating the product category and key traits, e.g. running shoes white breathable"],'
    '"summary":"one-sentence English API conversation summary of the user shopping need and key constraints"}'
)

# Streaming variant: model first "thinks out loud" (shown to the user live),
# then emits a fenced JSON block we parse into the structured brief.
_STREAM_SYSTEM_PROMPT = (
    "You are a shopping need analysis assistant. The user may describe any "
    "product category in Chinese or English, such as laptops, phones, or shoes. "
    "Respond in two parts:\n"
    "1. First write 3-5 short English sentences explaining how you understand "
    "the user's need, what key points matter, and which required and optional "
    "conditions you infer.\n"
    "2. Then on a new line, output exactly one fenced ```json code block that "
    "contains only this JSON object with these fields:\n"
    '{"category":"short product category",'
    '"budget":numeric budget in RMB, or null if not mentioned,'
    '"use_case":"short main use case in English",'
    '"platform":"No preference, Windows, or macOS",'
    '"must_haves":["3-5 short English phrases for required conditions"],'
    '"nice_to_haves":["2-4 short English phrases for optional preferences"],'
    '"search_keywords":["3-6 English catalog search keywords, e.g. running shoes white breathable"],'
    '"summary":"one-sentence English API conversation summary of the user shopping need and key constraints"}'
)


def build_messages(text: str, prior: dict | None = None, *, streaming: bool = False) -> list[dict]:
    user_content = text.strip()
    if prior:
        keep = {k: prior.get(k) for k in ("category", "budget", "use_case", "platform") if prior.get(k)}
        if keep:
            user_content = f"Known preferences: {keep}\nLatest description: {text.strip()}"
    system = _STREAM_SYSTEM_PROMPT if streaming else _SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _extract_json_object(text: str) -> dict | None:
    """Find the JSON brief inside free-form LLM text (fenced or bare)."""
    if not text:
        return None
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.rfind("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def normalize_analysis(data: dict) -> dict:
    return {
        "provider": "qwen",
        "category": str(data.get("category") or "").strip(),
        "budget": _to_int(data.get("budget")),
        "use_case": str(data.get("use_case") or "").strip(),
        "platform": _norm_platform(data.get("platform")),
        "must_haves": _str_list(data.get("must_haves"))[:6],
        "nice_to_haves": _str_list(data.get("nice_to_haves"))[:6],
        "search_keywords": _str_list(data.get("search_keywords"))[:6],
        "summary": str(data.get("summary") or "").strip(),
    }


def parse_analysis_text(full_text: str) -> dict | None:
    """Turn a streamed LLM answer into a normalized analysis dict, or None."""
    data = _extract_json_object(full_text)
    if data is None:
        return None
    return normalize_analysis(data)


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _norm_platform(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "mac" in text:
        return "macOS"
    if "windows" in text or "win" in text:
        return "Windows"
    return "No preference"


def _empty(provider: str, reason: str = "") -> dict:
    out = {
        "provider": provider,
        "category": "",
        "budget": None,
        "use_case": "",
        "platform": "No preference",
        "must_haves": [],
        "nice_to_haves": [],
        "search_keywords": [],
        "summary": "",
    }
    if reason:
        out["reason"] = reason
    return out


def analyze_need(text: str, prior: dict | None = None, images: list[str] | None = None) -> dict:
    """Return an LLM-derived shopping brief. `provider` is 'qwen' on success,
    otherwise 'fallback' / 'empty' so callers can show honest status.

    When `images` (list of data URLs) is provided, the image(s) and the text
    are sent together as a single multimodal turn to a VL model, so the brief
    reflects the actual picture rather than pre-extracted keywords."""
    text = (text or "").strip()
    if not text and not images:
        return _empty("empty", "no text")
    if not qwen_configured():
        return _empty("fallback", "DASHSCOPE_API_KEY not set")

    user_content = text
    if prior:
        # Give the model prior context so multi-turn refinement accumulates.
        keep = {k: prior.get(k) for k in ("category", "budget", "use_case", "platform") if prior.get(k)}
        if keep:
            user_content = f"Known preferences: {keep}\nLatest description: {text}"

    system_prompt = _SYSTEM_PROMPT + (_IMAGE_HINT if images else "")
    try:
        data = chat_json(
            [
                {"role": "system", "content": system_prompt},
                _user_message(user_content, images),
            ],
            model=_analyze_model(images),
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        return _empty("fallback", str(exc)[:200])

    if not isinstance(data, dict):
        return _empty("fallback", "non-dict LLM reply")

    return {
        "provider": "qwen",
        "category": str(data.get("category") or "").strip(),
        "budget": _to_int(data.get("budget")),
        "use_case": str(data.get("use_case") or "").strip(),
        "platform": _norm_platform(data.get("platform")),
        "must_haves": _str_list(data.get("must_haves"))[:6],
        "nice_to_haves": _str_list(data.get("nice_to_haves"))[:6],
        "search_keywords": _str_list(data.get("search_keywords"))[:6],
        "summary": str(data.get("summary") or "").strip(),
    }


# System prompt for the streaming path: narrate the reasoning first (shown to
# the user as "thinking"), then output the JSON on its own so the client can
# parse it after the narrative.
_STREAM_SYSTEM_PROMPT = (
    "You are a shopping need analysis assistant. The user may describe the "
    "product they want to buy in Chinese or English.\n"
    "First, write 3-5 short English sentences that naturally explain your "
    "understanding of the user's need and reasoning, such as budget, use case, "
    "and key trade-offs.\n"
    "After that narrative, output one strict JSON object on a new line. Do not "
    "wrap it in a code block, and do not write anything after the JSON. JSON fields:\n"
    '{"category":"short product category",'
    '"budget":numeric budget in RMB, or null if not mentioned,'
    '"use_case":"short main use case in English",'
    '"platform":"No preference, Windows, or macOS",'
    '"must_haves":["3-5 short English phrases for required conditions"],'
    '"nice_to_haves":["2-4 short English phrases for optional preferences"],'
    '"search_keywords":["3-6 English catalog search keywords, e.g. running shoes white breathable"],'
    '"summary":"one-sentence English API conversation summary of the user shopping need and key constraints"}'
)


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _analysis_from_data(data: dict) -> dict:
    return {
        "provider": "qwen",
        "category": str(data.get("category") or "").strip(),
        "budget": _to_int(data.get("budget")),
        "use_case": str(data.get("use_case") or "").strip(),
        "platform": _norm_platform(data.get("platform")),
        "must_haves": _str_list(data.get("must_haves"))[:6],
        "nice_to_haves": _str_list(data.get("nice_to_haves"))[:6],
        "search_keywords": _str_list(data.get("search_keywords"))[:6],
        "summary": str(data.get("summary") or "").strip(),
    }


def analyze_need_stream(text: str, prior: dict | None = None, images: list[str] | None = None):
    """Generator for streaming analysis.

    Yields ("delta", narrative_chunk) as the model narrates its reasoning, then
    finally ("done", analysis_dict). The JSON tail is buffered silently (not
    emitted as deltas) so the user only sees the natural-language thinking.

    When `images` (list of data URLs) is provided, the picture(s) are sent
    together with the text as a multimodal turn to a VL model.
    """
    text = (text or "").strip()
    if not text and not images:
        yield ("done", _empty("empty", "no text"))
        return
    if not qwen_configured():
        yield ("done", _empty("fallback", "DASHSCOPE_API_KEY not set"))
        return

    user_content = text
    if prior:
        keep = {k: prior.get(k) for k in ("category", "budget", "use_case", "platform") if prior.get(k)}
        if keep:
            user_content = f"Known preferences: {keep}\nLatest description: {text}"

    messages = [
        {"role": "system", "content": _STREAM_SYSTEM_PROMPT + (_IMAGE_HINT if images else "")},
        _user_message(user_content, images),
    ]

    full = ""
    emitted = 0
    json_started = False
    try:
        for delta in chat_stream(messages, model=_analyze_model(images), temperature=0.2, timeout=60.0):
            full += delta
            if not json_started:
                brace = full.find("{")
                narrative = full if brace == -1 else full[:brace]
                if brace != -1:
                    json_started = True
                if len(narrative) > emitted:
                    yield ("delta", narrative[emitted:])
                    emitted = len(narrative)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        yield ("done", _empty("fallback", str(exc)[:200]))
        return

    data = _extract_json(full)
    if not data:
        yield ("done", _empty("fallback", "no json in reply"))
        return
    yield ("done", _analysis_from_data(data))
