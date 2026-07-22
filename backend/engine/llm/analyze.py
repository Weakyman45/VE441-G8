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
    "\n注意：用户同时上传了商品参考图片（见随附图像）。请把图片作为多模态输入，"
    "结合图片中的视觉信息（品类、颜色、款式、材质、场景等）与文字描述一起分析，"
    "search_keywords 也要体现图片里看到的商品特征。"
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
        {"type": "text", "text": user_content or "请结合图片分析我的购物需求。"}
    ]
    for url in images:
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": parts}

_SYSTEM_PROMPT = (
    "你是购物需求分析助手。用户会用中文或英文描述想买的任意品类商品"
    "(如 笔记本/手机/耳机/相机/服饰/家居/运动器材等)。"
    "请只输出一个 JSON 对象，不要输出任何多余文字或解释。字段说明：\n"
    '{"category":"商品品类(简短，贴合用户输入，如 笔记本电脑/手机/跑鞋/咖啡机)",'
    '"budget":预算数字(人民币，未提到则为 null),'
    '"use_case":"主要用途(简短中文，如 学生日常/游戏/设计剪辑)",'
    '"platform":"No preference 或 Windows 或 macOS",'
    '"must_haves":["必须满足的条件，3-5 条简短中文短语"],'
    '"nice_to_haves":["加分项，2-4 条简短中文短语"],'
    '"search_keywords":["用于在英文商品库中检索的英文关键词，3-6 个，把商品品类与关键特征翻译成英文，如 running shoes white breathable"],'
    '"summary":"用一句话概括用户需求"}'
)

# Streaming variant: model first "thinks out loud" (shown to the user live),
# then emits a fenced JSON block we parse into the structured brief.
_STREAM_SYSTEM_PROMPT = (
    "你是购物需求分析助手。用户会用中文或英文描述想买的商品(任意品类，如笔记本、手机、鞋子等)。"
    "请分两部分回答：\n"
    "1) 先用中文写 3-5 句简短的分析思路，说明你如何理解用户的需求、需要关注哪些关键点、"
    "以及你会据此推导出哪些必要条件和加分项(用自然语言，像在思考)。\n"
    "2) 然后另起一行，输出一个 ```json 代码块，且只包含这一个 JSON 对象，字段如下：\n"
    '{"category":"商品品类(简短)",'
    '"budget":预算数字(人民币，未提到则为 null),'
    '"use_case":"主要用途(简短中文)",'
    '"platform":"No preference 或 Windows 或 macOS",'
    '"must_haves":["必须满足的条件，3-5 条简短中文短语"],'
    '"nice_to_haves":["加分项，2-4 条简短中文短语"],'
    '"search_keywords":["用于在英文商品库中检索的英文关键词，3-6 个，如 running shoes white breathable"],'
    '"summary":"用一句话概括用户需求"}'
)


def build_messages(text: str, prior: dict | None = None, *, streaming: bool = False) -> list[dict]:
    user_content = text.strip()
    if prior:
        keep = {k: prior.get(k) for k in ("category", "budget", "use_case", "platform") if prior.get(k)}
        if keep:
            user_content = f"已知偏好: {keep}\n最新描述: {text.strip()}"
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
            user_content = f"已知偏好: {keep}\n最新描述: {text}"

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
    "你是购物需求分析助手。用户会用中文或英文描述想买的商品。\n"
    "请先用 3-5 句简短中文,自然地说明你对用户需求的理解与分析思路"
    "(例如预算、用途、关键取舍),像在思考一样逐步展开。\n"
    "在分析叙述之后,另起一行输出一个严格的 JSON 对象(不要放进代码块,"
    "不要在 JSON 后再写任何文字)。JSON 字段:\n"
    '{"category":"商品品类(简短)",'
    '"budget":预算数字(人民币,未提到则 null),'
    '"use_case":"主要用途(简短中文)",'
    '"platform":"No preference 或 Windows 或 macOS",'
    '"must_haves":["必须满足条件,3-5 条简短中文短语"],'
    '"nice_to_haves":["加分项,2-4 条简短中文短语"],'
    '"search_keywords":["用于在英文商品库中检索的英文关键词,3-6 个,如 running shoes white breathable"],'
    '"summary":"一句话概括用户需求"}'
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
            user_content = f"已知偏好: {keep}\n最新描述: {text}"

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
