"""Text Talker stand-in for CLI (Qwen chat; no Omni audio)."""

from __future__ import annotations

from typing import Any

from ..llm.qwen_client import chat_completion, qwen_configured


def summarize_recommendations(
    *,
    user_text: str,
    preference: dict[str, Any],
    bundle: dict[str, Any],
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Ask Qwen chat to speak a short shopping summary (no invented products)."""
    _ = api_key  # kept for call-site compatibility; uses DASHSCOPE_API_KEY
    if not qwen_configured():
        return _fallback_summary(bundle)

    ranked = bundle.get("ranked") or []
    facts = []
    for i, item in enumerate(ranked[:3], 1):
        facts.append(
            {
                "rank": i,
                "name": item.get("name"),
                "price": item.get("price"),
                "score": item.get("score"),
                "reasons": item.get("reasons") or [],
            }
        )
    system = (
        "You are VoiceShop++, a concise retail shopping Talker. "
        "Always reply in English only. "
        "Summarize ONLY the Worker facts below in 2–4 short spoken sentences. "
        "Do not invent products, prices, or specs."
    )
    user = (
        f"user_said={user_text}\n"
        f"preference={preference}\n"
        f"worker_summary={bundle.get('summary')}\n"
        f"top_matches={facts}"
    )
    try:
        return chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=0.4,
        )
    except Exception as exc:
        return _fallback_summary(bundle) + f"\n(LLM unavailable: {exc})"


def _fallback_summary(bundle: dict[str, Any]) -> str:
    summary = (bundle.get("summary") or "").strip()
    if summary:
        return summary
    ranked = bundle.get("ranked") or []
    if not ranked:
        return "还没有找到合适的推荐，可以再说一下预算或用途。"
    top = ranked[0]
    return (
        f"目前最匹配的是 {top.get('name')}，大约 {top.get('price')}，"
        f"匹配分 {top.get('score')}。"
    )
