"""Text Talker stand-in for CLI (Qwen chat; no Omni audio)."""

from __future__ import annotations

import os
from typing import Any

from ..llm.qwen_client import chat_completion, qwen_configured
from .shopping_safety import (
    SHOPPING_SAFETY_BOUNDARIES,
    RiskAssessment,
    assess_shopping_risk,
    risk_context_message,
    shopping_safety_enabled,
)

INTERRUPTED_MARKER = "[INTERRUPTED]"
UNSPOKEN_MARKER = "[UNSPOKEN]"

INTERRUPTION_INSTRUCTIONS = f"""
If a prior assistant message contains {INTERRUPTED_MARKER}, that marker is the cut-off point.
The user did not hear anything after that marker.
If the message also contains {UNSPOKEN_MARKER}, the text after it is unspoken
draft content from the interrupted reply. Treat it only as private context, not
as something the user heard.

## Handling interruptions (barge-in)

The user can and will interrupt you mid-sentence. When that happens:

1. You did NOT finish speaking. The user only heard the words up to the
   point where they cut in - never assume the rest was heard. Never claim
   or imply you "already mentioned" or "just said" something that came
   after the cut-off point.

2. Whatever the user says when interrupting is their most current and
   highest-priority intent. Adopt it immediately and let it override your
   previous plan or the answer you were in the middle of giving.

3. Be brief. Respond directly to what they now want. Do NOT re-read,
   recap, or repeat the recommendations or details they already heard
   before interrupting.

4. Only restate something from the unspoken part if it is essential to the
   user's new request AND they were cut off before hearing it - and then
   keep it to a single short clause, not a re-listing.

5. If the interruption is just a backchannel ("uh-huh", "okay", "right",
   "mm-hmm", "got it") or clearly not addressed to you (background speech,
   someone else talking), do NOT treat it as a new instruction. Continue
   naturally from where you were.

Keep every reply short and spoken-style - one or two sentences.
""".strip()


def interruption_handling_enabled() -> bool:
    raw = os.environ.get("INTERRUPTION_HANDLING_ENABLED", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


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
    if shopping_safety_enabled():
        system = f"{system}\n\n{SHOPPING_SAFETY_BOUNDARIES}"
    if interruption_handling_enabled():
        system = f"{system}\n\n{INTERRUPTION_INSTRUCTIONS}"
    risk = assess_shopping_risk(user_text) if shopping_safety_enabled() else RiskAssessment()
    user = (
        f"user_said={user_text}\n"
        f"preference={preference}\n"
        f"{risk_context_message(risk)}\n"
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
        return "I have not found a good match yet. Tell me a bit more about your budget or use case."
    top = ranked[0]
    return (
        f"The strongest match right now is {top.get('name')}, at about {top.get('price')}, "
        f"with a match score of {top.get('score')}."
    )
