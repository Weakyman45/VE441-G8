from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ..llm.qwen_client import chat_json, qwen_configured
from ..intent import preference_search_query
from ..models import SessionState


def _fallback_query_hint(pref) -> str:
    """Best catalog query without an LLM. The catalog is English, so prefer the
    English search keywords / derived query over the (possibly Chinese) raw text."""
    if pref.search_keywords:
        return " ".join(pref.search_keywords)
    derived = preference_search_query(pref)
    if derived.strip():
        return derived.strip()
    return pref.use_case or ""


@dataclass
class Task:
    name: str
    depends_on: list[str]


def plan(session: SessionState) -> dict[str, Any]:
    """
    Qwen-assisted plan when DASHSCOPE is configured; always falls back to
    the fixed search → recommend pipeline so Workers keep running offline.
    """
    plan_id = uuid.uuid4().hex[:10]
    tasks = [
        Task(name="search", depends_on=[]),
        Task(name="recommend", depends_on=["search"]),
    ]
    base = {
        "plan_id": plan_id,
        "tasks": [{"name": t.name, "depends_on": t.depends_on} for t in tasks],
        "reason": "Fixed pipeline: search → recommend",
        "query_hint": _fallback_query_hint(session.preference),
        "planner": "rules",
    }
    if not qwen_configured():
        return base

    pref = session.preference
    try:
        data = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Planner agent for VoiceShop++, a general-purpose "
                        "shopping assistant (any product category, not only laptops). "
                        "The product catalog is in ENGLISH. Return ONLY a JSON object with keys: "
                        "query_hint (SHORT ENGLISH search keywords for the catalog — translate "
                        "the user's need into 2-6 English words describing the product and its key "
                        "features, e.g. 'white running shoes breathable'. Never output Chinese here.), "
                        "reason (one short sentence), "
                        "focus (array of soft constraints like color/size/material/os). "
                        "Do not invent product names."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User preference JSON:\n"
                        f"{pref.to_dict()}\n"
                        f"Recent turns: {[t for t in session.conversation[-6:]]}"
                    ),
                },
            ],
            temperature=0.2,
        )
        if isinstance(data.get("query_hint"), str) and data["query_hint"].strip():
            hint = data["query_hint"].strip()
            # The catalog is English; if the model returned mostly non-ASCII
            # (Chinese) text it won't match, so keep the English fallback.
            if any(ch.isascii() and ch.isalpha() for ch in hint):
                base["query_hint"] = hint
        if isinstance(data.get("reason"), str) and data["reason"].strip():
            base["reason"] = data["reason"].strip()
        if isinstance(data.get("focus"), list):
            base["focus"] = [str(x) for x in data["focus"][:8]]
        base["planner"] = "qwen"
    except Exception as exc:
        base["planner"] = "rules"
        base["planner_error"] = str(exc)[:200]
    return base
