"""Planner control plane: gates + compile Plan DAG. Does not execute stages."""

from __future__ import annotations

import uuid
from typing import Any

from .. import enrichment
from ..exp_config import CONFIG, ExpConfig, load_config
from ..intent import preference_search_query
from ..llm.qwen_client import chat_json, qwen_configured
from ..memory_store import MEMORY_STORE, MemoryStore
from ..modality_router import route as modality_route
from ..models import PreferenceProfile, SessionState
from ..talker.shopping_safety import assess_shopping_risk, shopping_safety_enabled
from .plan_schema import Plan, PlanHints, build_dag

_VALID_INTENTS = frozenset({"new_search", "refine", "followup", "compare"})


def _fallback_query_hint(pref: PreferenceProfile) -> str:
    if pref.search_keywords:
        return " ".join(pref.search_keywords)
    derived = preference_search_query(pref)
    if derived.strip():
        return derived.strip()
    return pref.use_case or ""


def _latest_utterance(session: SessionState, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    for turn in reversed(session.conversation or []):
        if turn.get("role") == "user" and (turn.get("text") or "").strip():
            return str(turn["text"]).strip()
    return (session.preference.raw_query or "").strip()


def _rule_intent(utterance: str, session: SessionState) -> str:
    text = (utterance or "").lower()
    if any(k in text for k in ("对比", "比较", "compare", "vs", "versus", "哪个好")):
        return "compare"
    if session.worker.last_bundle and any(
        k in text
        for k in (
            "更",
            "便宜",
            "贵一点",
            "换",
            "改",
            "只要",
            "不要",
            "再",
            "cheaper",
            "instead",
            "prefer",
            "only",
            "without",
            "refine",
        )
    ):
        return "refine"
    if session.worker.last_bundle and any(
        k in text
        for k in (
            "第二个",
            "第一个",
            "这个",
            "刚才",
            "那个",
            "详情",
            "说说",
            "tell me more",
            "the second",
            "this one",
            "that one",
        )
    ):
        return "followup"
    return "new_search"


def _llm_intent(utterance: str, session: SessionState) -> str | None:
    if not qwen_configured():
        return None
    try:
        data = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the shopper turn for a shopping planner. "
                        "Return ONLY JSON: {\"intent\": \"new_search|refine|followup|compare\"}. "
                        "new_search=fresh product hunt; refine=change constraints on current hunt; "
                        "followup=ask about existing candidates; compare=compare listed options."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"utterance={utterance}\n"
                        f"has_bundle={bool(session.worker.last_bundle)}\n"
                        f"preference={session.preference.to_dict()}\n"
                        f"recent={[t for t in session.conversation[-4:]]}"
                    ),
                },
            ],
            temperature=0.0,
        )
        intent = str((data or {}).get("intent") or "").strip()
        return intent if intent in _VALID_INTENTS else None
    except Exception:
        return None


def _llm_hints(
    session: SessionState,
    *,
    intent: str,
    replan_ctx: dict[str, Any] | None,
) -> PlanHints:
    pref = session.preference
    replan_note = ""
    if replan_ctx:
        replan_note = (
            f"REPLAN attempt={replan_ctx.get('attempt')} "
            f"reject_reason={replan_ctx.get('reject_reason')}. "
            "Propose relax_ops from: drop_hard, drop_soft, raise_budget_20, "
            "drop_platform, broaden_keywords."
        )
    try:
        data = chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Planner agent for VoiceShop++. "
                        "Catalog is ENGLISH. Return ONLY JSON with keys: "
                        "query_hint (2-6 ENGLISH catalog keywords, never Chinese), "
                        "reason (short), focus (string array), "
                        "fuse_weights (optional object with text/visual floats), "
                        "relax_ops (string array, especially on replan), "
                        "intent (new_search|refine|followup|compare). "
                        "Do not invent product names."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"intent_guess={intent}\n"
                        f"preference={pref.to_dict()}\n"
                        f"memory_summary={session.memory_summary}\n"
                        f"recent={[t for t in session.conversation[-6:]]}\n"
                        f"{replan_note}"
                    ),
                },
            ],
            temperature=0.2,
        )
        hints = PlanHints.from_dict(data if isinstance(data, dict) else {})
        if hints.query_hint and not any(
            ch.isascii() and ch.isalpha() for ch in hints.query_hint
        ):
            hints.query_hint = ""
        return hints
    except Exception:
        return PlanHints()


def _apply_relax_ops(pref: PreferenceProfile, relax_ops: list[str]) -> list[str]:
    applied: list[str] = []
    for op in relax_ops or []:
        name = str(op).strip().lower()
        if name == "drop_hard" and pref.hard:
            pref.hard = []
            applied.append(name)
        elif name == "drop_soft" and pref.soft:
            pref.soft = []
            applied.append(name)
        elif name == "drop_platform" and pref.platform not in ("", "No preference"):
            pref.platform = "No preference"
            applied.append(name)
        elif name == "raise_budget_20" and pref.budget:
            pref.budget = int(pref.budget * 1.2)
            applied.append(name)
        elif name == "broaden_keywords" and len(pref.search_keywords) > 1:
            pref.search_keywords = pref.search_keywords[: max(1, len(pref.search_keywords) // 2)]
            applied.append(name)
    return applied


def _capability_budget(cfg: ExpConfig) -> dict[str, Any]:
    has_vis = bool(cfg.enrichment and enrichment.has_enrichment())
    return {
        "enrichment_ready": has_vis,
        "visual_recall_ready": bool(cfg.visual_recall and has_vis),
        "reviews_ready": bool(cfg.reviews),
        "visual_top_k": cfg.visual_top_k,
        "max_replans": cfg.max_replans if cfg.planner_replan else 0,
        "verifier": cfg.verifier,
    }


def plan(
    session: SessionState,
    *,
    utterance: str | None = None,
    user_id: str | None = None,
    opted_in_memory: bool = False,
    replan_ctx: dict[str, Any] | None = None,
    memory: MemoryStore | None = None,
    config: ExpConfig | None = None,
) -> dict[str, Any]:
    """Compile an executable Plan dict (control plane only)."""
    cfg = config or load_config()
    store = memory or MEMORY_STORE
    plan_id = uuid.uuid4().hex[:10]
    text = _latest_utterance(session, utterance)

    # 1) Safety gate — hard refuse only for dangerous/restricted shopping.
    safety = {"category": "normal", "reasons": [], "refused": False}
    if shopping_safety_enabled():
        risk = assess_shopping_risk(text or session.preference.raw_query or "")
        safety = {
            "category": risk.category,
            "reasons": list(risk.reasons),
            "refused": risk.category == "dangerous_or_restricted",
        }
        if safety["refused"]:
            out = Plan(
                plan_id=plan_id,
                refused=True,
                message=(
                    "I can't help with restricted or dangerous purchases. "
                    "Please ask about lawful consumer products instead."
                ),
                planner="rules",
                decision={"safety": safety, "intent": "refuse"},
                loop={"on_reject": "relax_and_replan", "max_replans": 0},
            )
            return out.to_dict()

    # 2) Memory prefill (before intent/shortcircuit so refs work).
    memory_meta: dict[str, Any] = {"enabled": cfg.memory, "prefilled_keys": [], "resolved_refs": []}
    if cfg.memory:
        memory_meta.update(
            store.prefill(
                session,
                user_id=user_id,
                opted_in_memory=opted_in_memory,
                config=cfg,
            )
        )
        refs = store.resolve_refs(
            text,
            session,
            config=cfg,
            use_llm=bool(cfg.planner_llm),
        )
        memory_meta["resolved_refs"] = refs.get("product_ids") or []
        memory_meta["resolve_method"] = refs.get("method")
        memory_meta["opted_in"] = bool(opted_in_memory and user_id)

    # 3) Intent
    intent = _rule_intent(text, session)
    planner_mode = "rules"
    if cfg.planner_llm and qwen_configured():
        llm_intent = _llm_intent(text, session)
        if llm_intent:
            intent = llm_intent
            planner_mode = "llm"

    # 4) Replan relax
    applied_relax: list[str] = []
    if replan_ctx and isinstance(replan_ctx.get("relax_ops"), list):
        applied_relax = _apply_relax_ops(session.preference, replan_ctx["relax_ops"])
    elif replan_ctx and replan_ctx.get("reject_reason"):
        # Default gentle relax when verifier rejected everything.
        applied_relax = _apply_relax_ops(
            session.preference, ["drop_hard", "raise_budget_20", "broaden_keywords"]
        )

    # 5) LLM PlanHints
    hints = PlanHints(query_hint=_fallback_query_hint(session.preference), intent=intent)
    if cfg.planner_llm and qwen_configured():
        llm_hints = _llm_hints(session, intent=intent, replan_ctx=replan_ctx)
        if llm_hints.query_hint:
            hints.query_hint = llm_hints.query_hint
        if llm_hints.focus:
            hints.focus = llm_hints.focus
        if llm_hints.fuse_weights:
            hints.fuse_weights = llm_hints.fuse_weights
        if llm_hints.relax_ops:
            hints.relax_ops = llm_hints.relax_ops
        if llm_hints.reason:
            hints.reason = llm_hints.reason
        if llm_hints.intent in _VALID_INTENTS:
            intent = llm_hints.intent
            hints.intent = intent
        planner_mode = "llm"
        if replan_ctx and llm_hints.relax_ops and not applied_relax:
            applied_relax = _apply_relax_ops(session.preference, llm_hints.relax_ops)
    if not hints.query_hint:
        hints.query_hint = _fallback_query_hint(session.preference)
    if not hints.reason:
        hints.reason = f"Pipeline for intent={intent}"
    hints.intent = intent

    # 6) Capability / budget
    budget = _capability_budget(cfg)
    route_plan = modality_route(session, cfg)
    if not budget["visual_recall_ready"]:
        # Compile-time capability gate: disable visual stage even if router asked.
        route_plan.do_visual_recall = False

    focus_ids = list(memory_meta.get("resolved_refs") or [])
    has_bundle = bool(session.worker.last_bundle and session.worker.last_bundle.ranked)
    shortcircuit = bool(
        cfg.intent_shortcircuit
        and intent in ("refine", "followup", "compare")
        and has_bundle
    )
    if shortcircuit:
        planner_mode = "light" if planner_mode == "rules" else planner_mode

    stages = build_dag(
        route_plan=route_plan,
        shortcircuit=shortcircuit,
        do_verify=cfg.verifier_enabled,
        focus_product_ids=focus_ids,
        query_hint=hints.query_hint,
        visual_top_k=budget["visual_top_k"],
    )

    writeback = {
        "memory_write": [
            {
                "key": "preference_slots",
                "value": {
                    k: getattr(session.preference, k)
                    for k in (
                        "category",
                        "budget",
                        "use_case",
                        "platform",
                        "hard",
                        "soft",
                        "search_keywords",
                    )
                    if getattr(session.preference, k, None)
                    not in (None, "", [], "No preference", "Not required")
                },
                "scope": "session",
            }
        ],
        "opted_in_memory": bool(opted_in_memory),
        "user_id": user_id or "",
    }

    out = Plan(
        plan_id=plan_id,
        stages=stages,
        loop={
            "on_reject": "relax_and_replan",
            "max_replans": budget["max_replans"],
            "reject_threshold": 1.0,
        },
        decision={
            "safety": safety,
            "intent": intent,
            "memory": memory_meta,
            "budget": budget,
            "route": route_plan.to_dict(),
            "shortcircuit": shortcircuit,
            "applied_relax": applied_relax,
            "replan_attempt": int((replan_ctx or {}).get("attempt") or 0),
        },
        writeback=writeback,
        hints=hints,
        refused=False,
        message=hints.reason,
        planner=planner_mode,
        query_hint=hints.query_hint,
        reason=hints.reason,
    )
    return out.to_dict()
