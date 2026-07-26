"""Executor: run compiled Plan stages (Recall → Verify → Recommend)."""

from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..exp_config import CONFIG, ExpConfig, load_config
from ..models import PreferenceProfile, RankedProduct, RecommendationBundle, SessionState
from ..verifier import verify_candidates
from .plan_schema import (
    AGENT_RECOMMEND,
    AGENT_RERANK_EXISTING,
    AGENT_TEXT_RECALL,
    AGENT_VERIFY,
    AGENT_VISUAL_RECALL,
)
from .recall_worker import RecallAgent
from .recommend_worker import rank_products
from .search_worker import run_search


@dataclass
class ExecuteResult:
    ok: bool
    bundle: RecommendationBundle | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    need_replan: bool = False
    reject_reason: str = ""
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    stage_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "bundle": self.bundle.to_dict() if self.bundle else None,
            "candidate_count": len(self.candidates),
            "rejected": self.rejected,
            "need_replan": self.need_replan,
            "reject_reason": self.reject_reason,
            "memory_writes": self.memory_writes,
            "error": self.error,
            "stage_trace": self.stage_trace,
        }


class Executor:
    def __init__(
        self,
        search_fn: Callable[[dict], list[dict]],
        *,
        recall_agent: RecallAgent | None = None,
    ) -> None:
        self.search_fn = search_fn
        self.recall = recall_agent or RecallAgent(search_fn)

    def run(
        self,
        plan: dict[str, Any],
        state: SessionState,
        *,
        cancel: threading.Event | None = None,
        config: ExpConfig | None = None,
        memory_write_builder: Callable[[SessionState, RecommendationBundle], list[dict]] | None = None,
    ) -> ExecuteResult:
        cfg = config or load_config()
        cancel = cancel or threading.Event()
        result = ExecuteResult(ok=False)
        if plan.get("refused"):
            result.error = plan.get("message") or "refused"
            return result

        stages = plan.get("stages") or []
        if not stages:
            result.error = "empty plan"
            return result

        text_cands: list[dict] = []
        visual_cands: list[dict] = []
        candidates: list[dict] = []
        focus_ids = _focus_ids(plan)
        query_hint = (
            plan.get("query_hint")
            or (plan.get("hints") or {}).get("query_hint")
            or ""
        )
        query_image = _load_query_image(state)

        for stage in stages:
            if cancel.is_set():
                result.error = "cancelled"
                return result
            agent = str(stage.get("agent") or "")
            params = stage.get("params") or {}
            try:
                if agent == AGENT_TEXT_RECALL:
                    # Prefer planner query_hint by temporarily injecting keywords if empty.
                    pref = state.preference
                    if query_hint and not pref.search_keywords:
                        pref = PreferenceProfile(**{**pref.to_dict(), "search_keywords": query_hint.split()})
                    text_cands = run_search(
                        pref,
                        self.search_fn,
                        query_hint=params.get("query_hint") or query_hint,
                    )
                    result.stage_trace.append({"agent": agent, "count": len(text_cands)})
                elif agent == AGENT_VISUAL_RECALL:
                    from ..modality_router import RoutePlan

                    route = RoutePlan(
                        modality=str((plan.get("decision") or {}).get("route", {}).get("modality") or "text"),
                        do_text_recall=False,
                        do_visual_recall=True,
                        infer_category_from_image=bool(params.get("infer_category_from_image")),
                        reverse_verify=False,
                        do_verify=False,
                        reason="executor-visual",
                    )
                    visual_cands = self.recall.image_recall(
                        query_image,
                        state.preference.category if route.infer_category_from_image else None,
                        cfg,
                    )
                    result.stage_trace.append({"agent": agent, "count": len(visual_cands)})
                elif agent == AGENT_VERIFY:
                    candidates = _merge(text_cands, visual_cands) if not candidates else candidates
                    if query_image:
                        try:
                            from .. import enrichment

                            if enrichment.has_enrichment():
                                enrichment.attach_visual_scores(
                                    candidates, query_image[0], mime=query_image[1]
                                )
                        except Exception:
                            pass
                    verified = verify_candidates(state.preference, candidates, cfg)
                    candidates = list(verified.kept)
                    result.rejected = list(verified.rejected)
                    result.stage_trace.append(
                        {
                            "agent": agent,
                            "kept": len(candidates),
                            "rejected": len(result.rejected),
                            "method": verified.method,
                        }
                    )
                    if not candidates and result.rejected:
                        max_replans = int((plan.get("loop") or {}).get("max_replans") or 0)
                        attempt = int((plan.get("decision") or {}).get("replan_attempt") or 0)
                        if cfg.planner_replan and attempt < max_replans:
                            result.need_replan = True
                            result.reject_reason = _summarize_rejects(result.rejected)
                            return result
                elif agent == AGENT_RECOMMEND:
                    candidates = _merge(text_cands, visual_cands) if not candidates else candidates
                    result.candidates = candidates
                    bundle = rank_products(plan["plan_id"], state.preference, candidates)
                    if focus_ids:
                        bundle = _boost_focus(bundle, focus_ids)
                    result.bundle = bundle
                    result.ok = True
                    result.stage_trace.append(
                        {"agent": agent, "top": [r.id for r in bundle.ranked[:5]]}
                    )
                elif agent == AGENT_RERANK_EXISTING:
                    bundle = _rerank_existing(
                        plan["plan_id"], state, focus_ids=focus_ids
                    )
                    result.bundle = bundle
                    result.candidates = [
                        {"id": r.id, "name": r.name, "price": r.price}
                        for r in (bundle.ranked if bundle else [])
                    ]
                    result.ok = True
                    result.stage_trace.append(
                        {
                            "agent": agent,
                            "top": [r.id for r in bundle.ranked[:5]] if bundle else [],
                        }
                    )
                else:
                    result.stage_trace.append({"agent": agent, "error": "unknown_agent"})
            except Exception as exc:
                result.error = f"{agent}: {exc}"
                return result

        if result.ok and result.bundle and memory_write_builder is not None:
            try:
                result.memory_writes = list(memory_write_builder(state, result.bundle) or [])
            except Exception:
                result.memory_writes = []
        elif result.ok and result.bundle:
            # Minimal default writeback: refresh last_ranked for reference resolution.
            result.memory_writes = [
                {
                    "key": "last_ranked",
                    "value": [
                        {"id": r.id, "name": r.name, "index": i}
                        for i, r in enumerate(result.bundle.ranked[:10])
                    ],
                    "scope": "session",
                }
            ]
            # Merge planner-declared writes.
            planned = (plan.get("writeback") or {}).get("memory_write") or []
            if isinstance(planned, list):
                result.memory_writes = planned + result.memory_writes

        return result


def _focus_ids(plan: dict[str, Any]) -> list[str]:
    decision = plan.get("decision") or {}
    mem = decision.get("memory") or {}
    ids = mem.get("resolved_refs") or []
    out = [str(x) for x in ids if x]
    for stage in plan.get("stages") or []:
        params = stage.get("params") or {}
        for x in params.get("focus_product_ids") or []:
            if str(x) not in out:
                out.append(str(x))
    return out


def _merge(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            pid = str(item.get("id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            merged.append(item)
    return merged


def _summarize_rejects(rejected: list[dict[str, Any]]) -> str:
    reasons = []
    for item in rejected[:5]:
        reasons.append(str(item.get("reason") or "rejected"))
    return "; ".join(reasons) or "all candidates rejected"


def _boost_focus(bundle: RecommendationBundle, focus_ids: list[str]) -> RecommendationBundle:
    if not focus_ids or not bundle.ranked:
        return bundle
    order = {pid: i for i, pid in enumerate(focus_ids)}
    bundle.ranked.sort(key=lambda r: order.get(r.id, 10_000))
    return bundle


def _rerank_existing(
    plan_id: str,
    state: SessionState,
    *,
    focus_ids: list[str] | None = None,
) -> RecommendationBundle:
    prev = state.worker.last_bundle
    if not prev or not prev.ranked:
        return RecommendationBundle(
            plan_id=plan_id,
            ranked=[],
            summary="No previous matches to refine.",
            status="ready",
        )
    # Convert ranked products back to candidate dicts for scoring.
    candidates = [
        {
            "id": r.id,
            "name": r.name,
            "price": r.price,
            "rating": r.rating,
            "platform": r.platform,
            "display": r.display,
            "performance": r.performance,
            "weight_kg": r.weight_kg,
            "summary": r.summary,
            "image_url": r.image_url,
        }
        for r in prev.ranked
    ]
    if focus_ids:
        focused = [c for c in candidates if str(c["id"]) in set(focus_ids)]
        if focused:
            # Keep focus items first, then the rest for context.
            rest = [c for c in candidates if str(c["id"]) not in set(focus_ids)]
            candidates = focused + rest
    bundle = rank_products(plan_id, state.preference, candidates, top_n=max(6, len(candidates)))
    return bundle


def _load_query_image(state: SessionState) -> tuple[bytes, str] | None:
    refs = getattr(state, "image_refs", None) or []
    for ref in reversed(refs):
        if not isinstance(ref, dict):
            continue
        mime = str(ref.get("mime_type") or ref.get("mime") or "image/jpeg")
        path = ref.get("path")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                if raw:
                    return raw, mime
            except OSError:
                pass
        data_url = ref.get("data_url") or ""
        if isinstance(data_url, str) and data_url.startswith("data:") and "," in data_url:
            try:
                header, b64 = data_url.split(",", 1)
                if ";base64" in header:
                    raw = base64.b64decode(b64)
                    if raw:
                        return raw, mime
            except Exception:
                continue
    return None
