"""Executor: run compiled Plan stages (Recall → Verify → Recommend)."""

from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..conflicts import ConflictPacket
from ..exp_config import CONFIG, ExpConfig, load_config
from ..models import PreferenceProfile, RecommendationBundle, SessionState
from ..query_fusion import (
    apply_image_attributes,
    fuse_text_image_query,
    normalize_fuse_weights,
    rrf_fuse,
)
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


@dataclass
class ExecuteResult:
    ok: bool
    bundle: RecommendationBundle | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
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
            "conflicts": self.conflicts,
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
        verify_conflicts: list = []
        verify_unresolved: list = []
        focus_ids = _focus_ids(plan)
        query_hint = (
            plan.get("query_hint")
            or (plan.get("hints") or {}).get("query_hint")
            or ""
        )
        query_image = _load_query_image(state)
        decision = plan.get("decision") or {}
        route = decision.get("route") or {}
        modality = str(route.get("modality") or "text")
        hints = plan.get("hints") or {}
        fuse_weights = normalize_fuse_weights(
            hints.get("fuse_weights") or decision.get("fuse_weights"),
            modality=modality,
        )

        # A_I / F_VL before recall so keywords & category are ready.
        if query_image and not _is_shortcircuit(stages):
            if bool(route.get("infer_category_from_image")) or modality == "image":
                meta = apply_image_attributes(state, query_image)
                result.stage_trace.append({"agent": "a_i", **{k: meta.get(k) for k in ("ok", "category", "keywords", "warning")}})
            elif modality in ("text_image", "text+image"):
                meta = fuse_text_image_query(state, query_image)
                result.stage_trace.append({"agent": "f_vl", **{k: meta.get(k) for k in ("ok", "category", "keywords", "warning")}})

        for stage in stages:
            if cancel.is_set():
                result.error = "cancelled"
                return result
            agent = str(stage.get("agent") or "")
            params = stage.get("params") or {}
            try:
                if agent == AGENT_TEXT_RECALL:
                    pref = state.preference
                    if query_hint and not pref.search_keywords:
                        pref = PreferenceProfile(
                            **{**pref.to_dict(), "search_keywords": query_hint.split()}
                        )
                    text_cands = self.recall.text_recall(
                        pref,
                        cfg,
                        query_hint=params.get("query_hint") or query_hint,
                    )
                    result.stage_trace.append({"agent": agent, "count": len(text_cands)})
                elif agent == AGENT_VISUAL_RECALL:
                    cat = None
                    if params.get("infer_category_from_image") or modality == "image":
                        cat = state.preference.category or None
                    visual_cands = self.recall.image_recall(query_image, cat, cfg)
                    result.stage_trace.append({"agent": agent, "count": len(visual_cands)})
                elif agent == AGENT_VERIFY:
                    candidates = _merge(
                        text_cands,
                        visual_cands,
                        weights=[fuse_weights.get("text", 1.0), fuse_weights.get("visual", 1.0)],
                    ) if not candidates else candidates
                    try:
                        from .. import enrichment
                        from ..intent import preference_search_query

                        if query_image and enrichment.has_enrichment():
                            enrichment.attach_visual_scores(
                                candidates, query_image[0], mime=query_image[1]
                            )
                        q = (
                            " ".join(state.preference.search_keywords).strip()
                            or (query_hint or "").strip()
                            or preference_search_query(state.preference)
                        )
                        if q and enrichment.has_text_embeddings():
                            enrichment.attach_text_scores(
                                candidates,
                                q,
                                provider=getattr(cfg, "embedding_provider", None),
                            )
                    except Exception:
                        pass
                    reverse = bool(params.get("reverse_verify") or route.get("reverse_verify"))
                    verified = verify_candidates(
                        state.preference,
                        candidates,
                        cfg,
                        reverse_verify=reverse,
                    )
                    candidates = list(verified.kept)
                    result.rejected = list(getattr(verified, "rejected", None) or [])
                    verify_conflicts = list(getattr(verified, "conflicts", None) or [])
                    verify_unresolved = list(getattr(verified, "unresolved", None) or [])
                    result.conflicts = [
                        c.to_dict() if hasattr(c, "to_dict") else c
                        for c in verify_conflicts
                    ]
                    result.stage_trace.append(
                        {
                            "agent": agent,
                            "kept": len(candidates),
                            "rejected": len(result.rejected),
                            "conflicts": len(verify_conflicts),
                            "unresolved": len(verify_unresolved),
                            "method": verified.method,
                            "reverse_verify": reverse,
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
                    candidates = _merge(
                        text_cands,
                        visual_cands,
                        weights=[fuse_weights.get("text", 1.0), fuse_weights.get("visual", 1.0)],
                    ) if not candidates else candidates
                    if query_image and candidates:
                        try:
                            from .. import enrichment

                            if enrichment.has_enrichment():
                                enrichment.attach_visual_scores(
                                    candidates, query_image[0], mime=query_image[1]
                                )
                        except Exception:
                            pass
                    result.candidates = candidates
                    bundle = rank_products(
                        plan["plan_id"],
                        state.preference,
                        candidates,
                        fuse_weights=fuse_weights,
                        modality=modality,
                        conflicts=verify_conflicts,
                        unresolved=verify_unresolved,
                    )
                    if focus_ids:
                        bundle = _boost_focus(bundle, focus_ids)
                    result.bundle = bundle
                    result.conflicts = list(bundle.conflicts or result.conflicts)
                    result.ok = True
                    result.stage_trace.append(
                        {
                            "agent": agent,
                            "top": [r.id for r in bundle.ranked[:5]],
                            "fuse_weights": fuse_weights,
                        }
                    )
                elif agent == AGENT_RERANK_EXISTING:
                    bundle = _rerank_existing(
                        plan["plan_id"],
                        state,
                        focus_ids=focus_ids,
                        fuse_weights=fuse_weights,
                        modality=modality,
                    )
                    result.bundle = bundle
                    result.candidates = [
                        {"id": r.id, "name": r.name, "price": r.price}
                        for r in (bundle.ranked if bundle else [])
                    ]
                    result.conflicts = list(bundle.conflicts or [])
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
            planned = (plan.get("writeback") or {}).get("memory_write") or []
            if isinstance(planned, list):
                result.memory_writes = planned + result.memory_writes

        return result


def _is_shortcircuit(stages: list[dict[str, Any]]) -> bool:
    return any(str(s.get("agent") or "") == AGENT_RERANK_EXISTING for s in stages)


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


def _merge(*groups: list[dict], weights: list[float] | None = None) -> list[dict]:
    """Merge ranked recall lists with RRF (falls back to empty → [])."""
    lists = [g for g in groups if g]
    if not lists:
        return []
    ws = None
    if weights is not None:
        ws = []
        for i, g in enumerate(groups):
            if not g:
                continue
            ws.append(float(weights[i]) if i < len(weights) else 1.0)
    return rrf_fuse(lists, weights=ws)


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
    fuse_weights: dict[str, float] | None = None,
    modality: str = "text",
) -> RecommendationBundle:
    prev = state.worker.last_bundle
    if not prev or not prev.ranked:
        return RecommendationBundle(
            plan_id=plan_id,
            ranked=[],
            summary="No previous matches to refine.",
            status="ready",
            talker_brief="No previous matches to refine. Tell me what to change.",
        )
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
            rest = [c for c in candidates if str(c["id"]) not in set(focus_ids)]
            candidates = focused + rest
    return rank_products(
        plan_id,
        state.preference,
        candidates,
        top_n=max(6, len(candidates)),
        fuse_weights=fuse_weights,
        modality=modality,
        conflicts=[
            ConflictPacket.from_dict(c)
            for c in (prev.conflicts or [])
            if isinstance(c, dict)
        ],
    )


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
