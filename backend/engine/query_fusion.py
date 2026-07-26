"""A_I / F_VL query fusion + multi-retriever RRF (Reciprocal Rank Fusion).

A_I / F_VL run online before recall so image-only requests get text keywords, and
joint text+image requests ground ambiguous language in the photo.

RRF merges ranked lists from keyword / text-semantic / visual retrievers:
  score(d) = Σ_i w_i / (k + rank_i(d)),  rank 1-based, default k=60.
"""

from __future__ import annotations

from typing import Any

from .llm.vision import describe_shopping_image, visual_context_text
from .models import PreferenceProfile, SessionState

# Cormack et al. RRF constant
RRF_K = 60


def default_fuse_weights(modality: str) -> dict[str, float]:
    m = (modality or "text").strip().lower()
    if m in ("image", "image_only"):
        return {"text": 0.35, "visual": 0.65}
    if m in ("text_image", "text+image", "multimodal"):
        return {"text": 0.55, "visual": 0.45}
    return {"text": 1.0, "visual": 0.0}


def normalize_fuse_weights(raw: dict[str, Any] | None, *, modality: str) -> dict[str, float]:
    base = default_fuse_weights(modality)
    if not raw:
        return base
    text_w = raw.get("text", base["text"])
    visual_w = raw.get("visual", base["visual"])
    try:
        text_w = float(text_w)
        visual_w = float(visual_w)
    except (TypeError, ValueError):
        return base
    total = text_w + visual_w
    if total <= 0:
        return base
    return {"text": text_w / total, "visual": visual_w / total}


def rrf_fuse(
    ranked_lists: list[list[dict[str, Any]]] | tuple[list[dict[str, Any]], ...],
    *,
    k: int = RRF_K,
    weights: list[float] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over already-ranked candidate lists.

    Each list is treated as rank-ordered (index 0 = rank 1). Items are merged by
    ``id``; score fields ``_text_score`` / ``_visual_score`` keep the max seen.
    Result is sorted by ``_rrf_score`` descending.
    """
    k = max(1, int(k or RRF_K))
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = {}

    for li, group in enumerate(ranked_lists or []):
        w = 1.0
        if weights is not None and li < len(weights):
            try:
                w = float(weights[li])
            except (TypeError, ValueError):
                w = 1.0
        if w <= 0:
            continue
        for rank, item in enumerate(group or [], start=1):
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "")
            if not pid:
                continue
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank)
            if pid not in items:
                items[pid] = dict(item)
                sources[pid] = []
            else:
                for key in ("_text_score", "_visual_score"):
                    cur = float(items[pid].get(key) or 0.0)
                    nxt = float(item.get(key) or 0.0)
                    if nxt > cur:
                        items[pid][key] = item.get(key)
                src = item.get("_recall_source")
                if src and src not in sources[pid]:
                    sources[pid].append(str(src))
            src0 = item.get("_recall_source")
            if src0 and str(src0) not in sources[pid]:
                sources[pid].append(str(src0))

    ordered = sorted(scores.keys(), key=lambda p: (-scores[p], p))
    out: list[dict[str, Any]] = []
    for pid in ordered:
        d = items[pid]
        d["_rrf_score"] = round(scores[pid], 6)
        if sources.get(pid):
            d["_recall_sources"] = list(sources[pid])
        out.append(d)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def rrf_fuse_score_lists(
    candidates: list[dict[str, Any]],
    score_keys: list[tuple[str, float]],
    *,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Build per-key ranked lists from score fields, then RRF-fuse.

    ``score_keys`` is ``[(field_name, weight), ...]``; zero/empty fields skip.
    """
    if not candidates:
        return []
    lists: list[list[dict[str, Any]]] = []
    weights: list[float] = []
    for key, weight in score_keys:
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        ranked = sorted(
            candidates,
            key=lambda c, kk=key: -float(c.get(kk) or 0.0),
        )
        # Skip lists where nobody has a positive signal
        if not any(float(c.get(key) or 0.0) > 0 for c in ranked):
            continue
        lists.append(ranked)
        weights.append(w)
    if not lists:
        # Preserve input order but still stamp a tiny RRF from a single pass
        return rrf_fuse([list(candidates)], k=k, weights=[1.0])
    return rrf_fuse(lists, k=k, weights=weights)


def apply_image_attributes(
    state: SessionState,
    query_image: tuple[bytes, str] | None,
    *,
    user_text: str = "",
) -> dict[str, Any]:
    """A_I: extract category / keywords / visual cues from the query image.

    Mutates ``state.preference`` in place when the VL call succeeds.
    """
    if not query_image:
        return {"ok": False, "reason": "no_image"}
    image_bytes, mime = query_image
    analysis = describe_shopping_image(
        image_bytes,
        mime_type=mime or "image/jpeg",
        user_text=user_text,
    )
    _merge_analysis_into_preference(state.preference, analysis, mode="a_i")
    return {
        "ok": True,
        "mode": "a_i",
        "provider": analysis.get("provider"),
        "category": state.preference.category,
        "keywords": list(state.preference.search_keywords[:8]),
        "visual_context": state.preference.visual_context[:200],
        "warning": analysis.get("warning") or "",
    }


def fuse_text_image_query(
    state: SessionState,
    query_image: tuple[bytes, str] | None,
) -> dict[str, Any]:
    """F_VL: text constrains image interpretation; image grounds text.

    Mutates ``state.preference`` with fused keywords / visual_context.
    """
    if not query_image:
        return {"ok": False, "reason": "no_image"}
    pref = state.preference
    user_text = (pref.raw_query or pref.use_case or " ".join(pref.search_keywords)).strip()
    image_bytes, mime = query_image
    analysis = describe_shopping_image(
        image_bytes,
        mime_type=mime or "image/jpeg",
        user_text=user_text,
    )
    _merge_analysis_into_preference(pref, analysis, mode="f_vl")
    return {
        "ok": True,
        "mode": "f_vl",
        "provider": analysis.get("provider"),
        "category": pref.category,
        "keywords": list(pref.search_keywords[:8]),
        "visual_context": pref.visual_context[:200],
        "warning": analysis.get("warning") or "",
    }


def _merge_analysis_into_preference(
    pref: PreferenceProfile,
    analysis: dict[str, Any],
    *,
    mode: str,
) -> None:
    cat = str(analysis.get("product_category") or "").strip()
    if cat and (not pref.category or mode == "a_i"):
        pref.category = cat

    keywords = analysis.get("search_keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    merged_kw: list[str] = []
    for k in list(pref.search_keywords or []) + [str(x) for x in keywords]:
        k = k.strip()
        if k and k.lower() not in {x.lower() for x in merged_kw}:
            merged_kw.append(k)
    if merged_kw:
        pref.search_keywords = merged_kw[:12]

    soft = analysis.get("soft_preferences") or analysis.get("visual_preferences") or []
    if isinstance(soft, str):
        soft = [soft]
    for s in soft:
        s = str(s).strip()
        if s and s not in pref.soft:
            pref.soft.append(s)

    hard = analysis.get("hard_constraints") or []
    if isinstance(hard, str):
        hard = [hard]
    for h in hard:
        h = str(h).strip()
        if h and h not in pref.hard:
            # Image-inferred hard constraints stay soft unless user confirmed.
            if mode == "a_i" and h not in pref.soft:
                pref.soft.append(f"visual:{h}")
            elif mode == "f_vl" and h not in pref.soft:
                pref.soft.append(f"visual:{h}")

    ctx = visual_context_text(analysis)
    if ctx:
        if pref.visual_context and mode == "f_vl":
            pref.visual_context = f"{pref.visual_context}; {ctx}"[:700]
        else:
            pref.visual_context = ctx[:700]

    if mode == "a_i" and not (pref.raw_query or "").strip():
        pref.raw_query = (cat or "product from photo").strip()
