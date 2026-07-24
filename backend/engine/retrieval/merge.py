"""Merge and deduplicate candidates from text and visual retrieval."""

from __future__ import annotations

from typing import Any, Iterable


def merge_candidates(
    text_candidates: Iterable[dict[str, Any]],
    visual_candidates: Iterable[dict[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Union both retrieval result sets while preserving per-source scores."""
    merged: dict[str, dict[str, Any]] = {}
    for source, candidates in (
        ("text", text_candidates),
        ("visual", visual_candidates),
    ):
        for rank, item in enumerate(candidates, start=1):
            key = _candidate_key(item)
            if not key:
                continue
            current = merged.get(key)
            if current is None:
                current = dict(item)
                current.pop("image_embedding", None)
                current.pop("review_embedding", None)
                current["retrieval"] = dict(item.get("retrieval") or {})
                current["retrieval"]["sources"] = []
                current["retrieval"]["source_ranks"] = {}
                merged[key] = current
            else:
                _merge_missing_fields(current, item)

            retrieval = current["retrieval"]
            if source not in retrieval["sources"]:
                retrieval["sources"].append(source)
            retrieval["source_ranks"][source] = rank
            incoming = item.get("retrieval")
            if isinstance(incoming, dict):
                for meta_key, value in incoming.items():
                    if meta_key not in {"sources", "source_ranks"}:
                        retrieval[meta_key] = value

            score_key = f"{source}_similarity"
            score = _clamp(item.get(score_key))
            current[score_key] = max(_clamp(current.get(score_key)), score)

    for item in merged.values():
        text_score = _clamp(item.get("text_similarity"))
        visual_score = _clamp(item.get("visual_similarity"))
        sources = item["retrieval"]["sources"]
        if len(sources) == 2:
            score = min(1.0, 0.50 * text_score + 0.50 * visual_score + 0.05)
        elif sources == ["visual"]:
            score = visual_score
        else:
            score = text_score
        item["retrieval_score"] = round(score, 6)
        item["retrieval"]["retrieval_score"] = item["retrieval_score"]
        if "text" in sources:
            item["retrieval"]["text_similarity"] = text_score
        else:
            item["retrieval"].pop("text_similarity", None)
        if "visual" in sources:
            item["retrieval"]["visual_similarity"] = visual_score
        else:
            item["retrieval"].pop("visual_similarity", None)

    ranked = sorted(
        merged.values(),
        key=lambda item: (
            _clamp(item.get("retrieval_score")),
            len((item.get("retrieval") or {}).get("sources") or []),
            _number(item.get("rating")),
            _number(item.get("rating_number")),
        ),
        reverse=True,
    )
    return ranked[:max(0, limit)]


def _candidate_key(item: dict[str, Any]) -> str:
    product_id = str(item.get("id") or "").strip().lower()
    if product_id:
        return f"id:{product_id}"
    name = " ".join(str(item.get("name") or "").lower().split())
    store = " ".join(str(item.get("store") or "").lower().split())
    return f"name:{name}|{store}" if name else ""


def _merge_missing_fields(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key in {"retrieval", "image_embedding", "review_embedding"}:
            continue
        if target.get(key) in (None, "", [], {}):
            target[key] = value


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
