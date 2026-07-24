"""Recommend Agent for already retrieved and verified product candidates.

This module intentionally does not plan tasks, retrieve products, compare raw
embeddings, or decide whether a product satisfies a must-have.  Those are the
responsibilities of the upstream Planner, Retrieval, and Reject/Verify agents.

Expected optional candidate annotations from those agents:

    verified: bool
    rejection_reason: str
    visual_similarity: float  # normalized to 0..1 by Visual Retrieval
    text_similarity: float    # used only as a deterministic tie-breaker

The ranker also understands the enrichment columns in the new catalog.db:
``visual_attrs``, ``enriched_text``, ``review_aspects``, and
``review_count_used``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from .llm.qwen_client import chat_completion, qwen_configured
from .models import PreferenceProfile, RankedProduct, RecommendationBundle


# The five inputs in the architecture sketch.  Unavailable criteria are removed
# and the remaining weights are normalized per candidate.
RANKING_WEIGHTS: dict[str, float] = {
    "visual_similarity": 0.30,
    "nice_to_have": 0.25,
    "quality": 0.20,
    "price": 0.15,
    "preference_history": 0.10,
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "has",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "this", "to", "want", "with", "your", "need", "prefer", "preference",
    "product", "products", "item", "items", "good", "best", "nice", "use",
    "case", "budget", "under", "below", "around", "about", "rmb", "usd",
}


def rank_products(
    plan_id: str,
    profile: PreferenceProfile,
    candidates: list[dict[str, Any]],
    top_n: int = 40,
) -> RecommendationBundle:
    """Rank verified candidates using only Recommend-Agent responsibilities.

    Candidates without a verification annotation are accepted for compatibility
    with the current Search worker.  An explicit rejection is always honored.
    """
    scored: list[
        tuple[float, float, float, int, dict[str, Any], list[str], dict[str, Any], dict[str, float]]
    ] = []
    excluded: list[dict[str, str]] = []

    for item in _deduplicate(candidates):
        rejection = _upstream_rejection(item)
        if rejection:
            excluded.append({"id": str(item.get("id", "")), "reason": rejection})
            continue

        candidate_text = _candidate_text(item)
        visual, visual_available = _visual_score(profile, item, candidate_text)
        nice, nice_available, nice_matches = _preference_score(profile.soft, candidate_text)
        quality, review = _quality_score(item)
        price, price_available = _price_score(profile.budget, item.get("price"))
        history, history_available, history_matches = _history_score(profile, item, candidate_text)

        components = {
            "visual_similarity": (visual, visual_available),
            "nice_to_have": (nice, nice_available),
            "quality": (quality, True),
            "price": (price, price_available),
            "preference_history": (history, history_available),
        }
        total, breakdown = _weighted_score(components)
        reasons = _explain(
            item,
            breakdown,
            nice_matches=nice_matches,
            history_matches=history_matches,
            budget=profile.budget,
        )
        retrieval = _number(
            _nested_signal(item, "text_similarity", "text_score"),
            _nested_signal(item, "retrieval_score"),
        ) or 0.0
        rating = _float(item.get("rating"))
        price_value = _int(item.get("price"))
        scored.append((total, retrieval, rating, price_value, item, reasons, review, breakdown))

    scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3] or 10**9))
    selected = scored[: max(1, min(int(top_n or 1), 100))]
    ranked = [
        _to_ranked_product(item, total, reasons, review, breakdown)
        for total, _retrieval, _rating, _price, item, reasons, review, breakdown in selected
    ]

    summary = _summary(profile, ranked)
    if ranked and qwen_configured():
        summary = _qwen_summary(profile, ranked, summary)

    return RecommendationBundle(
        plan_id=plan_id,
        ranked=ranked,
        excluded=excluded,
        summary=summary,
        status="ready",
        ranking_policy={
            "agent": "recommend",
            "version": "2",
            "weights": RANKING_WEIGHTS,
            "expects": "merged candidates verified upstream; visual_similarity normalized to 0..1",
            "considered": len(candidates),
            "ranked": len(ranked),
            "excluded_upstream": len(excluded),
        },
    )


def _deduplicate(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per product after Text/Visual Retrieval candidate merge."""
    chosen: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        product_id = str(item.get("id") or "").strip()
        if not product_id:
            anonymous.append(item)
            continue
        previous = chosen.get(product_id)
        if previous is None or _retrieval_strength(item) > _retrieval_strength(previous):
            chosen[product_id] = item
    return list(chosen.values()) + anonymous


def _retrieval_strength(item: dict[str, Any]) -> float:
    return max(
        _normalized_signal(_nested_signal(item, "text_similarity", "text_score"), missing=0.0),
        _normalized_signal(_nested_signal(item, "visual_similarity", "visual_score"), missing=0.0),
        _normalized_signal(_nested_signal(item, "retrieval_score"), missing=0.0),
    )


def _upstream_rejection(item: dict[str, Any]) -> str | None:
    reason = str(item.get("rejection_reason") or "").strip()
    if reason:
        return reason
    if item.get("verified") is False:
        return "Rejected by upstream verification"
    verification = item.get("verification")
    if isinstance(verification, dict):
        status = str(verification.get("status") or "").lower()
        if status in {"rejected", "reject", "failed", "invalid"}:
            return str(verification.get("reason") or "Rejected by upstream verification")
    elif str(verification or "").lower() in {"rejected", "reject", "failed", "invalid"}:
        return "Rejected by upstream verification"
    return None


def _candidate_text(item: dict[str, Any]) -> str:
    fields: list[Any] = [
        item.get("name"), item.get("summary"), item.get("display"),
        item.get("performance"), item.get("battery"), item.get("platform"),
        item.get("enriched_text"), item.get("visual_attrs"),
        item.get("review_aspects"), item.get("review_sentiment"),
    ]
    return " ".join(_flatten_text(value) for value in fields if value not in (None, "")).lower()


def _flatten_text(value: Any) -> str:
    parsed = _json_object(value)
    if isinstance(parsed, dict):
        return " ".join(_flatten_text(v) for v in parsed.values())
    if isinstance(parsed, list):
        return " ".join(_flatten_text(v) for v in parsed)
    return str(parsed or "")


def _visual_score(
    profile: PreferenceProfile,
    item: dict[str, Any],
    candidate_text: str,
) -> tuple[float, bool]:
    explicit = _nested_signal(item, "visual_similarity", "visual_score")
    if explicit is not None:
        return _normalized_signal(explicit), True
    # Textual visual attributes are an explainable fallback, not vector retrieval.
    if profile.visual_context.strip():
        score, available, _ = _preference_score([profile.visual_context], candidate_text)
        return score, available
    return 0.0, False


def _preference_score(
    preferences: Iterable[str],
    candidate_text: str,
) -> tuple[float, bool, list[str]]:
    requested = [str(value).strip() for value in preferences if str(value).strip()]
    if not requested:
        return 0.0, False, []
    haystack = _tokens(candidate_text)
    all_tokens: set[str] = set()
    matched_tokens: set[str] = set()
    matched_labels: list[str] = []
    for preference in requested:
        tokens = _tokens(preference)
        if not tokens:
            continue
        all_tokens.update(tokens)
        overlap = tokens & haystack
        matched_tokens.update(overlap)
        if len(overlap) / len(tokens) >= 0.5:
            matched_labels.append(preference)
    if not all_tokens:
        return 0.0, False, []
    return len(matched_tokens) / len(all_tokens), True, matched_labels[:4]


def _quality_score(item: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    rating = max(0.0, min(5.0, _float(item.get("rating"))))
    rating_count = max(0, _int(item.get("rating_number")))
    raw_review = _json_object(item.get("review_aspects"))
    review = dict(raw_review) if isinstance(raw_review, dict) else {}
    review_count = max(rating_count, _int(item.get("review_count_used")))

    # Bayesian smoothing prevents a five-star product with one review from
    # automatically outranking a well-established product.
    prior_rating = 4.0
    prior_strength = 20
    adjusted_rating = (
        (rating * review_count + prior_rating * prior_strength)
        / (review_count + prior_strength)
        if rating > 0
        else prior_rating
    )
    rating_component = adjusted_rating / 5.0 if rating > 0 else 0.5

    pros = _string_list(review.get("pros"))
    cons = _string_list(review.get("cons"))
    issues = _string_list(review.get("issues"))
    aspects = review.get("aspects") if isinstance(review.get("aspects"), dict) else {}
    positive = sum(1 for value in aspects.values() if "positive" in str(value).lower())
    negative = sum(1 for value in aspects.values() if "negative" in str(value).lower())
    review_positive = len(pros) + positive
    review_negative = len(cons) + len(issues) + negative
    review_balance = (review_positive + 1) / (review_positive + review_negative + 2)
    evidence = min(1.0, math.log1p(max(review_count, _int(item.get("review_count_used")))) / math.log(101))
    balanced_with_confidence = 0.5 + (review_balance - 0.5) * evidence
    quality = 0.80 * rating_component + 0.20 * balanced_with_confidence
    review["_pros"] = pros
    review["_cons"] = cons
    review["_issues"] = issues
    review["_summary"] = str(review.get("summary") or item.get("review_sentiment") or "").strip()
    return _clamp(quality), review


def _price_score(budget: int | None, raw_price: Any) -> tuple[float, bool]:
    if not budget or budget <= 0:
        return 0.0, False
    price = _int(raw_price)
    if price <= 0:
        return 0.20, True
    comparable_budget = _catalog_budget(budget)
    ratio = price / comparable_budget
    if ratio <= 1.0:
        return 1.0, True
    # Price is a ranking preference, not a must-have verifier.  Over-budget
    # products decay smoothly and remain available if Verify accepted them.
    return _clamp(1.0 - (ratio - 1.0) / 0.5), True


def _history_score(
    profile: PreferenceProfile,
    item: dict[str, Any],
    candidate_text: str,
) -> tuple[float, bool, list[str]]:
    explicit = _nested_signal(item, "history_affinity", "preference_history_score")
    if explicit is not None:
        return _normalized_signal(explicit), True, []
    return _preference_score(profile.preference_history, candidate_text)


def _weighted_score(
    components: dict[str, tuple[float, bool]],
) -> tuple[float, dict[str, float]]:
    active_weight = sum(
        RANKING_WEIGHTS[name]
        for name, (_value, available) in components.items()
        if available
    )
    if active_weight <= 0:
        return 0.0, {}
    total = 0.0
    breakdown: dict[str, float] = {}
    for name, (value, available) in components.items():
        if not available:
            continue
        normalized = _clamp(value)
        total += normalized * RANKING_WEIGHTS[name] / active_weight
        breakdown[name] = round(normalized * 100.0, 1)
    return _clamp(total), breakdown


def _explain(
    item: dict[str, Any],
    breakdown: dict[str, float],
    *,
    nice_matches: list[str],
    history_matches: list[str],
    budget: int | None,
) -> list[str]:
    reasons: list[str] = []
    if breakdown.get("visual_similarity", 0) >= 50:
        reasons.append(f"Visual match {breakdown['visual_similarity']:.0f}%")
    if nice_matches:
        reasons.append("Nice-to-have match: " + ", ".join(nice_matches[:2]))
    rating = _float(item.get("rating"))
    rating_count = _int(item.get("rating_number"))
    if rating > 0:
        suffix = f" from {rating_count:,} ratings" if rating_count else ""
        reasons.append(f"Quality {rating:.1f}/5{suffix}")
    price = _int(item.get("price"))
    if budget and price > 0 and price <= _catalog_budget(budget):
        reasons.append("Within budget")
    if history_matches:
        reasons.append("Matches earlier preference: " + history_matches[0])
    return reasons[:5] or ["Strongest available recommendation score"]


def _to_ranked_product(
    item: dict[str, Any],
    total: float,
    reasons: list[str],
    review: dict[str, Any],
    breakdown: dict[str, float],
) -> RankedProduct:
    return RankedProduct(
        id=str(item.get("id", "")),
        name=str(item.get("name") or "Product"),
        price=_int(item.get("price")),
        score=int(round(total * 100.0)),
        reasons=reasons,
        summary=str(item.get("summary") or review.get("_summary") or ""),
        rating=_float(item.get("rating")),
        rating_number=_int(item.get("rating_number")),
        platform=str(item.get("platform") or ""),
        display=str(item.get("display") or ""),
        performance=str(item.get("performance") or ""),
        battery=str(item.get("battery") or ""),
        weight_kg=_float(item.get("weight_kg")),
        image_url=str(item.get("image_url") or ""),
        score_breakdown=breakdown,
        review_summary=str(review.get("_summary") or ""),
        review_pros=review.get("_pros", [])[:5],
        review_cons=review.get("_cons", [])[:5],
        review_issues=review.get("_issues", [])[:5],
    )


def _summary(profile: PreferenceProfile, ranked: list[RankedProduct]) -> str:
    if not ranked:
        return "No verified candidates are ready to recommend yet."
    top = ranked[0]
    bits = [f"Top pick is {top.name} with a recommendation score of {top.score}."]
    if profile.budget:
        bits.append(f"It was ranked against your {profile.budget} budget and stated preferences.")
    return " ".join(bits)


def _qwen_summary(
    profile: PreferenceProfile,
    ranked: list[RankedProduct],
    fallback: str,
) -> str:
    facts = [
        {
            "name": item.name,
            "price": item.price,
            "score": item.score,
            "score_breakdown": item.score_breakdown,
            "reasons": item.reasons,
        }
        for item in ranked[:3]
    ]
    try:
        return chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Recommend Agent for VoiceShop++. Explain the supplied "
                        "deterministic ranking in 1-3 short sentences. Use only the supplied "
                        "products and facts; do not change the order or invent evidence. Match "
                        "the user's language."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"preference={profile.to_dict()}\nranked={facts}\n"
                        f"deterministic_summary={fallback}"
                    ),
                },
            ],
            temperature=0.2,
        )
    except Exception:
        return fallback


def _nested_signal(item: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    retrieval = item.get("retrieval") or item.get("_retrieval")
    if isinstance(retrieval, dict):
        for name in names:
            if retrieval.get(name) is not None:
                return retrieval[name]
    return None


def _normalized_signal(value: Any, *, missing: float = 0.0) -> float:
    if value is None or value == "":
        return missing
    number = _float(value)
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    elif number < 0.0 and number >= -1.0:
        number = (number + 1.0) / 2.0
    return _clamp(number)


def _catalog_budget(budget: int) -> int:
    # Existing catalog prices are USD-like while the Android demo accepts CNY.
    return max(800, int(budget / 7)) if budget >= 3000 else budget


def _tokens(value: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value.lower())
    return {token for token in raw if len(token) >= 2 and token not in _STOPWORDS}


def _json_object(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _number(*values: Any) -> float | None:
    for value in values:
        if value not in (None, ""):
            return _float(value)
    return None


def _float(value: Any) -> float:
    try:
        number = float(value or 0)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
