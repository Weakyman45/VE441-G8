from __future__ import annotations

from typing import Any

from ..conflicts import (
    ConflictPacket,
    build_talker_brief,
    build_tradeoffs,
)
from ..llm.qwen_client import chat_completion, qwen_configured
from ..models import PreferenceProfile, RankedProduct, RecommendationBundle
from ..query_fusion import normalize_fuse_weights


def rank_products(
    plan_id: str,
    profile: PreferenceProfile,
    candidates: list[dict[str, Any]],
    top_n: int = 40,
    *,
    fuse_weights: dict[str, float] | None = None,
    modality: str = "text",
    conflicts: list[ConflictPacket] | list[dict[str, Any]] | None = None,
    unresolved: list[ConflictPacket] | list[dict[str, Any]] | None = None,
) -> RecommendationBundle:
    weights = normalize_fuse_weights(fuse_weights, modality=modality)
    query_tokens = _keyword_tokens(profile.search_keywords)
    scored: list[tuple[float, int, dict[str, Any], list[str]]] = []
    excluded: list[dict[str, str]] = []
    soft_conflicts: list[ConflictPacket] = []

    for item in candidates:
        score, reasons, exclude_reason = _score(profile, item, weights)
        if exclude_reason:
            excluded.append({"id": str(item.get("id", "")), "reason": exclude_reason})
            continue
        name = str(item.get("name") or "").lower()
        text_hits = sum(1 for t in query_tokens if t in name)
        # Primary sort key blends keyword relevance with fused text/visual score.
        fused_rank = (
            weights["text"] * float(text_hits)
            + weights["visual"] * float(item.get("_visual_score") or 0.0) * 5.0
        )
        scored.append((fused_rank, score, item, reasons))

        # Soft trade-off: expensive but strong visual match.
        price = int(item.get("price") or 0)
        cmp_b = _cmp_budget(profile.budget)
        vis = float(item.get("_visual_score") or 0.0)
        if cmp_b and price > cmp_b and vis >= 0.55:
            soft_conflicts.append(
                ConflictPacket(
                    product_id=str(item.get("id") or ""),
                    product_name=str(item.get("name") or ""),
                    conflict_type="soft_tradeoff",
                    constraint=f"strong visual match but above budget ({price} > ~{cmp_b})",
                    status="soft_tradeoff",
                    evidence=[
                        {"source": "visual", "snippet": f"similarity={vis:.2f}", "confidence": vis},
                        {"source": "price", "snippet": str(price), "confidence": 0.9},
                    ],
                    user_action="accept_tradeoff",
                )
            )

    scored.sort(key=lambda x: (-x[0], -x[1], x[2].get("price") or 10**9))
    relevant = [t for t in scored if t[0] > 0 or t[1] >= 70]
    selected = (relevant if relevant else scored[:6])[:top_n]
    ranked = [
        RankedProduct(
            id=str(item.get("id", "")),
            name=str(item.get("name") or "Product"),
            price=int(item.get("price") or 0),
            score=score,
            reasons=reasons[:4],
            summary=str(item.get("summary") or ""),
            rating=float(item.get("rating") or 0),
            platform=str(item.get("platform") or "Windows"),
            display=str(item.get("display") or ""),
            performance=str(item.get("performance") or ""),
            weight_kg=float(item.get("weight_kg") or 0),
            image_url=str(item.get("image_url") or ""),
        )
        for fused_rank, score, item, reasons in selected
    ]
    summary = _summary(profile, ranked)
    if ranked and qwen_configured():
        summary = _qwen_summary(profile, ranked, summary)

    packets = _as_packets(conflicts) + _as_packets(unresolved) + soft_conflicts[:3]
    tradeoffs = build_tradeoffs(ranked)
    open_questions = [
        f"I could not verify '{c.constraint}' for {c.product_name or 'a candidate'}. "
        "Should that stay a hard requirement?"
        for c in _as_packets(unresolved)[:2]
    ]
    talker_brief = build_talker_brief(
        summary=summary,
        ranked=ranked,
        conflicts=packets,
        tradeoffs=tradeoffs,
        open_questions=open_questions,
    )

    return RecommendationBundle(
        plan_id=plan_id,
        ranked=ranked,
        excluded=excluded[:8],
        summary=summary,
        status="ready",
        conflicts=[c.to_dict() for c in packets[:12]],
        tradeoffs=tradeoffs,
        open_questions=open_questions,
        talker_brief=talker_brief,
    )


def _as_packets(
    items: list[ConflictPacket] | list[dict[str, Any]] | None,
) -> list[ConflictPacket]:
    out: list[ConflictPacket] = []
    for item in items or []:
        if isinstance(item, ConflictPacket):
            out.append(item)
        elif isinstance(item, dict):
            out.append(ConflictPacket.from_dict(item))
    return out


def _cmp_budget(budget: int | None) -> int | None:
    if not budget:
        return None
    return budget if budget < 3000 else max(800, int(budget / 7))


def _qwen_summary(
    profile: PreferenceProfile,
    ranked: list[RankedProduct],
    fallback: str,
) -> str:
    facts = [
        {
            "name": r.name,
            "price": r.price,
            "score": r.score,
            "reasons": r.reasons,
        }
        for r in ranked[:3]
    ]
    try:
        return chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Recommend Worker for VoiceShop++, a general-purpose "
                        "shopping assistant (any product category). "
                        "Write 1–3 short spoken sentences summarizing ONLY these ranked products. "
                        "Match the user's language (Chinese if they spoke Chinese). "
                        "Do not invent products or prices."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"preference={profile.to_dict()}\n"
                        f"ranked={facts}\n"
                        f"rule_summary={fallback}"
                    ),
                },
            ],
            temperature=0.4,
        )
    except Exception:
        return fallback


def _score(
    profile: PreferenceProfile,
    item: dict[str, Any],
    weights: dict[str, float],
) -> tuple[int, list[str], str | None]:
    reasons: list[str] = []
    price = int(item.get("price") or 0)
    rating = float(item.get("rating") or 0)
    weight = float(item.get("weight_kg") or 0)
    display = str(item.get("display") or "")
    platform = str(item.get("platform") or "Windows")
    score = int(max(60, min(99, rating / 5.0 * 100))) if rating else 70

    if profile.budget and price > 0:
        budget = profile.budget
        cmp_budget = budget if budget < 3000 else max(800, int(budget / 7))
        if price <= cmp_budget:
            score += 4
            reasons.append(f"Within budget band (~{cmp_budget})")
        elif price > cmp_budget * 1.35:
            return score, reasons, f"Over budget ({price} > {cmp_budget})"
        else:
            score -= 3
            reasons.append("Slightly above budget")

    if profile.platform in ("Windows", "macOS") and platform != profile.platform:
        return score, reasons, f"Platform mismatch ({platform})"

    soft = " ".join(
        profile.soft + profile.hard + [profile.use_case, profile.raw_query, profile.visual_context]
    ).lower()
    haystack = " ".join(
        str(item.get(k) or "") for k in ("name", "summary", "display", "performance", "platform", "enriched_text")
    ).lower()
    name = str(item.get("name") or "").lower()
    query_tokens = _keyword_tokens(profile.search_keywords)
    matched = sorted({t for t in query_tokens if t in name})
    text_bonus = 0
    if matched:
        text_bonus = min(30, 8 * len(matched))
        reasons.append("Matches: " + ", ".join(matched)[:60])

    visual = float(item.get("_visual_score") or 0.0)
    visual_bonus = 0
    if visual > 0:
        visual_bonus = int(round(visual * 20))
        if visual >= 0.45:
            reasons.append(f"Visual similarity {visual:.2f}")

    # Blend text/visual contribution into the heuristic score using fuse weights.
    score += int(round(weights["text"] * text_bonus + weights["visual"] * visual_bonus))

    for token in _important_tokens(profile.visual_context):
        if token in haystack:
            score += 3
            reasons.append(f"Matches image cue: {token}")
    if "oled" in soft or "display" in soft or "design" in soft or "color" in soft:
        if "OLED" in display.upper():
            score += 3
            reasons.append("Strong display match")
    if "portable" in soft or "lightweight" in soft or "campus" in soft:
        if 0 < weight <= 1.4:
            score += 2
            reasons.append("Portable weight")
        elif weight > 1.8:
            score -= 2
    if "gaming" in soft or "video" in soft or "creative" in soft:
        score += 1
        reasons.append("Fits stated use case")
    if rating >= 4.5:
        reasons.append(f"Well reviewed ({rating})")
    if not reasons:
        reasons.append("Matches your search")
    return max(60, min(99, score)), reasons, None


_GENERIC_WORDS = {
    "white", "black", "red", "blue", "green", "yellow", "gray", "grey", "pink",
    "purple", "orange", "brown", "silver", "gold", "beige", "navy", "neutral",
    "color", "colors", "colour", "colours", "colorway", "colorways", "multicolor",
    "breathable", "lightweight", "comfortable", "comfy", "casual", "waterproof",
    "durable", "premium", "quality", "soft", "warm", "cool", "versatile",
    "simple", "classic", "stylish", "fashion", "fashionable", "new", "best",
    "minimalist", "design", "designs", "style", "look", "everyday", "daily",
    "size", "sizes", "women", "womens", "woman", "men", "mens", "man",
    "unisex", "adult", "adults", "kids", "boys", "girls", "the", "for", "and",
    "with", "your",
}


def _keyword_tokens(keywords: list[str]) -> set[str]:
    out: set[str] = set()
    for phrase in keywords or []:
        for raw in str(phrase).lower().replace("-", " ").split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if len(token) >= 3 and token not in _GENERIC_WORDS:
                out.add(token)
    return out


def _important_tokens(text: str) -> list[str]:
    tokens = []
    for raw in text.lower().replace("-", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) < 4:
            continue
        if token in {
            "laptop", "under", "below", "with", "screen", "storage", "price",
            "advertisement", "highlighting", "specifications",
        }:
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens))[:8]


def _summary(profile: PreferenceProfile, ranked: list[RankedProduct]) -> str:
    if not ranked:
        return "I could not find a strong match yet. Tell me budget or use case to refine."
    top = ranked[0]
    bits = [f"Top pick is {top.name}"]
    if profile.budget:
        bits.append(f"for around your {profile.budget} budget")
    if profile.use_case:
        bits.append(f"aimed at {profile.use_case}")
    bits.append(f"with {len(ranked)} options ready to compare.")
    return " ".join(bits)
