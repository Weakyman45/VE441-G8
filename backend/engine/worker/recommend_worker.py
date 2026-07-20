from __future__ import annotations

from typing import Any

from ..llm.qwen_client import chat_completion, qwen_configured
from ..models import PreferenceProfile, RankedProduct, RecommendationBundle


def rank_products(
    plan_id: str,
    profile: PreferenceProfile,
    candidates: list[dict[str, Any]],
    top_n: int = 3,
) -> RecommendationBundle:
    scored: list[tuple[int, dict[str, Any], list[str]]] = []
    excluded: list[dict[str, str]] = []
    for item in candidates:
        score, reasons, exclude_reason = _score(profile, item)
        if exclude_reason:
            excluded.append({"id": str(item.get("id", "")), "reason": exclude_reason})
            continue
        scored.append((score, item, reasons))
    scored.sort(key=lambda x: (-x[0], x[1].get("price") or 10**9))
    ranked = [
        RankedProduct(
            id=str(item.get("id", "")),
            name=str(item.get("name") or "Laptop"),
            price=int(item.get("price") or 0),
            score=score,
            reasons=reasons[:4],
            summary=str(item.get("summary") or ""),
            rating=float(item.get("rating") or 0),
            platform=str(item.get("platform") or "Windows"),
            display=str(item.get("display") or ""),
            performance=str(item.get("performance") or ""),
            weight_kg=float(item.get("weight_kg") or 0),
        )
        for score, item, reasons in scored[:top_n]
    ]
    summary = _summary(profile, ranked)
    if ranked and qwen_configured():
        summary = _qwen_summary(profile, ranked, summary)
    return RecommendationBundle(
        plan_id=plan_id,
        ranked=ranked,
        excluded=excluded[:8],
        summary=summary,
        status="ready",
    )


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
                        "You are the Recommend Worker for VoiceShop++. "
                        "Write 1–3 short spoken sentences summarizing ONLY these ranked laptops. "
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


def _score(profile: PreferenceProfile, item: dict[str, Any]) -> tuple[int, list[str], str | None]:
    reasons: list[str] = []
    price = int(item.get("price") or 0)
    rating = float(item.get("rating") or 0)
    weight = float(item.get("weight_kg") or 0)
    display = str(item.get("display") or "")
    platform = str(item.get("platform") or "Windows")
    score = int(max(60, min(99, rating / 5.0 * 100))) if rating else 70

    if profile.budget and price > 0:
        # Compare in catalog currency loosely
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
        str(item.get(k) or "") for k in ("name", "summary", "display", "performance", "platform")
    ).lower()
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
        reasons.append("Matches your laptop search")
    return max(60, min(99, score)), reasons, None


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
