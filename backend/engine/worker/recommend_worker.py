from __future__ import annotations

from typing import Any

from ..llm.qwen_client import chat_completion, qwen_configured
from ..models import PreferenceProfile, RankedProduct, RecommendationBundle


def rank_products(
    plan_id: str,
    profile: PreferenceProfile,
    candidates: list[dict[str, Any]],
    top_n: int = 40,
) -> RecommendationBundle:
    query_tokens = _keyword_tokens(profile.search_keywords)
    scored: list[tuple[int, int, dict[str, Any], list[str]]] = []
    excluded: list[dict[str, str]] = []
    for item in candidates:
        score, reasons, exclude_reason = _score(profile, item)
        if exclude_reason:
            excluded.append({"id": str(item.get("id", "")), "reason": exclude_reason})
            continue
        # Relevance = how many specific product words appear in the name. This is
        # the PRIMARY ranking signal: a genuine match must outrank a merely
        # high-rated item, otherwise ratings saturate the score cap and cheap
        # irrelevant products win the tiebreak.
        name = str(item.get("name") or "").lower()
        relevance = sum(1 for t in query_tokens if t in name)
        scored.append((relevance, score, item, reasons))
    # Sort by relevance first, then quality score, then price ascending.
    scored.sort(key=lambda x: (-x[0], -x[1], x[2].get("price") or 10**9))
    # Show ALL genuinely relevant products (name matched >=1 specific product
    # word), not just a fixed top-3. Only when nothing matched the keywords do
    # we fall back to a small set so the screen is not empty.
    relevant = [t for t in scored if t[0] >= 1]
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
        for relevance, score, item, reasons in selected
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
    # Reward products whose NAME contains the specific product WORDS from the
    # search keywords. Match individual tokens (not whole phrases) and drop
    # generic color/adjective words, so a real "running shoe" outranks a
    # "neutral color" hair clip. Weighted heavily so relevance beats raw rating.
    name = str(item.get("name") or "").lower()
    query_tokens = _keyword_tokens(profile.search_keywords)
    matched = sorted({t for t in query_tokens if t in name})
    if matched:
        score += min(30, 8 * len(matched))
        reasons.append("Matches: " + ", ".join(matched)[:60])
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


# Generic color / adjective words that match almost anything in an English
# catalog and therefore must NOT count as relevance signal on their own.
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
    """Split search-keyword phrases into specific product words (≥3 chars,
    excluding generic color/adjective words)."""
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
