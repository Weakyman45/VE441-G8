"""Deterministic text retrieval over the enriched product catalog."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Iterable

from ..models import PreferenceProfile


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "be", "buy", "for", "from", "good", "have",
    "i", "in", "is", "it", "me", "my", "need", "of", "on", "or", "please",
    "budget", "dollar", "dollars", "product", "rmb", "some", "that", "the",
    "this", "to", "under", "up", "usd", "want", "with",
}


def retrieve_text(
    profile: PreferenceProfile,
    catalog: Iterable[dict[str, Any]],
    *,
    query_hint: str | None = None,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Return catalog candidates ranked by weighted lexical relevance.

    This worker retrieves only; it deliberately does not enforce must-haves,
    reject products, or perform final recommendation scoring.
    """
    products = list(catalog)
    if not products or limit <= 0:
        return []

    query_parts = _query_parts(profile, query_hint)
    query_tokens = _tokens(" ".join(query_parts))
    if not query_tokens:
        # With no searchable text, keep the pipeline useful by passing popular
        # catalog items to the Recommend Agent with an honest zero similarity.
        fallback = sorted(products, key=_popularity_key, reverse=True)[:limit]
        return [_with_text_score(item, 0.0, query_tokens) for item in fallback]

    documents = [_document_fields(item) for item in products]
    doc_frequency: Counter[str] = Counter()
    for fields in documents:
        seen = set().union(*(tokens for _weight, tokens, _text in fields))
        doc_frequency.update(seen)

    unique_query = list(dict.fromkeys(query_tokens))
    idf = {
        term: math.log((len(products) + 1) / (doc_frequency.get(term, 0) + 1)) + 1.0
        for term in unique_query
    }
    scored: list[tuple[float, int, dict[str, Any]]] = []
    phrases = [part.lower().strip() for part in query_parts if len(_tokens(part)) >= 2]
    for item, fields in zip(products, documents):
        score = _weighted_coverage(unique_query, idf, fields)
        if score > 0:
            searchable = " ".join(text for _weight, _tokens_set, text in fields)
            phrase_match = any(phrase in searchable for phrase in phrases)
            matched_count = sum(
                1
                for token in unique_query
                if any(token in field_tokens for _weight, field_tokens, _text in fields)
            )
            score = min(1.0, score * 0.90 + (0.10 if phrase_match else 0.0))
            scored.append((score, matched_count, item))

    # Once the catalog contains real multi-token matches, one-token hits are
    # usually category drift (for example a coffee "charcoal filter" for a
    # "charcoal cleanser" request). Keep single-token results only when they
    # are the best evidence the catalog can provide.
    if any(matched_count >= 2 for _score, matched_count, _item in scored):
        scored = [entry for entry in scored if entry[1] >= 2]

    scored.sort(
        key=lambda entry: (entry[0], entry[1], *_popularity_key(entry[2])),
        reverse=True,
    )
    return [
        _with_text_score(item, score, unique_query)
        for score, _matched_count, item in scored[:limit]
    ]


def _query_parts(profile: PreferenceProfile, query_hint: str | None) -> list[str]:
    parts: list[str] = []
    parts.extend(str(value).strip() for value in profile.search_keywords if str(value).strip())
    for value in (query_hint, profile.category, profile.use_case):
        if value and str(value).strip():
            parts.append(str(value).strip())
    parts.extend(str(value).strip() for value in profile.hard + profile.soft if str(value).strip())
    if profile.raw_query.strip():
        parts.append(profile.raw_query.strip())
    return list(dict.fromkeys(parts))


def _document_fields(item: dict[str, Any]) -> list[tuple[float, set[str], str]]:
    fields = [
        (1.00, _flatten(item.get("name"))),
        (0.85, _flatten(item.get("enriched_text"))),
        (0.80, _flatten(item.get("visual_attrs"))),
        (
            0.55,
            " ".join(
                _flatten(item.get(key))
                for key in (
                    "summary", "display", "performance", "battery", "platform",
                    "review_sentiment", "review_aspects", "reasons", "trade_offs",
                    "weakness", "store",
                )
            ),
        ),
    ]
    return [(weight, set(_tokens(text)), text.lower()) for weight, text in fields]


def _weighted_coverage(
    query_tokens: list[str],
    idf: dict[str, float],
    fields: list[tuple[float, set[str], str]],
) -> float:
    denominator = sum(idf[token] for token in query_tokens)
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    for token in query_tokens:
        best = max(
            (weight for weight, field_tokens, _text in fields if token in field_tokens),
            default=0.0,
        )
        numerator += best * idf[token]
    return max(0.0, min(1.0, numerator / denominator))


def _with_text_score(
    item: dict[str, Any],
    score: float,
    query_tokens: list[str],
) -> dict[str, Any]:
    candidate = _public_candidate(item)
    candidate["text_similarity"] = round(max(0.0, min(1.0, score)), 6)
    searchable_tokens = set().union(
        *(tokens for _weight, tokens, _text in _document_fields(item))
    )
    candidate["retrieval"] = {
        "sources": ["text"],
        "text_similarity": candidate["text_similarity"],
        "matched_query_tokens": [
            token for token in query_tokens
            if token in searchable_tokens
        ][:12],
    }
    return candidate


def _tokens(value: str) -> list[str]:
    return [
        token for token in _TOKEN_RE.findall((value or "").lower())
        if token not in _STOPWORDS and not token.isdigit()
    ]


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return _flatten(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                pass
        return text
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _public_candidate(item: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(item)
    candidate.pop("image_embedding", None)
    candidate.pop("review_embedding", None)
    return candidate


def _popularity_key(item: dict[str, Any]) -> tuple[float, int]:
    try:
        rating = float(item.get("rating") or 0)
    except (TypeError, ValueError):
        rating = 0.0
    try:
        count = int(item.get("rating_number") or 0)
    except (TypeError, ValueError):
        count = 0
    return rating, count
