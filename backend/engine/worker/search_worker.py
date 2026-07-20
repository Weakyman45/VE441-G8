from __future__ import annotations

from typing import Any, Callable

from ..intent import preference_search_query
from ..models import PreferenceProfile


SearchFn = Callable[[dict], list[dict]]


def run_search(
    profile: PreferenceProfile,
    search_fn: SearchFn,
    limit: int = 40,
    query_hint: str | None = None,
) -> list[dict[str, Any]]:
    q = (query_hint or "").strip() or preference_search_query(profile)
    params: dict[str, list] = {"q": [q], "limit": [str(limit)], "sort": ["popular"]}
    if profile.budget:
        # Catalog prices are USD-ish; App shows ¥ — keep numeric filter soft.
        # If budget looks like CNY (>=3000), also try USD-ish ceiling.
        max_price = profile.budget
        if max_price >= 3000:
            max_price = max(800, int(max_price / 7))  # rough ¥→$ for demo catalog
        params["max_price"] = [str(max_price)]
    results = search_fn(params)
    if profile.platform in ("Windows", "macOS"):
        filtered = [r for r in results if (r.get("platform") or "Windows") == profile.platform]
        if filtered:
            results = filtered
    return results
