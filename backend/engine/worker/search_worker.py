from __future__ import annotations

from typing import Any, Callable

from ..models import PreferenceProfile


SearchFn = Callable[[dict], list[dict]]


def run_search(
    profile: PreferenceProfile,
    search_fn: SearchFn,
    limit: int = 80,
) -> list[dict[str, Any]]:
    # 文本检索只由**用户输入**驱动:LLM 从用户文字(+图片)抽出的英文目录关键词
    # search_keywords。不再回退到 planner 的 query_hint,也不补任何规则默认值——
    # 用户没给可检索的文字信息,查询就为空,关键词召回直接返回空(交给图片召回)。
    q = " ".join(profile.search_keywords).strip()
    if not q:
        return []
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
