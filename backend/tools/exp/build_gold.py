"""从已富集的 catalog.db 自动挖掘"金标准"(gold sets),供 Reviewer / Enrichment
文本实验使用。核心思想:利用库里已落地的 review_aspects / visual_attrs,自动找出
"某属性只在评论里出现"或"某视觉属性只在富集里出现(标题没有)"的商品集合,从而
无需人工标注即可构造可验证的相关集。

用法(打印/落盘挖到的金标准,便于检查):
    python tools/exp/build_gold.py
"""
from __future__ import annotations

import os

from common import (BACKEND, DB_PATH, load_catalog, product_category, tokenize,
                    ensure_results_dir, dump_json)

import sys
sys.path.insert(0, BACKEND)
from engine.reviewer import _looks_adjective  # noqa: E402
from engine.verifier import _GENERIC as _VERIFIER_GENERIC  # noqa: E402
from server import _STOPWORDS  # noqa: E402

# 校验器会把这些"通用词"忽略(不作为有效 must-have/品类信号),因此它们无法被
# 规则校验筛选,不适合作为 Reviewer 实验的属性 —— 从挖掘结果中排除。
_EXCLUDE_ATTR = set(_VERIFIER_GENERIC)


# --------------------------------------------------------------------------- #
# 文本侧:商品自身可命中的 token(name / enriched_text / summary)
# --------------------------------------------------------------------------- #
def _self_text_tokens(item: dict, include_enriched: bool = True) -> set[str]:
    parts = [item.get("name") or "", item.get("summary") or ""]
    if include_enriched:
        parts.append(item.get("enriched_text") or "")
    return tokenize(" ".join(parts))


def _name_tokens(item: dict) -> set[str]:
    return tokenize(item.get("name") or "")


# --------------------------------------------------------------------------- #
# 评论侧:买家反复提到的属性 token(keywords + pros + aspects 键)
# --------------------------------------------------------------------------- #
def _review_tokens(item: dict) -> set[str]:
    rv = item.get("_review") or {}
    bag: list[str] = []
    for k in ("keywords", "pros"):
        v = rv.get(k) or []
        if isinstance(v, list):
            bag.extend(str(x) for x in v)
    asp = rv.get("aspects") or {}
    if isinstance(asp, dict):
        bag.extend(str(k) for k in asp.keys())
    return tokenize(" ".join(bag))


# --------------------------------------------------------------------------- #
# Reviewer 金标准:属性 → {gold(评论有、标题/富集没有), distractor(都没有)}
# --------------------------------------------------------------------------- #
def mine_review_attributes(catalog: list[dict], *, min_gold: int = 5,
                           max_attrs: int = 12,
                           whitelist: list[str] | None = None) -> dict:
    """返回 {attr: {"gold": [ids], "distractor": [ids]}}。
    gold = 该形容词属性出现在评论里、但 name/enriched_text 里没有的商品(即"只有
    评论知道"的软属性);distractor = 该属性在评论/标题/富集里都没有的同池商品。"""
    # 预计算每个商品的三类 token
    for it in catalog:
        it["_rvtok"] = _review_tokens(it)
        it["_selftok"] = _self_text_tokens(it, include_enriched=True)

    # 候选属性:所有评论 token 里"看起来是形容词"的,统计其 review-only 计数
    from collections import Counter
    review_only_count: Counter = Counter()
    for it in catalog:
        review_only = it["_rvtok"] - it["_selftok"]
        for tok in review_only:
            if (len(tok) >= 4 and tok.isascii() and tok not in _EXCLUDE_ATTR
                    and _looks_adjective(tok)):
                review_only_count[tok] += 1

    if whitelist:
        attrs = [a for a in whitelist if review_only_count.get(a, 0) >= 1]
    else:
        attrs = [a for a, c in review_only_count.most_common() if c >= min_gold][:max_attrs]

    out: dict = {}
    for attr in attrs:
        gold, distractor = [], []
        for it in catalog:
            in_review = attr in it["_rvtok"]
            in_self = attr in it["_selftok"]
            if in_review and not in_self:
                gold.append(it["id"])
            elif (not in_review) and (not in_self):
                distractor.append(it["id"])
        if len(gold) >= min_gold:
            out[attr] = {"gold": gold, "distractor": distractor}
    return out


# --------------------------------------------------------------------------- #
# Enrichment 文本金标准:视觉属性(材质/款式/图案) → 标题没有但 enriched 有的商品
# --------------------------------------------------------------------------- #
def _visual_tokens(item: dict) -> set[str]:
    vis = (item.get("_visual") or {}).get("visual") or {}
    bag: list[str] = []
    for k in ("colors", "style", "keywords"):
        v = vis.get(k) or []
        if isinstance(v, list):
            bag.extend(str(x) for x in v)
    for k in ("material_look", "pattern", "shape", "product_category"):
        v = vis.get(k)
        if v:
            bag.append(str(v))
    return tokenize(" ".join(bag))


def mine_visual_attributes(catalog: list[dict], *, min_gold: int = 5,
                           max_attrs: int = 12,
                           whitelist: list[str] | None = None) -> dict:
    """返回 {attr: {"gold": [ids]}}。attr = 出现在 visual_attrs、但**不在商品标题**里
    的材质/款式/图案词(过滤掉检索停用词与裸颜色,因为 server.search 会丢弃它们)。
    gold = 标题没有该词、但 enriched(视觉属性)有该词的商品 → 只有富集能命中。"""
    for it in catalog:
        it["_vistok"] = _visual_tokens(it)
        it["_nametok"] = _name_tokens(it)

    from collections import Counter
    enrich_only_count: Counter = Counter()
    for it in catalog:
        enrich_only = it["_vistok"] - it["_nametok"]
        for tok in enrich_only:
            # 必须是 server.search 不会丢弃的词(裸颜色/generic 会被停用词过滤)
            if len(tok) >= 4 and tok.isascii() and tok not in _STOPWORDS:
                enrich_only_count[tok] += 1

    if whitelist:
        attrs = [a for a in whitelist if enrich_only_count.get(a, 0) >= 1]
    else:
        attrs = [a for a, c in enrich_only_count.most_common() if c >= min_gold][:max_attrs]

    out: dict = {}
    for attr in attrs:
        gold = [it["id"] for it in catalog
                if attr in it["_vistok"] and attr not in it["_nametok"]]
        if len(gold) >= min_gold:
            out[attr] = {"gold": gold}
    return out


# --------------------------------------------------------------------------- #
# 图片实验:按视觉品类分组(供"同类=相关"的相关性判定)
# --------------------------------------------------------------------------- #
def group_by_category(catalog: list[dict], *, min_size: int = 4) -> dict:
    """{category: [ids]}。只保留成员数 >= min_size 的品类,便于做同类相关性评测。"""
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for it in catalog:
        cat = product_category(it)
        if cat and (it.get("image_embedding") or "").strip():
            groups[cat].append(it["id"])
    return {c: ids for c, ids in groups.items() if len(ids) >= min_size}


def _main() -> None:
    catalog = load_catalog(DB_PATH)
    print(f"catalog: {len(catalog)} products")
    rev = mine_review_attributes(catalog)
    vis = mine_visual_attributes(catalog)
    cats = group_by_category(catalog)
    print("\n[Reviewer 软属性金标准] attr -> #gold / #distractor")
    for a, d in rev.items():
        print(f"  {a:<16} gold={len(d['gold']):>4}  distractor={len(d['distractor']):>4}")
    print("\n[Enrichment 视觉属性金标准] attr -> #gold")
    for a, d in vis.items():
        print(f"  {a:<16} gold={len(d['gold']):>4}")
    print(f"\n[图片实验] 可用品类组: {len(cats)}  (示例)")
    for c, ids in list(cats.items())[:8]:
        print(f"  {c:<24} size={len(ids)}")

    out = ensure_results_dir()
    dump_json(os.path.join(out, "gold_review.json"), rev)
    dump_json(os.path.join(out, "gold_visual.json"), vis)
    dump_json(os.path.join(out, "gold_categories.json"), cats)
    print(f"\n已落盘到 {out}/gold_*.json")


if __name__ == "__main__":
    _main()
