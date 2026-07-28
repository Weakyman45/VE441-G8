"""实验 R:Reviewer 对"评论型软属性"校验的贡献(评论 ON vs OFF)。

要证明的命题:很多体验类软需求(durable/comfortable/soft/breathable…)只存在于
**用户评论**里,商品标题与富集文本都没有。开启 Reviewer(review_aspects 进入
校验器命中判据)后,校验器能据此满足这些 must-have;关闭后则无从判定而漏掉。

输入(input):对每个软属性 attr,构造一条用户查询
    profile = {raw_query: "... attr ...", hard(must-have)=[attr], category=""}
候选池 = gold(评论提到 attr、但标题/富集没有的商品) + distractor(都没提到的同池商品)。
(category 置空以隔离变量:只考察 must-have=attr 这一硬约束。)

被测系统:verifier.verify_candidates,配置只差一个开关 reviews=True/False
(verifier 模式默认 rule,确定、免 LLM、可复现;--verifier llm 可另测 LLM 判官)。

输出(output):
  - 每个属性一行:#gold / recall_off / recall_on / Δ / distractor 保留率(on/off)
  - 汇总均值 + 跨 gold 商品的配对 Wilcoxon 显著性
  - 一个定性例子(某属性的 gold 商品 + 其评论关键词 + on/off 是否保留)
  - 柱状图 recall_off vs recall_on
落盘:results/reviewer_*.csv / .md / .png / _example.json

用法:
    python tools/exp/exp_reviewer.py                 # 默认 rule 校验、自动挖属性
    python tools/exp/exp_reviewer.py --verifier llm  # 用 LLM 判官(需 DASHSCOPE_API_KEY)
    python tools/exp/exp_reviewer.py --attrs durable comfortable soft breathable
"""
from __future__ import annotations

import argparse
import os
import random

import common as C
from build_gold import mine_review_attributes

from engine.exp_config import ExpConfig
from engine.models import PreferenceProfile
from engine import verifier


# 论文默认聚焦这些"真·产品体验属性"(可 --attrs 覆盖)。留空则自动挖掘。
DEFAULT_WHITELIST = [
    "durable", "comfortable", "soft", "breathable", "lightweight", "sturdy",
    "reliable", "effective", "affordable", "waterproof", "adjustable", "gentle",
]


def _cfg(reviews: bool, verifier_mode: str) -> ExpConfig:
    return ExpConfig(
        modality_routing=True, visual_recall=True, enrichment=True,
        source_layering=True, verifier=verifier_mode, reviews=reviews,
        visual_top_k=40, review_top_k=20, embedding_provider="dashscope",
    )


def _run_pool(profile, pool: list[dict], cfg: ExpConfig) -> set[str]:
    """对候选池跑一次校验,返回被保留(kept)的 id 集合。每次用独立副本,避免
    verifier 往 item 里写 _verify 造成串扰。"""
    cands = [dict(it) for it in pool]
    res = verifier.verify_candidates(profile, cands, cfg)
    return {str(k.get("id")) for k in res.kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="rule", choices=["rule", "llm", "llm_strict"])
    ap.add_argument("--attrs", nargs="*", default=None,
                    help="指定属性;不给则用内置白名单;传 auto 则全自动挖掘")
    ap.add_argument("--min-gold", type=int, default=5)
    ap.add_argument("--distractor-ratio", type=int, default=3,
                    help="每个属性 distractor 数 = ratio * #gold(上限 200)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    catalog = C.load_catalog()
    by_id = {it["id"]: it for it in catalog}

    if args.attrs == ["auto"] or args.attrs == []:
        whitelist = None
    else:
        whitelist = args.attrs or DEFAULT_WHITELIST
    gold_map = mine_review_attributes(catalog, min_gold=args.min_gold, whitelist=whitelist)
    if not gold_map:
        print("没有挖到满足条件的属性,试试 --min-gold 3 或 --attrs auto")
        return

    cfg_on = _cfg(True, args.verifier)
    cfg_off = _cfg(False, args.verifier)

    rows: list[dict] = []
    paired_on: list[float] = []   # 每个 gold 商品:on 是否保留(1/0)
    paired_off: list[float] = []
    example = None

    for attr in gold_map:
        gold = [g for g in gold_map[attr]["gold"] if g in by_id]
        distractor_all = [d for d in gold_map[attr]["distractor"] if d in by_id]
        n_dist = min(len(distractor_all), args.distractor_ratio * len(gold), 200)
        distractor = random.sample(distractor_all, n_dist) if n_dist else []

        gold_items = [by_id[g] for g in gold]
        dist_items = [by_id[d] for d in distractor]
        pool = gold_items + dist_items

        profile = PreferenceProfile(raw_query=f"a {attr} product", hard=[attr],
                                    category="")
        kept_on = _run_pool(profile, pool, cfg_on)
        kept_off = _run_pool(profile, pool, cfg_off)

        gold_set = set(gold)
        dist_set = set(distractor)
        recall_on = C.mean([1.0 if g in kept_on else 0.0 for g in gold])
        recall_off = C.mean([1.0 if g in kept_off else 0.0 for g in gold])
        dkeep_on = C.mean([1.0 if d in kept_on else 0.0 for d in distractor]) if distractor else 0.0
        dkeep_off = C.mean([1.0 if d in kept_off else 0.0 for d in distractor]) if distractor else 0.0

        for g in gold:
            paired_on.append(1.0 if g in kept_on else 0.0)
            paired_off.append(1.0 if g in kept_off else 0.0)

        rows.append({
            "attribute": attr,
            "#gold": len(gold),
            "#distractor": len(distractor),
            "recall_off": recall_off,
            "recall_on": recall_on,
            "delta": recall_on - recall_off,
            "distractor_keep_off": dkeep_off,
            "distractor_keep_on": dkeep_on,
        })

        if example is None and len(gold) >= 3:
            example = {"attribute": attr, "samples": []}
            for g in gold[:5]:
                it = by_id[g]
                rv = it.get("_review") or {}
                example["samples"].append({
                    "id": g,
                    "name": (it.get("name") or "")[:70],
                    "review_keywords": (rv.get("keywords") or [])[:8],
                    "kept_when_reviews_ON": g in kept_on,
                    "kept_when_reviews_OFF": g in kept_off,
                })

    # 汇总行
    agg = {
        "attribute": "AVG",
        "#gold": sum(r["#gold"] for r in rows),
        "#distractor": sum(r["#distractor"] for r in rows),
        "recall_off": C.mean([r["recall_off"] for r in rows]),
        "recall_on": C.mean([r["recall_on"] for r in rows]),
        "delta": C.mean([r["delta"] for r in rows]),
        "distractor_keep_off": C.mean([r["distractor_keep_off"] for r in rows]),
        "distractor_keep_on": C.mean([r["distractor_keep_on"] for r in rows]),
    }
    p = C.wilcoxon_p(paired_on, paired_off)

    out = C.ensure_results_dir()
    headers = ["attribute", "#gold", "#distractor", "recall_off", "recall_on",
               "delta", "distractor_keep_off", "distractor_keep_on"]
    C.write_csv(os.path.join(out, "reviewer_recall.csv"), rows + [agg], headers)
    md = C.write_markdown_table(
        os.path.join(out, "reviewer_recall.md"), rows + [agg], headers,
        title=f"Reviewer 消融:评论型软属性 must-have 满足率(verifier={args.verifier})")
    md += (f"\n**跨 {len(paired_on)} 个 gold 商品的配对 Wilcoxon**:reviews ON vs OFF, "
           f"p = {p:.3g} ({C.sig_mark(p)})\n")
    with open(os.path.join(out, "reviewer_recall.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if example:
        C.dump_json(os.path.join(out, "reviewer_example.json"), example)

    C.grouped_bar(
        os.path.join(out, "reviewer_recall.png"),
        categories=[r["attribute"] for r in rows],
        series={"reviews OFF": [r["recall_off"] for r in rows],
                "reviews ON": [r["recall_on"] for r in rows]},
        ylabel="must-have satisfaction (recall of gold)",
        title="Reviewer ablation: must-have satisfaction for review-only attributes")

    print(md)
    print(f"\n结果已落盘到 {out}/reviewer_*")


if __name__ == "__main__":
    main()
