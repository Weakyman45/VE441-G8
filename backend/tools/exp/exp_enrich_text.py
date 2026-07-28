"""实验 E-text:纯文本输入下,富集出的视觉属性(材质/款式/图案)能否让检索
命中"标题里没写、但图片里能看出"的商品。

要证明的命题:用户用纯文字描述一个视觉属性(如 leather / floral / ceramic /
metallic / velvet),很多相关商品的**标题并不含该词**,只有离线富集(Qwen-VL 抽的
visual_attrs 写进 enriched_text)才带这个词。开启富集检索 → 召回显著提升。

输入(input):文本查询 q = 某视觉属性词(材质/款式/图案)。
被测系统:真实的 server.search(),仅切换 VS_ENRICHMENT=1/0
  - ON :匹配 name + enriched_text(用富集)
  - OFF:只匹配 name(不用富集,即使库里 enriched_text 仍在)
金标准:gold(attr) = 该属性出现在 visual_attrs、但**不在商品标题**里的商品。

输出(output):
  - 每个属性一行:#gold / recall@K(off/on)/ Δ / enrich-only 命中数(仅富集才召回的相关商品)
  - 汇总均值 + 跨 gold 商品 hit@K 的配对 Wilcoxon
  - 柱状图 recall@K off vs on
落盘:results/enrich_text_*.csv / .md / .png

用法:
    python tools/exp/exp_enrich_text.py
    python tools/exp/exp_enrich_text.py --k 20 --attrs leather floral ceramic metallic velvet
"""
from __future__ import annotations

import argparse
import os

import common as C
from build_gold import mine_visual_attributes

import server


def _search_ids(attr: str, use_enrichment: bool, limit: int) -> list[str]:
    """用真实 server.search 检索,返回按序 id 列表。通过环境变量切换富集开关
    (server.search 内部每次调用都重新读 load_config())。"""
    os.environ["VS_ENRICHMENT"] = "1" if use_enrichment else "0"
    rows = server.search({"q": [attr], "limit": [str(limit)], "sort": ["popular"]})
    return [str(r.get("id")) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=80, help="检索返回条数(供 recall 计算)")
    ap.add_argument("--attrs", nargs="*", default=None,
                    help="指定视觉属性;不给则自动挖掘材质/款式/图案词")
    ap.add_argument("--min-gold", type=int, default=5)
    args = ap.parse_args()
    K = args.k

    catalog = C.load_catalog()
    gold_map = mine_visual_attributes(catalog, min_gold=args.min_gold,
                                      whitelist=args.attrs)
    if not gold_map:
        print("没挖到视觉属性,试试 --min-gold 3")
        return

    rows: list[dict] = []
    paired_on: list[float] = []   # 每个 gold 商品:on 是否在 top-K(1/0)
    paired_off: list[float] = []

    for attr in gold_map:
        gold = set(gold_map[attr]["gold"])
        ranked_on = _search_ids(attr, True, args.limit)
        ranked_off = _search_ids(attr, False, args.limit)

        recall_on = C.recall_at_k(ranked_on, gold, K)
        recall_off = C.recall_at_k(ranked_off, gold, K)
        ndcg_on = C.ndcg_at_k(ranked_on, gold, K)
        ndcg_off = C.ndcg_at_k(ranked_off, gold, K)

        topk_on = set(ranked_on[:K])
        topk_off = set(ranked_off[:K])
        enrich_only = len((gold & topk_on) - topk_off)  # 仅富集才召回的相关商品数

        for g in gold:
            paired_on.append(1.0 if g in topk_on else 0.0)
            paired_off.append(1.0 if g in topk_off else 0.0)

        rows.append({
            "attribute": attr,
            "#gold": len(gold),
            f"recall@{K}_off": recall_off,
            f"recall@{K}_on": recall_on,
            "delta": recall_on - recall_off,
            f"ndcg@{K}_off": ndcg_off,
            f"ndcg@{K}_on": ndcg_on,
            "enrich_only_hits": enrich_only,
        })

    agg = {
        "attribute": "AVG",
        "#gold": sum(r["#gold"] for r in rows),
        f"recall@{K}_off": C.mean([r[f"recall@{K}_off"] for r in rows]),
        f"recall@{K}_on": C.mean([r[f"recall@{K}_on"] for r in rows]),
        "delta": C.mean([r["delta"] for r in rows]),
        f"ndcg@{K}_off": C.mean([r[f"ndcg@{K}_off"] for r in rows]),
        f"ndcg@{K}_on": C.mean([r[f"ndcg@{K}_on"] for r in rows]),
        "enrich_only_hits": sum(r["enrich_only_hits"] for r in rows),
    }
    p = C.wilcoxon_p(paired_on, paired_off)

    out = C.ensure_results_dir()
    headers = ["attribute", "#gold", f"recall@{K}_off", f"recall@{K}_on", "delta",
               f"ndcg@{K}_off", f"ndcg@{K}_on", "enrich_only_hits"]
    C.write_csv(os.path.join(out, "enrich_text_recall.csv"), rows + [agg], headers)
    md = C.write_markdown_table(
        os.path.join(out, "enrich_text_recall.md"), rows + [agg], headers,
        title=f"Enrichment(文本):视觉属性检索 recall@{K}(富集 ON vs OFF)")
    md += (f"\n**跨 {len(paired_on)} 个 gold 商品的配对 Wilcoxon**:enrichment ON vs OFF, "
           f"p = {p:.3g} ({C.sig_mark(p)})\n")
    with open(os.path.join(out, "enrich_text_recall.md"), "w", encoding="utf-8") as f:
        f.write(md)

    C.grouped_bar(
        os.path.join(out, "enrich_text_recall.png"),
        categories=[r["attribute"] for r in rows],
        series={"enrichment OFF": [r[f"recall@{K}_off"] for r in rows],
                "enrichment ON": [r[f"recall@{K}_on"] for r in rows]},
        ylabel=f"recall@{K}",
        title="Enrichment (text): visual-attribute retrieval recall")

    print(md)
    print(f"\n结果已落盘到 {out}/enrich_text_*")


if __name__ == "__main__":
    main()
