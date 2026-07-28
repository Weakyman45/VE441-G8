"""实验 J:Rejector / Verifier —— 校验闸门对"输出与用户要求匹配度"的贡献。

要证明的命题:在召回与排序之间插入约束感知校验(Reject Agent),能把"违反用户
硬约束(品类不符 / 缺 must-have)"的商品剔除,使最终展示的商品与用户要求高度匹配;
关闭它则候选原样进入结果,匹配度=候选里正样本的基础占比(低)。

评测协议(留一构造,自动金标准,无需人工):
  对每个测试查询(品类 c + must-have 属性 a):
    - 正样本 positives = 品类为 c 且**结构化标注**里具备属性 a 的商品(真正满足要求);
    - 负样本(错品类)  = 品类不是 c 的商品(违反品类约束);
    - 负样本(缺属性)  = 品类为 c 但没有属性 a 的商品(违反 must-have)。
  候选池 = 三者混合(均衡采样),直接喂给校验器(跳过召回以隔离 Verifier)。
  金标准 gold = positives。品类用 visual_attrs.product_category(结构化标签),属性用
  review_aspects / visual_attrs 的结构化列表判定,尽量独立于校验器的"文本命中"机制。

指标(output):对每种模式(off / rule / llm)统计**输出列表**的:
    - match_rate(= precision@kept):**输出中真正满足用户要求的比例(核心"匹配度")**;
    - coverage(= recall):正样本被保留的比例(校验器有没有误杀正样本);
    - F1:上两者的调和均值;
    - avg_output_size:平均输出商品数(off 会很大,on 会收敛)。

落盘:results/rejector_*.csv / .md / .png
用法:
    python tools/exp/exp_rejector.py                      # off vs rule(确定、免 API)
    python tools/exp/exp_rejector.py --llm                # 额外加 LLM 判官(需 DASHSCOPE_API_KEY)
    python tools/exp/exp_rejector.py --max-queries 40 --min-pos 5
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter

import common as C
from build_gold import (_review_tokens, _visual_tokens, group_by_category,
                        _EXCLUDE_ATTR)

from engine.exp_config import ExpConfig
from engine.models import PreferenceProfile
from engine import verifier
from server import _STOPWORDS


def _evidence(it: dict) -> set[str]:
    """商品"结构化"具备的属性 token(评论方面 ∪ 视觉属性),用于金标准判定。"""
    return _review_tokens(it) | _visual_tokens(it)


def _valid_attr(tok: str) -> bool:
    return (len(tok) >= 4 and tok.isascii()
            and tok not in _EXCLUDE_ATTR and tok not in _STOPWORDS)


def _cfg(mode: str) -> ExpConfig:
    return ExpConfig(
        modality_routing=True, visual_recall=True, enrichment=True,
        source_layering=True, verifier=mode, reviews=True,
        visual_top_k=40, review_top_k=20, embedding_provider="dashscope",
    )


def _kept_ids(profile, pool: list[dict], cfg: ExpConfig) -> set[str]:
    cands = [dict(it) for it in pool]
    res = verifier.verify_candidates(profile, cands, cfg)
    return {str(k.get("id")) for k in res.kept}


def _prf(kept: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not kept:
        return 0.0, 0.0, 0.0
    tp = len(kept & gold)
    precision = tp / len(kept)
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cat", type=int, default=6, help="品类最小成员数")
    ap.add_argument("--min-pos", type=int, default=5, help="每个查询最少正样本数")
    ap.add_argument("--attrs-per-cat", type=int, default=3, help="每个品类最多取几个属性")
    ap.add_argument("--max-queries", type=int, default=40)
    ap.add_argument("--llm", action="store_true", help="额外评测 LLM 判官(需 API)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    catalog = C.load_catalog()
    by_id = {it["id"]: it for it in catalog}
    cats = group_by_category(catalog, min_size=args.min_cat)
    cat_of = {it["id"]: (it.get("_visual") or {}).get("visual", {}).get("product_category", "")
              for it in catalog}
    cat_of = {k: str(v or "").strip().lower() for k, v in cat_of.items()}
    all_wrong_pool_base = list(by_id.keys())

    # 组装测试查询 (category, attribute)
    queries: list[tuple[str, str]] = []
    for c, members in cats.items():
        cnt: Counter = Counter()
        for pid in members:
            for t in _evidence(by_id[pid]):
                if _valid_attr(t):
                    cnt[t] += 1
        picked = 0
        for a, n_pos in cnt.most_common():
            if n_pos >= args.min_pos:
                queries.append((c, a))
                picked += 1
            if picked >= args.attrs_per_cat:
                break
    random.shuffle(queries)
    queries = queries[:args.max_queries]
    if not queries:
        print("没有可用查询,试试 --min-cat 4 --min-pos 3")
        return
    print(f"构造了 {len(queries)} 条测试查询(品类×属性)")

    modes = ["off", "rule"] + (["llm"] if args.llm else [])
    if args.llm and not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        print("!! --llm 需要 DASHSCOPE_API_KEY;未检测到,跳过 llm。")
        modes = ["off", "rule"]

    # 每种模式收集每条查询的 precision/recall/f1/output_size
    agg = {m: {"prec": [], "rec": [], "f1": [], "size": []} for m in modes}
    per_query: list[dict] = []

    for (c, a) in queries:
        members = cats[c]
        positives = [pid for pid in members if a in _evidence(by_id[pid])]
        if len(positives) < args.min_pos:
            continue
        neg_missing = [pid for pid in members if a not in _evidence(by_id[pid])]
        wrong_cat = [pid for pid in all_wrong_pool_base if cat_of.get(pid) and cat_of[pid] != c]

        n = len(positives)
        neg_missing_s = random.sample(neg_missing, min(len(neg_missing), n))
        wrong_cat_s = random.sample(wrong_cat, min(len(wrong_cat), n))
        pool_ids = positives + neg_missing_s + wrong_cat_s
        pool = [by_id[i] for i in pool_ids]
        gold = set(positives)

        profile = PreferenceProfile(category=c, hard=[a], raw_query=f"{a} {c}")

        row = {"category": c, "attribute": a, "#pos": n,
               "#pool": len(pool_ids)}
        for m in modes:
            kept = _kept_ids(profile, pool, _cfg(m))
            p, r, f1 = _prf(kept, gold)
            agg[m]["prec"].append(p); agg[m]["rec"].append(r)
            agg[m]["f1"].append(f1); agg[m]["size"].append(len(kept))
            row[f"{m}_match"] = p
            row[f"{m}_recall"] = r
            row[f"{m}_size"] = len(kept)
        per_query.append(row)

    if not per_query:
        print("正样本不足,降低 --min-pos 再试。")
        return

    # 汇总表:每种模式一行
    summary = []
    for m in modes:
        summary.append({
            "mode": m,
            "match_rate(precision)": C.mean(agg[m]["prec"]),
            "coverage(recall)": C.mean(agg[m]["rec"]),
            "F1": C.mean(agg[m]["f1"]),
            "avg_output_size": C.mean(agg[m]["size"]),
        })

    out = C.ensure_results_dir()
    C.write_csv(os.path.join(out, "rejector_perquery.csv"), per_query)
    headers = ["mode", "match_rate(precision)", "coverage(recall)", "F1", "avg_output_size"]
    md = C.write_markdown_table(
        os.path.join(out, "rejector_summary.md"), summary, headers,
        title=f"Rejector 消融:输出与用户要求的匹配度(N={len(per_query)} 条查询)")
    # off vs rule 的配对 Wilcoxon(match_rate)
    if "rule" in modes:
        p = C.wilcoxon_p(agg["rule"]["prec"], agg["off"]["prec"])
        md += (f"\n**match_rate 配对 Wilcoxon(rule vs off)**:p = {p:.3g} ({C.sig_mark(p)})\n")
    with open(os.path.join(out, "rejector_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    C.grouped_bar(
        os.path.join(out, "rejector_summary.png"),
        categories=["match_rate", "coverage", "F1"],
        series={m: [C.mean(agg[m]["prec"]), C.mean(agg[m]["rec"]), C.mean(agg[m]["f1"])]
                for m in modes},
        ylabel="score", title="Rejector ablation: output-vs-requirement match")

    print(md)
    print(f"\n结果已落盘到 {out}/rejector_*")


if __name__ == "__main__":
    main()
