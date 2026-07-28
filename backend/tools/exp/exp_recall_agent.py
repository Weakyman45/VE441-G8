"""实验 E-recall:文本&图片召回 Agent 的消融(text-only / image-only / fused）。

对同一个"图文查询",比较一个基线 + 三种召回配置在检索质量上的差异,验证融合
（同时用文本关键词召回 + 图片视觉召回）是否优于任一单模态、以及是否优于"无 LLM"的
朴素关键词检索。T/I/F 三种条件都走**真实的 RecallAgent**（通过构造不同的 RoutePlan
打开/关闭两路召回）：

  N   keyword-noLLM: 基线——**不经过 LLM** 的朴素关键词检索：直接把商品原始标题交给
                   server.search 做正则分词 + 关键词命中（不走 analyze.py 的 LLM 抽词）。
  T   text-only  : 只做关键词召回（run_search，用 LLM/VL 提炼的关键词: name + enriched_text）。
  I   image-only : 只做视觉向量召回（enrichment.image_embedding + visual_recall，
                   纯视觉、不给品类，避免把 gold 品类泄漏给检索）。
  T+I fused       : 两路并集去重 + 给全部候选补视觉分（完整 Agent）。

排序（用于 precision/nDCG）：统一用与 recommend_worker 一致的排序键——
  文本/图文：先按关键词命中商品名的个数，再按视觉相似度；
  纯图片：直接按视觉相似度。
（不调用 recommend_worker 本体，避免其 LLM 摘要开销；排序键与其保持一致。）

评测协议（留一法，不含自身）：
  查询输入：商品 P 的视觉关键词当"用户文字"、P 的图片当"用户上传图"。
  相关集 gold：与 P **同一视觉品类**（visual_attrs.product_category）的其它商品。
  指标：precision@K / nDCG@K / recall@K；配对 Wilcoxon 比较 fused vs T、fused vs I。

输出（results/）：
  recall_agent_perquery.csv / recall_agent_summary.md / recall_agent_summary.png

依赖：DASHSCOPE_API_KEY（image-only 与 fused 需要多模态 embedding 编码用户图）。
用法：
    python tools/exp/exp_recall_agent.py --n 30 --k 30
"""
from __future__ import annotations

import argparse
import os
import random
import urllib.request

import common as C
from build_gold import group_by_category

import server
from engine import enrichment as e
from engine.models import SessionState, PreferenceProfile
from engine.modality_router import RoutePlan
from engine.worker.recall_worker import RecallAgent
from engine.worker.recommend_worker import _keyword_tokens


def _fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    """下载图片字节（带 UA，规避部分 CDN 对无 UA 请求的拒绝）。失败返回 b""。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return b""


def _mime_of(url: str) -> str:
    u = url.lower()
    if u.endswith(".png"):
        return "image/png"
    if u.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _text_query_for(item: dict) -> list[str]:
    """把商品的视觉关键词当作"用户文字需求"；无则退回品类词。"""
    vis = (item.get("_visual") or {}).get("visual") or {}
    kws = [str(x).strip() for x in (vis.get("keywords") or []) if str(x).strip()]
    if kws:
        return kws
    cat = C.product_category(item)
    return [cat] if cat else []


def _route(modality: str, *, text: bool, visual: bool) -> RoutePlan:
    return RoutePlan(
        modality=modality,
        do_text_recall=text,
        do_visual_recall=visual,
        infer_category_from_image=False,  # 不从图推品类：避免把 gold 品类泄漏给检索
        reverse_verify=False,
        do_verify=False,
        reason=f"ablation:{modality}",
    )


def _rank(cands: list[dict], query_tokens: set[str], modality: str) -> list[str]:
    """与 recommend_worker 一致的排序键，返回排序后的 id 列表。"""
    scored = []
    for it in cands:
        name = str(it.get("name") or "").lower()
        rel = sum(1 for t in query_tokens if t in name)
        vis = float(it.get("_visual_score") or 0.0)
        scored.append((rel, vis, it))
    if modality == "image":
        scored.sort(key=lambda x: -x[1])            # 纯图片：按视觉相似度
    else:
        scored.sort(key=lambda x: (-x[0], -x[1]))   # 文本/图文：相关性→视觉
    return [str(it.get("id")) for _, _, it in scored]


def _recall_rank(agent: RecallAgent, cfg, *, self_id: str, kws: list[str],
                 query_image, modality: str, text: bool, visual: bool) -> list[str]:
    """构造一次召回并排序，返回 top 结果 id 列表（留一法：排除查询自身）。"""
    pref = PreferenceProfile(search_keywords=list(kws))
    state = SessionState(session_id="exp", preference=pref)
    res = agent.recall(state=state, route_plan=_route(modality, text=text, visual=visual),
                       cfg=cfg, query_image=query_image)
    ranked = _rank(res.candidates, _keyword_tokens(kws), modality)
    return [x for x in ranked if x != self_id]


def _no_llm_keyword(search_fn, *, self_id: str, raw_text: str, depth: int) -> list[str]:
    """N 基线:关键词检索,但**不经过 LLM**。直接把原始文本(商品标题)交给
    server.search 做朴素正则分词 + 关键词命中(不走 analyze.py 的 LLM 抽词),
    代表"没有 LLM 参与"的纯关键词召回。留一法:排除查询自身。"""
    rows = search_fn({"q": [raw_text], "limit": [str(depth)], "sort": ["popular"]})
    return [str(r.get("id")) for r in rows if str(r.get("id")) != self_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="抽样的查询商品数")
    ap.add_argument("--k", type=int, default=30, help="评测 top-K")
    ap.add_argument("--pool", type=int, default=80, help="视觉召回候选池深度(visual_top_k)")
    ap.add_argument("--min-cat", type=int, default=4, help="品类最小成员数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    K = args.k
    random.seed(args.seed)

    if not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        print("!! 需要 DASHSCOPE_API_KEY（image-only 与 fused 需编码用户图）。请先设置。")
        return

    from types import SimpleNamespace
    cfg = SimpleNamespace(visual_top_k=max(K, args.pool))
    agent = RecallAgent(server.search)

    catalog = C.load_catalog()
    by_id = {it["id"]: it for it in catalog}
    cats = group_by_category(catalog, min_size=args.min_cat)
    cat_of = {it["id"]: C.product_category(it) for it in catalog}

    # 候选查询：有图 url、有图向量、有可用文字（视觉关键词/品类）、且同类里还有别人
    pool = [it["id"] for it in catalog
            if (it.get("image_url") or "").strip()
            and (it.get("image_embedding") or "").strip()
            and cat_of[it["id"]] in cats
            and _text_query_for(it)]
    random.shuffle(pool)
    queries = pool[:args.n]
    print(f"候选查询 {len(queries)} 条（共可用 {len(pool)}）")

    per_query: list[dict] = []
    # 四条件 × 三指标 的配对样本（N=无 LLM 的原始关键词检索基线；T/I/F=召回三设置）
    depth = max(K, args.pool)
    P = {"N": [], "T": [], "I": [], "F": []}   # precision
    Nd = {"N": [], "T": [], "I": [], "F": []}  # ndcg
    R = {"N": [], "T": [], "I": [], "F": []}   # recall

    for qi, pid in enumerate(queries, 1):
        it = by_id[pid]
        url = (it.get("image_url") or "").strip()
        cat = cat_of[pid]
        gold = set(cats[cat]) - {pid}
        kws = _text_query_for(it)
        if not gold or not url or not kws:
            continue
        img = _fetch_bytes(url)
        if not img:
            if args.debug:
                print(f"  [debug] q#{qi} id={pid} 图片下载失败，跳过")
            continue
        query_image = (img, _mime_of(url))

        ranked_N = _no_llm_keyword(server.search, self_id=pid,
                                   raw_text=str(it.get("name") or ""), depth=depth)
        ranked_T = _recall_rank(agent, cfg, self_id=pid, kws=kws, query_image=None,
                                modality="text", text=True, visual=False)
        ranked_I = _recall_rank(agent, cfg, self_id=pid, kws=[], query_image=query_image,
                                modality="image", text=False, visual=True)
        ranked_F = _recall_rank(agent, cfg, self_id=pid, kws=kws, query_image=query_image,
                                modality="text_image", text=True, visual=True)

        if args.debug and qi <= 3:
            print(f"  [debug] q#{qi} id={pid} cat={cat!r} #gold={len(gold)} kws={kws}")
            print(f"    N top cats={[cat_of.get(x) for x in ranked_N[:5]]}")
            print(f"    T top cats={[cat_of.get(x) for x in ranked_T[:5]]}")
            print(f"    I top cats={[cat_of.get(x) for x in ranked_I[:5]]}")
            print(f"    F top cats={[cat_of.get(x) for x in ranked_F[:5]]}")

        row = {"query_id": pid, "category": cat, "#gold": len(gold)}
        for tag, ranked in (("N", ranked_N), ("T", ranked_T), ("I", ranked_I), ("F", ranked_F)):
            p = C.precision_at_k(ranked, gold, K)
            n = C.ndcg_at_k(ranked, gold, K)
            r = C.recall_at_k(ranked, gold, K)
            P[tag].append(p); Nd[tag].append(n); R[tag].append(r)
            row[f"{tag}_precision@{K}"] = p
            row[f"{tag}_ndcg@{K}"] = n
            row[f"{tag}_recall@{K}"] = r
        per_query.append(row)
        if qi % 5 == 0:
            print(f"  ... {qi}/{len(queries)}")

    if not per_query:
        print("没有可用查询（可能同类样本太少或图片下载失败），降低 --min-cat 或检查网络。")
        return

    def _mk(metric_name: str, d: dict) -> dict:
        row = {
            "metric": metric_name,
            "none": C.mean(d["N"]),
            "text_only": C.mean(d["T"]),
            "image_only": C.mean(d["I"]),
            "fused": C.mean(d["F"]),
        }
        # 每个召回条件 vs "无 LLM 原始关键词检索"基线 N（体现 LLM/图片/融合的价值）
        for tag in ("T", "I", "F"):
            row[f"delta({tag}-N)"] = C.mean(d[tag]) - C.mean(d["N"])
            p = C.wilcoxon_p(d[tag], d["N"])
            row[f"p({tag}-N)"] = p
            row[f"sig({tag}-N)"] = C.sig_mark(p)
        # 融合 vs 单模态（体现融合相对单路的增益）
        for tag in ("T", "I"):
            row[f"delta(F-{tag})"] = C.mean(d["F"]) - C.mean(d[tag])
            p = C.wilcoxon_p(d["F"], d[tag])
            row[f"p(F-{tag})"] = p
            row[f"sig(F-{tag})"] = C.sig_mark(p)
        return row

    summary = [
        _mk(f"precision@{K}", P),
        _mk(f"ndcg@{K}", Nd),
        _mk(f"recall@{K}", R),
    ]

    out = C.ensure_results_dir()
    C.write_csv(os.path.join(out, "recall_agent_perquery.csv"), per_query)
    headers = ["metric", "none", "text_only", "image_only", "fused",
               "delta(T-N)", "sig(T-N)", "delta(I-N)", "sig(I-N)",
               "delta(F-N)", "sig(F-N)",
               "delta(F-T)", "sig(F-T)", "delta(F-I)", "sig(F-I)"]
    md = C.write_markdown_table(
        os.path.join(out, "recall_agent_summary.md"), summary, headers,
        title=f"Recall Agent 消融：keyword-noLLM vs text vs image vs fused "
              f"(N={len(per_query)}, K={K}，相关性=同视觉品类)")

    C.grouped_bar(
        os.path.join(out, "recall_agent_summary.png"),
        categories=[f"precision@{K}", f"ndcg@{K}", f"recall@{K}"],
        series={"keyword (no LLM)": [C.mean(P["N"]), C.mean(Nd["N"]), C.mean(R["N"])],
                "text-only": [C.mean(P["T"]), C.mean(Nd["T"]), C.mean(R["T"])],
                "image-only": [C.mean(P["I"]), C.mean(Nd["I"]), C.mean(R["I"])],
                "fused (T+I)": [C.mean(P["F"]), C.mean(Nd["F"]), C.mean(R["F"])]},
        ylabel="score", title="Recall Agent ablation: keyword(no LLM) vs text vs image vs fused")

    print(md)
    print(f"结果已落盘到 {out}/recall_agent_*")


if __name__ == "__main__":
    main()
