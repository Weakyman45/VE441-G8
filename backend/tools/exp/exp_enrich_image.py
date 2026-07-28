"""实验 E-image:有图片输入时,两种利用图片的方式哪个更准。

对比两条链路(输入都是同一张用户图片):
  A) 图→图(image-to-image):把输入图编码成向量,与商品图向量做余弦 → 视觉召回
     (enrichment 的 image_embedding + visual_recall)。
  B) 图→文(image-to-text):先用 Qwen-VL 把输入图抽成文字需求(search_keywords),
     再用这些关键词与**商品名字**做文本检索(describe_shopping_image + server.search)。

评测协议(留一法,不含自身):
  输入(input):把某商品 P 的图片 url 当作"用户上传的图片"。
  相关集 gold:与 P **同一视觉品类**(visual_attrs.product_category)的其它商品
             (排除 P 自己)—— 即"给一张图,能不能召回同类/视觉相似的商品"。
  分别用 A、B 得到排序,算 precision@K / nDCG@K / recall@K。

输出(output):
  - 汇总:A vs B 的 mean precision@K / nDCG@K / recall@K
  - 每条查询的明细(CSV)
  - 跨查询的配对 Wilcoxon(precision@K:A vs B)
  - 柱状图 A vs B
落盘:results/enrich_image_*.csv / .md / .png

依赖:需要 DASHSCOPE_API_KEY(A 用多模态 embedding,B 用 Qwen-VL)。
用法:
    python tools/exp/exp_enrich_image.py --n 30 --k 10
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


def _fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
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


def _image_to_image(url: str, self_id: str, k: int,
                    debug: bool = False) -> list[str]:
    """A:图→图。返回按余弦排序的 id(排除自身)。
    强制 provider='dashscope',避免环境里 VS_EMBEDDING_PROVIDER=hash 导致查询向量
    维度(64)与库向量(1024)不一致 → cosine 恒为 0 → 召回失效。"""
    idx = e.get_index(C.DB_PATH)
    qvec = e.embed_image_url(url, provider="dashscope")
    if debug:
        dim_idx = len(idx._vecs[0]) if idx.size else 0  # noqa: SLF001
        print(f"    [debug] idx.size={idx.size} idx_dim={dim_idx} "
              f"qvec_dim={len(qvec)} qvec_empty={not qvec}")
    if not qvec or idx.size == 0:
        return []
    hits = idx.search(qvec, top_k=k + 5)
    if debug:
        print(f"    [debug] top hits(score): "
              + ", ".join(f"{pid}:{s:.3f}" for pid, s in hits[:5]))
    return [pid for pid, _ in hits if pid != self_id][:k]


def _image_to_text(img: bytes, mime: str, self_id: str, k: int, limit: int,
                   debug: bool = False) -> list[str]:
    """B:图→文。VL 抽 search_keywords → 文本检索商品名。返回 id(排除自身)。"""
    from engine.llm.vision import describe_shopping_image
    analysis = describe_shopping_image(img, mime_type=mime)
    kws = analysis.get("search_keywords") or []
    q = " ".join(str(x) for x in kws).strip()
    if debug:
        print(f"    [debug] VL keywords={kws!r} -> q={q!r}")
    if not q:
        return []
    os.environ["VS_ENRICHMENT"] = "1"
    rows = server.search({"q": [q], "limit": [str(limit)], "sort": ["popular"]})
    return [str(r.get("id")) for r in rows if str(r.get("id")) != self_id][:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="抽样的查询商品数")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--min-cat", type=int, default=4, help="品类最小成员数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true",
                    help="打印前几条查询的维度/召回明细,用于排查全 0 问题")
    args = ap.parse_args()
    K = args.k
    random.seed(args.seed)

    if not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        print("!! 需要 DASHSCOPE_API_KEY(A 用 embedding,B 用 Qwen-VL)。请先设置。")
        return

    catalog = C.load_catalog()
    by_id = {it["id"]: it for it in catalog}
    cats = group_by_category(catalog, min_size=args.min_cat)
    cat_of = {it["id"]: C.product_category(it) for it in catalog}

    # 候选查询:有图 url、有向量、且所属品类可用(同类里还有别人)
    pool = [it["id"] for it in catalog
            if (it.get("image_url") or "").strip()
            and (it.get("image_embedding") or "").strip()
            and cat_of[it["id"]] in cats]
    random.shuffle(pool)
    queries = pool[:args.n]
    print(f"候选查询 {len(queries)} 条(共可用 {len(pool)})")

    per_query: list[dict] = []
    prA, prB = [], []   # precision@K 配对
    ndA, ndB = [], []
    rcA, rcB = [], []

    for qi, pid in enumerate(queries, 1):
        it = by_id[pid]
        url = (it.get("image_url") or "").strip()
        cat = cat_of[pid]
        gold = set(cats[cat]) - {pid}
        if not gold or not url:
            continue

        dbg = args.debug and qi <= 3
        if dbg:
            print(f"  [debug] q#{qi} id={pid} cat={cat!r} #gold={len(gold)}")
        ranked_a = _image_to_image(url, pid, K, debug=dbg)
        img = _fetch_bytes(url)
        ranked_b = _image_to_text(img, _mime_of(url), pid, K, args.limit, debug=dbg) if img else []
        if dbg:
            print(f"    [debug] ranked_a cats={[cat_of.get(x) for x in ranked_a[:5]]}")
            print(f"    [debug] ranked_b cats={[cat_of.get(x) for x in ranked_b[:5]]}")
            print(f"    [debug] gold∩A={len(set(ranked_a)&gold)} gold∩B={len(set(ranked_b)&gold)}")

        pa, pb = C.precision_at_k(ranked_a, gold, K), C.precision_at_k(ranked_b, gold, K)
        na, nb = C.ndcg_at_k(ranked_a, gold, K), C.ndcg_at_k(ranked_b, gold, K)
        ra, rb = C.recall_at_k(ranked_a, gold, K), C.recall_at_k(ranked_b, gold, K)
        prA.append(pa); prB.append(pb); ndA.append(na); ndB.append(nb); rcA.append(ra); rcB.append(rb)

        per_query.append({
            "query_id": pid,
            "category": cat,
            "#gold": len(gold),
            f"A_precision@{K}": pa, f"B_precision@{K}": pb,
            f"A_ndcg@{K}": na, f"B_ndcg@{K}": nb,
            f"A_recall@{K}": ra, f"B_recall@{K}": rb,
        })
        if qi % 5 == 0:
            print(f"  ... {qi}/{len(queries)}")

    if not per_query:
        print("没有可用查询(可能同类样本太少),降低 --min-cat 再试。")
        return

    summary = [
        {"metric": f"precision@{K}", "A_image2image": C.mean(prA), "B_image2text": C.mean(prB),
         "delta(A-B)": C.mean(prA) - C.mean(prB), "wilcoxon_p": C.wilcoxon_p(prA, prB)},
        {"metric": f"ndcg@{K}", "A_image2image": C.mean(ndA), "B_image2text": C.mean(ndB),
         "delta(A-B)": C.mean(ndA) - C.mean(ndB), "wilcoxon_p": C.wilcoxon_p(ndA, ndB)},
        {"metric": f"recall@{K}", "A_image2image": C.mean(rcA), "B_image2text": C.mean(rcB),
         "delta(A-B)": C.mean(rcA) - C.mean(rcB), "wilcoxon_p": C.wilcoxon_p(rcA, rcB)},
    ]
    for s in summary:
        s["sig"] = C.sig_mark(s["wilcoxon_p"])

    out = C.ensure_results_dir()
    C.write_csv(os.path.join(out, "enrich_image_perquery.csv"), per_query)
    headers = ["metric", "A_image2image", "B_image2text", "delta(A-B)", "wilcoxon_p", "sig"]
    md = C.write_markdown_table(
        os.path.join(out, "enrich_image_summary.md"), summary, headers,
        title=f"Enrichment(图片):图→图 vs 图→文(N={len(per_query)}, K={K},相关性=同视觉品类)")
    with open(os.path.join(out, "enrich_image_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    C.grouped_bar(
        os.path.join(out, "enrich_image_summary.png"),
        categories=[f"precision@{K}", f"ndcg@{K}", f"recall@{K}"],
        series={"A: image->image": [C.mean(prA), C.mean(ndA), C.mean(rcA)],
                "B: image->text": [C.mean(prB), C.mean(ndB), C.mean(rcB)]},
        ylabel="score", title="Enrichment (image): image-to-image vs image-to-text")

    print(md)
    if C.mean(prA) == 0 and C.mean(rcA) == 0:
        print("\n[!] A(图→图)全为 0,极可能是查询向量维度与库向量不一致或 embedding "
              "失败。请加 --debug 重跑,检查 qvec_dim 是否等于 idx_dim、qvec_empty 是否为 True;"
              "并确认 DASHSCOPE_API_KEY 对 multimodal-embedding 端点有效、未设 "
              "VS_EMBEDDING_PROVIDER=hash。")
    print(f"\n结果已落盘到 {out}/enrich_image_*")


if __name__ == "__main__":
    main()
