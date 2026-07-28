"""实验公共工具:路径引导 / 评测指标 / 统计检验 / 结果落盘(CSV+Markdown)/
可选画图(matplotlib,未安装则跳过,不影响出数据)。

所有实验脚本共享本模块。纯标准库实现,唯一可选依赖是 matplotlib(仅画图用)。
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

# --------------------------------------------------------------------------- #
# 路径引导:让 `from engine import ...` 在 backend/tools/exp/ 下也能用
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))     # backend/tools/exp
BACKEND = os.path.dirname(os.path.dirname(HERE))      # backend
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

DB_PATH = os.path.join(BACKEND, "data", "catalog.db")
RESULTS_DIR = os.path.join(HERE, "results")


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# --------------------------------------------------------------------------- #
# 检索评测指标(gold 为相关 id 集合;ranked 为按序返回的 id 列表)
# --------------------------------------------------------------------------- #
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in gold)
    return hit / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    topk = ranked[:k]
    hit = sum(1 for x in topk if x in gold)
    return hit / len(topk)


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, x in enumerate(ranked, 1):
        if x in gold:
            return 1.0 / i
    return 0.0


def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    rels = [1.0 if x in gold else 0.0 for x in ranked[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# 统计检验:配对 Wilcoxon 符号秩(正态近似,给两侧 p 值)
#   注:样本很小(<10)时正态近似不精确,论文里可再用 scipy.stats.wilcoxon
#   复核(exact)。这里零依赖给出一个可用的显著性参考。
# --------------------------------------------------------------------------- #
def wilcoxon_p(a: list[float], b: list[float]) -> float:
    """配对样本 a,b 的 Wilcoxon 符号秩检验两侧 p 值(正态近似)。返回 1.0 表示
    无有效差异或样本过小。"""
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n < 1:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 处理并列:取平均秩
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_w == 0:
        return 1.0
    z = (w - mean_w) / sd_w
    # 两侧 p:正态 CDF
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return max(0.0, min(1.0, p))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sig_mark(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# --------------------------------------------------------------------------- #
# 结果落盘:CSV + Markdown 表格(论文可直接复制)
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: list[dict], headers: list[str] | None = None) -> None:
    if not rows:
        return
    headers = headers or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def write_markdown_table(path: str, rows: list[dict], headers: list[str] | None = None,
                         title: str = "") -> str:
    headers = headers or (list(rows[0].keys()) if rows else [])
    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(h, "")) for h in headers) + " |")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def dump_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 可选画图:未装 matplotlib 就静默跳过(仍产出 CSV/Markdown)
# --------------------------------------------------------------------------- #
def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无显示环境(WSL/服务器)也能存图
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def grouped_bar(path: str, categories: list[str], series: dict[str, list[float]],
                ylabel: str = "", title: str = "") -> bool:
    """分组柱状图。series = {系列名: 每个 category 的值}。返回是否画成。"""
    plt = _try_matplotlib()
    if plt is None:
        print(f"[plot skipped] matplotlib 未安装,跳过 {os.path.basename(path)} "
              f"(WSL 里 `pip install matplotlib` 后重跑即可出图)")
        return False
    n = len(categories)
    k = len(series)
    width = 0.8 / max(1, k)
    xs = list(range(n))
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4.5))
    for j, (name, vals) in enumerate(series.items()):
        offs = [x + (j - (k - 1) / 2) * width for x in xs]
        ax.bar(offs, vals, width=width, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] 已保存 {path}")
    return True


def line_at_k(path: str, ks: list[int], series: dict[str, list[float]],
              ylabel: str = "", title: str = "") -> bool:
    """随 K 变化的折线图(如 Recall@K / nDCG@K)。"""
    plt = _try_matplotlib()
    if plt is None:
        print(f"[plot skipped] matplotlib 未安装,跳过 {os.path.basename(path)}")
        return False
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, vals in series.items():
        ax.plot(ks, vals, marker="o", label=name)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] 已保存 {path}")
    return True


# --------------------------------------------------------------------------- #
# DB / 属性解析工具
# --------------------------------------------------------------------------- #
def load_catalog(db_path: str = DB_PATH) -> list[dict]:
    """读全表为 dict 列表,并解析 visual_attrs / review_aspects 两个 JSON 列。"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM laptops").fetchall()
    finally:
        conn.close()
    out: list[dict] = []
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        d["_visual"] = _parse(d.get("visual_attrs"))
        d["_review"] = _parse(d.get("review_aspects"))
        out.append(d)
    return out


def _parse(raw) -> dict:
    if not raw or not isinstance(raw, str):
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def product_category(item: dict) -> str:
    """商品的视觉品类(VL 抽的 product_category),小写。"""
    vis = (item.get("_visual") or {}).get("visual") or {}
    return str(vis.get("product_category") or "").strip().lower()


def tokenize(text: str) -> set[str]:
    """与检索/校验一致的粗分词:小写、去符号、>=3 字符。"""
    out: set[str] = set()
    for raw in (text or "").lower().replace("-", " ").replace("/", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) >= 3:
            out.add(tok)
    return out
