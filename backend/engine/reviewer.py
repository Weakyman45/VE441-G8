"""创新点一(下半):Reviewer Agent —— 评论知识结构化。

对应《创新点与实验设计》。很多用户的"体验类软需求"(风扇安静、续航实测久、
夏天不闷脚、正码偏大…)只存在于**用户评论**里,商品 metadata 与图片都没有。
本模块把评论沉淀成**结构化、可校验**的知识(不做评论语义召回——召回由图文负责)。

离线(建库后一次性,见文件末尾 __main__ CLI):对每个商品
  1. 从评论文件中按 parent_asin(= catalog.db 的 id)聚合其评论;
  2. 用 LLM 做**基于方面的抽取**(Aspect-Based):pros / cons / issues / 各方面情感 /
     **形容词关键词**(买家反复用来描述商品/体验的形容词);无 LLM 时用确定性词频 +
     形容词启发式抽取兜底,保证 keywords 始终有值;
  3. 写回库列 review_aspects(JSON)/ review_count_used。

在线(查询期,零 LLM):
  - get_review_aspects():取某商品的 pros/cons/issues/keywords 供**校验器核对
    must-have / nice-to-have** 与前端展示。

依赖:LLM 用 qwen_client,无 key / 无网时优雅降级(方面为空,不影响主流程)。
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from collections import Counter
from typing import Any, Iterable, Iterator

from .enrichment import DEFAULT_DB, _parse_json_object

# reviewer 新增列(评论仅用于校验 must-have/nice-to-have 与展示,不再存向量)
REVIEW_COLUMNS = {
    "review_aspects": "TEXT",       # JSON —— {pros, cons, issues, aspects, summary, keywords}
    "review_count_used": "INTEGER", # 实际聚合的评论条数
}


# --------------------------------------------------------------------------- #
# DB schema
# --------------------------------------------------------------------------- #
def ensure_review_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
    for col, typ in REVIEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE laptops ADD COLUMN {col} {typ}")
    conn.commit()


def has_review_aspects(db_path: str = DEFAULT_DB) -> bool:
    """库里是否已抽到评论方面(供校验/展示使用的判据)。"""
    if not os.path.exists(db_path):
        return False
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        if "review_aspects" not in cols:
            return False
        n = conn.execute(
            "SELECT COUNT(*) FROM laptops "
            "WHERE review_aspects IS NOT NULL AND review_aspects<>'' AND review_aspects<>'{}'"
        ).fetchone()[0]
        return n > 0
    except sqlite3.Error:
        return False
    finally:
        if conn is not None:
            conn.close()


# --------------------------------------------------------------------------- #
# 读取 Amazon Reviews 2023 评论文件(JSONL / .gz)
# --------------------------------------------------------------------------- #
def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_reviews(path: str) -> Iterator[dict]:
    """逐行产出评论 dict。Amazon Reviews 2023 字段:rating/title/text/images/
    asin/parent_asin/user_id/timestamp/helpful_vote/verified_purchase。"""
    with _open_text(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _review_key(review: dict) -> str:
    return str(review.get("parent_asin") or review.get("asin") or "").strip()


def _review_quality(review: dict) -> tuple:
    """排序键:优先"已验证购买 + 高 helpful 票 + 有正文"的评论。"""
    verified = 1 if review.get("verified_purchase") else 0
    helpful = int(review.get("helpful_vote") or 0)
    has_text = 1 if (review.get("text") or "").strip() else 0
    return (verified, helpful, has_text)


def collect_reviews_for_ids(
    reviews_path: str, ids: set[str], per_product: int = 20
) -> dict[str, list[dict]]:
    """单次流式扫描评论文件,只为目标商品 id 收集至多 per_product 条**最优质**评论
    (按 verified/helpful 保留 top-N,避免把整份评论读进内存)。"""
    buckets: dict[str, list[dict]] = {}
    for review in iter_reviews(reviews_path):
        key = _review_key(review)
        if key not in ids:
            continue
        text = (review.get("text") or "").strip()
        if not text:
            continue
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_product:
            bucket.append(review)
        else:
            # 用当前评论替换桶里质量最低的一条(若更优)
            worst_idx = min(range(len(bucket)), key=lambda i: _review_quality(bucket[i]))
            if _review_quality(review) > _review_quality(bucket[worst_idx]):
                bucket[worst_idx] = review
    return buckets


# --------------------------------------------------------------------------- #
# 评论聚合 + 基于方面的抽取(离线)
# --------------------------------------------------------------------------- #
def aggregate_review_text(reviews: list[dict], max_chars: int = 3000) -> str:
    """把一个商品的多条评论拼成一段文本(用于 embedding)。优先高质量评论。"""
    ordered = sorted(reviews, key=_review_quality, reverse=True)
    parts: list[str] = []
    total = 0
    for r in ordered:
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        rating = r.get("rating")
        snippet = f"[{rating}★] {title}. {text}".strip()
        if not snippet:
            continue
        parts.append(snippet)
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


_ASPECT_PROMPT = (
    "You summarize customer reviews of ONE product into structured, retrieval-friendly "
    "aspects. Read the reviews and return ONLY a JSON object with keys: "
    "\"pros\" (list of short phrases users praised), "
    "\"cons\" (list of short phrases users complained about), "
    "\"issues\" (list of recurring defects/problems), "
    "\"aspects\" (object mapping aspect keywords like battery/noise/comfort/sizing/"
    "durability/value to \"positive\"|\"negative\"|\"mixed\"), "
    "\"keywords\" (list of 5-12 single lowercase english ADJECTIVES that buyers "
    "REPEATEDLY use to describe the product/experience — ONLY descriptive adjectives, "
    "NOT nouns or verbs, e.g. \"comfortable\", \"quiet\", \"breathable\", \"durable\", "
    "\"lightweight\", \"soft\", \"true-to-size\"), "
    "\"summary\" (one neutral sentence). "
    "Base everything ONLY on the reviews; do not invent. Keep lists <=8 items, "
    "phrases short and lowercase English."
)


# 关键词频率抽取用的英文停用词(过滤无信息词与纯情感词,保留特征/体验类词)
_KW_STOP = {
    "the", "and", "for", "with", "that", "this", "have", "has", "had", "was", "were",
    "are", "not", "you", "your", "but", "they", "them", "then", "than", "from", "just",
    "very", "all", "any", "can", "will", "would", "could", "should", "been", "being",
    "its", "out", "get", "got", "one", "two", "use", "used", "using", "really", "great",
    "good", "nice", "love", "like", "well", "product", "item", "amazon", "buy", "bought",
    "purchase", "purchased", "order", "ordered", "also", "much", "more", "most", "some",
    "when", "what", "which", "who", "how", "why", "there", "here", "because", "about",
    "into", "over", "after", "before", "again", "still", "too", "only", "even", "ever",
    "never", "every", "each", "other", "such", "own", "same", "day", "days", "time",
    "times", "week", "month", "year", "star", "stars", "review", "reviews", "little",
}

# 形容词识别(纯启发式,无第三方 NLP 依赖):常见形容词词典 + 强形容词后缀。
_ADJ_LEXICON = {
    "comfortable", "soft", "hard", "light", "lightweight", "heavy", "small", "large",
    "big", "tiny", "compact", "sturdy", "durable", "cheap", "expensive", "affordable",
    "pricey", "quiet", "loud", "noisy", "fast", "slow", "quick", "easy", "simple",
    "smooth", "rough", "warm", "cool", "cold", "hot", "dry", "wet", "thick", "thin",
    "wide", "narrow", "long", "short", "tall", "strong", "weak", "bright", "dark",
    "clear", "clean", "dirty", "fresh", "natural", "organic", "beautiful", "pretty",
    "cute", "elegant", "stylish", "modern", "classic", "fancy", "plain", "solid",
    "flimsy", "fragile", "reliable", "stable", "flexible", "stiff", "tight", "loose",
    "breathable", "waterproof", "adjustable", "washable", "reusable", "portable",
    "wireless", "spacious", "roomy", "slim", "sleek", "glossy", "matte", "shiny",
    "colorful", "vibrant", "delicious", "tasty", "gentle", "mild", "sweet", "fragrant",
    "scented", "true", "perfect", "excellent", "amazing", "wonderful", "fantastic",
    "awful", "terrible", "poor", "decent", "average", "fine", "useful", "handy",
    "functional", "practical", "accurate", "precise", "effective", "efficient",
    "powerful", "safe", "sharp", "dull", "greasy", "oily", "sticky", "crisp", "fluffy",
    "dense", "huge", "mini", "generous", "ample", "lasting", "old", "new", "original",
    "genuine", "authentic", "fake", "real", "worth", "worthy", "cozy", "warm", "chic",
    "vintage", "trendy", "neat", "tidy", "cushioned", "padded", "supportive", "snug",
}
# 强形容词后缀(以此结尾基本可判形容词)
_ADJ_SUFFIXES = ("able", "ible", "ful", "ous", "ive", "less", "ish")
# 以上述后缀结尾却其实是名词/副词的常见词,排除
_ADJ_SUFFIX_EXCEPT = {
    "business", "witness", "address", "process", "access", "progress", "purchase",
    "increase", "because", "surface", "office", "service", "notice", "price", "advice",
    "device", "choice", "source", "course", "house", "mouse", "noise", "series",
    "species", "canvas", "bristle",
    "unless", "regardless", "nevertheless", "doubtless",  # -less 结尾但非形容词
}


def _looks_adjective(word: str) -> bool:
    """启发式判断一个词是否形容词(词典 + 后缀 + 比较级/最高级归并)。"""
    w = word.lower()
    if w in _ADJ_LEXICON:
        return True
    if w in _ADJ_SUFFIX_EXCEPT:
        return False
    for suf in ("est", "er"):  # smallest/softer → small/soft
        if len(w) > len(suf) + 2 and w.endswith(suf):
            base = w[:-len(suf)]
            if base in _ADJ_LEXICON or (base + "e") in _ADJ_LEXICON:
                return True
    return len(w) >= 6 and w.endswith(_ADJ_SUFFIXES)


def extract_review_keywords(reviews: list[dict], top_k: int = 12,
                            adjectives_only: bool = True) -> list[str]:
    """确定性关键词抽取(无 LLM,纯词频):统计评论标题+正文里高频词,按文档频率
    加权。adjectives_only=True(默认)时只保留形容词(启发式识别),否则也含名词
    与二元词组。作为 LLM 关键词的补充/兜底。"""
    if not reviews:
        return []
    uni: Counter = Counter()
    docf: Counter = Counter()  # 文档频率:出现在多少条评论里
    bi: Counter = Counter()
    for r in reviews:
        text = f"{r.get('title') or ''} {r.get('text') or ''}".lower()
        toks: list[str] = []
        for w in text.replace("-", " ").replace("/", " ").split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= 3 and not w.isdigit() and w not in _KW_STOP:
                toks.append(w)
        for w in toks:
            uni[w] += 1
        for w in set(toks):
            docf[w] += 1
        for a, b in zip(toks, toks[1:]):
            if a != b:
                bi[f"{a} {b}"] += 1
    scored: list[tuple[float, str]] = []
    for w, c in uni.items():
        if adjectives_only and not _looks_adjective(w):
            continue
        scored.append((docf[w] * 2 + c, w))
    if not adjectives_only:  # 只要形容词时不产出名词短语
        for phrase, c in bi.items():
            if c >= 2:  # 只保留至少复现两次的短语,避免噪声
                scored.append((c * 2.5, phrase))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    chosen_bigrams: list[str] = []
    for _, term in scored:
        if term in out:
            continue
        if " " in term:
            chosen_bigrams.append(term)
            out.append(term)
        elif not any(term in bg.split() for bg in chosen_bigrams):
            out.append(term)  # 该词未被已选短语包含才单独保留,减少冗余
        if len(out) >= top_k:
            break
    return out


def _merge_keywords(*groups: list, limit: int = 15) -> list[str]:
    """按传入顺序合并多组关键词(LLM 优先),小写去重。"""
    out: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for term in (g or []):
            s = str(term).strip().lower()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= limit:
                return out
    return out


def extract_review_aspects(name: str, reviews: list[dict], with_llm: bool = True) -> dict:
    """把评论抽成结构化方面 + 关键词。with_llm=False 或无评论返回 {}。
    LLM 抽 pros/cons/issues/aspects/summary/keywords;关键词再并入确定性词频
    抽取结果(即使 LLM 不可用也保证 keywords 有值)。"""
    if not with_llm or not reviews:
        return {}
    data: dict = {}
    # 延迟导入,避免在线路径强依赖
    from .llm.qwen_client import chat_json, qwen_configured
    if qwen_configured():
        snippets = aggregate_review_text(reviews, max_chars=4000)
        user = json.dumps({"product": (name or "")[:120], "reviews": snippets},
                          ensure_ascii=False)
        try:
            resp = chat_json(
                [{"role": "system", "content": _ASPECT_PROMPT},
                 {"role": "user", "content": user}],
                temperature=0.1,
            )
            data = resp if isinstance(resp, dict) else _parse_json_object(str(resp))
        except Exception:
            data = {}
    # 关键词:LLM 抽的(若有)+ 词频兜底,去重合并,保证始终有 keywords
    llm_kw = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    merged = _merge_keywords(llm_kw, extract_review_keywords(reviews))
    if merged:
        data["keywords"] = merged
    return data


# --------------------------------------------------------------------------- #
# 在线:读取评论方面(供校验器 / 前端展示)
# --------------------------------------------------------------------------- #
def get_review_aspects(product_id: str, db_path: str = DEFAULT_DB) -> dict:
    """取某商品的 pros/cons/issues/aspects(供校验器判软需求 & 前端展示)。"""
    if not os.path.exists(db_path):
        return {}
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        if "review_aspects" not in cols:
            return {}
        row = conn.execute(
            "SELECT review_aspects FROM laptops WHERE id=?", (product_id,)
        ).fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            return data if isinstance(data, dict) else {}
    except (sqlite3.Error, json.JSONDecodeError):
        return {}
    finally:
        if conn is not None:
            conn.close()
    return {}


# --------------------------------------------------------------------------- #
# 离线富集 CLI
# --------------------------------------------------------------------------- #
def _target_ids(conn: sqlite3.Connection, only_missing: bool, limit: int | None) -> list[str]:
    where = ""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
    if only_missing and "review_aspects" in cols:
        # 只补"还没抽到方面"的商品(评论现只产出 aspects,不再产出 embedding)
        where = ("WHERE review_aspects IS NULL OR review_aspects='' "
                 "OR review_aspects='{}'")
    sql = f"SELECT id FROM laptops {where}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql).fetchall() if r[0]]


def enrich_reviews(db_path: str = DEFAULT_DB, *, reviews_path: str,
                   limit: int | None = None, per_product: int = 20,
                   provider: str | None = None, with_llm: bool = True,
                   only_missing: bool = True, verbose: bool = True) -> dict:
    """遍历商品,聚合其评论 → 抽"方面 / 关键词"(review_aspects) → 写回库。

    设计定位:评论仅用于**校验 must-have / nice-to-have 与前端展示**,召回由图文负责,
    因此**只产出 review_aspects,不生成任何评论向量**。provider 参数保留但忽略。"""
    _ = provider  # 兼容旧调用签名;评论不再产出向量
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_review_columns(conn)

    ids = set(_target_ids(conn, only_missing, limit))
    if verbose:
        print(f"Target products: {len(ids)}; scanning reviews from {reviews_path} ...")
    buckets = collect_reviews_for_ids(reviews_path, ids, per_product=per_product)
    if verbose:
        print(f"Products with >=1 review: {len(buckets)}")

    names = {r["id"]: (r["name"] or "") for r in
             conn.execute("SELECT id, name FROM laptops").fetchall() if r["id"] in buckets}

    done = 0
    asp_ok = 0
    for pid, reviews in buckets.items():
        aspects = extract_review_aspects(names.get(pid, ""), reviews, with_llm=with_llm)
        # 空值不覆盖:aspects 抽到才写 review_aspects(避免用 {} 冲掉已有结果)。
        sets = ["review_count_used=?"]
        params: list = [len(reviews)]
        if aspects:
            asp_ok += 1
            sets.append("review_aspects=?")
            params.append(json.dumps(aspects, ensure_ascii=False))
        params.append(pid)
        conn.execute(f"UPDATE laptops SET {', '.join(sets)} WHERE id=?", params)
        done += 1
        if verbose and done % 100 == 0:
            print(f"  ... {done}/{len(buckets)} (aspects {asp_ok})")
            conn.commit()
    conn.commit()
    conn.close()
    stats = {"processed": done, "aspects": asp_ok, "with_llm": with_llm}
    if verbose:
        print(f"Done. {stats}")
    return stats


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="离线评论富集 catalog.db(Reviewer Agent)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--reviews", required=True,
                    help="Amazon Reviews 2023 评论文件(.jsonl 或 .jsonl.gz)")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个商品(子集)")
    ap.add_argument("--per-product", type=int, default=20, help="每个商品最多聚合的评论数")
    ap.add_argument("--provider", default=None, choices=["dashscope", "hash"],
                    help="(已废弃)评论不再生成向量,此参数忽略")
    ap.add_argument("--no-llm", action="store_true",
                    help="不调用 LLM 抽方面(将不产出 aspects;评论富集本就依赖 LLM)")
    ap.add_argument("--all", action="store_true", help="重算全部(默认只补缺失的 aspects)")
    args = ap.parse_args()
    enrich_reviews(args.db, reviews_path=args.reviews, limit=args.limit,
                   per_product=args.per_product, provider=args.provider,
                   with_llm=not args.no_llm, only_missing=not args.all)


if __name__ == "__main__":
    _main()
