"""创新点二:离线多模态知识富集 + 属性来源分层(Source-Aware Enrichment)。

对应《创新点与实验设计.md》§2。核心两件事:

1) 离线(build 阶段一次性,见文件末尾 __main__ CLI):对每个商品
   - 用图像 embedding 把商品图变成向量 → 写回库列 `image_embedding`(视觉召回用);
   - 按"属性来源分层"抽属性:视觉属性走 Qwen-VL、物理属性走 metadata(如
     weight_kg)、文字属性走 name/summary → 合成 `visual_attrs`(JSON)与可被
     关键词命中的 `enriched_text`。
   * 关键:DashScope 的 embedding 与 VL 均可直接吃远程 image_url,因此**无需把
     15k 张图下载到本地**,原图不落盘。

2) 在线(查询期,零 VL):
   - 把用户上传图算成向量,与库中商品向量做余弦 → 视觉召回 top-K;
   - 读取 `enriched_text` 供关键词召回、`visual_attrs` 供校验器判 must-have。

依赖:仅标准库(纯 Python 余弦,无 numpy)。图像/文本向量与视觉属性统一只走
   DashScope(multimodal-embedding-v1 / qwen-vl-plus,同一把 DASHSCOPE_API_KEY):
   调用失败**不再偷偷兜底成 hash 伪向量**,而是返回空——该商品对应字段留空(可见,
   便于发现 key/网络问题)。`hash` 仅在显式 provider='hash' 时供离线单测使用,
   绝不作为线上 dashscope 失败后的降级路径。
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import struct
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

from .exp_config import CONFIG

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "data", "catalog.db")

# 富集新增列
ENRICH_COLUMNS = {
    "image_embedding": "TEXT",   # base64(float32[]) —— 商品图向量
    "visual_attrs": "TEXT",      # JSON —— 分层属性 {visual:{}, physical:{}, textual:{}}
    "enriched_text": "TEXT",     # 可被关键词命中的富集检索文本
}

_HASH_DIM = 64  # hash 兜底向量维度


# --------------------------------------------------------------------------- #
# 向量编解码 + 余弦(纯 Python)
# --------------------------------------------------------------------------- #
def encode_vec(vec: list[float]) -> str:
    """float 列表 → base64(float32 packed),存进 TEXT 列。"""
    if not vec:
        return ""
    return base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")


def decode_vec(blob: str | None) -> list[float]:
    if not blob:
        return []
    try:
        raw = base64.b64decode(blob)
        n = len(raw) // 4
        return list(struct.unpack(f"<{n}f", raw[: n * 4]))
    except (ValueError, struct.error):
        return []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --------------------------------------------------------------------------- #
# Embedding provider
# --------------------------------------------------------------------------- #
def _dashscope_multimodal_embed(content: dict, timeout: float = 30.0,
                                retries: int = 4) -> list[float] | None:
    """调用 DashScope multimodal-embedding。content 形如 {"image": url} 或
    {"image": "data:image/jpeg;base64,..."} 或 {"text": "..."}。失败返回 None。

    带指数退避重试:批量富集时最易触发限流(429)/瞬时网络错误,重试可显著降低
    "静默失败→写空值"的概率(配合调用方"空值不覆盖"策略,避免冲掉已有数据)。"""
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        return None
    model = (os.environ.get("QWEN_MM_EMBED_MODEL") or "multimodal-embedding-v1").strip()
    url = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )
    body = json.dumps({"model": model, "input": {"contents": [content]}}).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts = max(1, retries)
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            embs = (((data.get("output") or {}).get("embeddings")) or [])
            if embs and isinstance(embs[0], dict):
                vec = embs[0].get("embedding")
                if isinstance(vec, list) and vec:
                    return [float(x) for x in vec]
            return None  # HTTP 200 但无向量:重试也无意义
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError):
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))  # 退避后重试(应对限流/瞬时错误)
                continue
            return None
    return None


def _hash_embed(seed: str) -> list[float]:
    """确定性伪向量(无网兜底):同一输入永远得到同一向量,可跑通召回但非语义。"""
    vec: list[float] = []
    i = 0
    while len(vec) < _HASH_DIM:
        h = hashlib.sha256(f"{seed}#{i}".encode("utf-8")).digest()
        for j in range(0, len(h), 4):
            if len(vec) >= _HASH_DIM:
                break
            val = struct.unpack("<I", h[j:j + 4])[0] / 0xFFFFFFFF
            vec.append(val - 0.5)
        i += 1
    return vec


def embed_image_url(url: str, provider: str | None = None) -> list[float]:
    """商品图 URL → 向量,仅走 DashScope;失败/无 key 返回空列表(不兜底)。
    provider='hash' 仅供离线单测显式使用。"""
    prov = provider or CONFIG.embedding_provider
    if prov == "hash":
        return _hash_embed(f"url:{url}")
    return _dashscope_multimodal_embed({"image": url}) or []


def embed_image_bytes(image_bytes: bytes, mime: str = "image/jpeg",
                      provider: str | None = None) -> list[float]:
    """图片字节 → 向量,仅走 DashScope;失败返回空列表(不兜底)。
    provider='hash' 仅供离线单测显式使用。"""
    prov = provider or CONFIG.embedding_provider
    if prov == "hash":
        seed = hashlib.sha256(image_bytes or b"").hexdigest() if image_bytes else "empty"
        return _hash_embed(f"bytes:{seed}")
    if not image_bytes:
        return []
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    return _dashscope_multimodal_embed({"image": data_url}) or []


# --------------------------------------------------------------------------- #
# 属性来源分层抽取(离线)
# --------------------------------------------------------------------------- #
_VL_ATTR_PROMPT = (
    "You are a product image annotator. Look at the product image and return ONLY a "
    "JSON object of clearly VISIBLE attributes for retrieval, keys: "
    "product_category, colors(list), style(list), shape, material_look, pattern, "
    "keywords(list of 3-8 short english search words). "
    "Only report what is clearly visible; use \"\" or [] if unsure. Do NOT guess weight, "
    "battery, size or any non-visual/physical property."
)


def extract_visual_attributes(image_ref: str, *, is_url: bool = True,
                              timeout: float = 40.0) -> dict:
    """用 Qwen-VL 从商品图抽"视觉"属性(颜色/款式/形态/图案…)。image_ref 为
    远程 url(is_url=True)或 data URL。无 key/失败返回空 dict。"""
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key or not image_ref:
        return {}
    base = (os.environ.get("QWEN_CHAT_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = (os.environ.get("QWEN_VL_MODEL") or "qwen-vl-plus").strip()
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _VL_ATTR_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_ref}},
                {"type": "text", "text": "Extract visible attributes as JSON."},
            ]},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    attempts = 3
    for attempt in range(attempts):
        req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            return _parse_json_object(raw)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError):
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))  # 退避后重试(应对限流)
                continue
            return {}
    return {}


def extract_physical_attributes(row: dict) -> dict:
    """"物理/数值"属性一律走 metadata,绝不让 VL 脑补(来源分层的核心原则)。"""
    phys: dict[str, Any] = {}
    weight = float(row.get("weight_kg") or 0)
    if weight > 0:
        phys["weight_kg"] = round(weight, 2)
        # 轻便的判定阈值随品类不同,这里给出粗粒度标签,交由在线校验/排序再精化
        phys["lightweight"] = weight <= 1.5
    battery = (row.get("battery") or "").strip()
    if battery:
        phys["battery"] = battery[:60]
    price = int(row.get("price") or 0)
    if price > 0:
        phys["price"] = price
    display = (row.get("display") or "").strip()
    if display:
        phys["display"] = display[:80]
    return phys


def _textual_attributes(row: dict) -> dict:
    out: dict[str, Any] = {}
    summ = (row.get("summary") or "").strip()
    if summ:
        out["summary"] = summ[:160]
    reasons = row.get("reasons")
    if isinstance(reasons, str) and reasons:
        out["features"] = [p.strip() for p in reasons.split("||") if p.strip()][:4]
    elif isinstance(reasons, list) and reasons:
        out["features"] = [str(p).strip() for p in reasons if str(p).strip()][:4]
    return out


def build_enriched_text(row: dict, visual: dict, physical: dict) -> str:
    """把可命中的属性拼成一段富集检索文本(补 name/summary 的稀疏)。"""
    parts: list[str] = []
    name = (row.get("name") or "").strip()
    if name:
        parts.append(name)
    # 视觉
    for key in ("product_category", "shape", "material_look", "pattern"):
        v = str(visual.get(key) or "").strip()
        if v:
            parts.append(v)
    for key in ("colors", "style", "keywords"):
        vals = visual.get(key) or []
        if isinstance(vals, list):
            parts.extend(str(v).strip() for v in vals if str(v).strip())
        elif isinstance(vals, str) and vals.strip():
            parts.append(vals.strip())
    # 物理(转成可命中的词)
    if physical.get("lightweight"):
        parts.append("lightweight portable")
    if physical.get("battery"):
        parts.append("long battery")
    # 文字
    summ = (row.get("summary") or "").strip()
    if summ:
        parts.append(summ)
    # 去重保序
    seen: set[str] = set()
    uniq = []
    for p in parts:
        low = p.lower()
        if low and low not in seen:
            seen.add(low)
            uniq.append(p)
    return "; ".join(uniq)[:600]


def _parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        text = text[s:e + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# DB schema 与读写
# --------------------------------------------------------------------------- #
def ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
    for col, typ in ENRICH_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE laptops ADD COLUMN {col} {typ}")
    conn.commit()


def has_enrichment(db_path: str = DEFAULT_DB) -> bool:
    """库里是否已存在富集列且至少有一条向量(在线是否可用视觉召回的判据)。"""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
        if "image_embedding" not in cols:
            return False
        n = conn.execute(
            "SELECT COUNT(*) FROM laptops WHERE image_embedding IS NOT NULL AND image_embedding<>''"
        ).fetchone()[0]
        return n > 0
    except sqlite3.Error:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_products_by_ids(db_path: str, ids: Iterable[str]) -> list[dict]:
    ids = [i for i in ids if i]
    if not ids:
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        qmarks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM laptops WHERE id IN ({qmarks})", ids
        ).fetchall()
        by_id = {r["id"]: _row_to_dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]  # 保持传入顺序
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    item = {k: row[k] for k in row.keys()}
    for field in ("reasons", "trade_offs"):
        raw = item.get(field) or ""
        if isinstance(raw, str):
            item[field] = [p.strip() for p in raw.split("||") if p.strip()]
    return item


# --------------------------------------------------------------------------- #
# 在线:视觉索引 + 召回
# --------------------------------------------------------------------------- #
class VisualIndex:
    """把库中商品向量一次性加载进内存,查询期做纯 Python 余弦 top-K。"""

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        self.db_path = db_path
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._cats: list[str] = []
        self._loaded = False

    def load(self) -> "VisualIndex":
        if self._loaded:
            return self
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(laptops)").fetchall()}
                if "image_embedding" in cols:
                    has_cat = "visual_attrs" in cols
                    q = "SELECT id, image_embedding" + (", visual_attrs" if has_cat else "") + \
                        " FROM laptops WHERE image_embedding IS NOT NULL AND image_embedding<>''"
                    for r in conn.execute(q):
                        vec = decode_vec(r[0 + 1] if False else r[1])
                        if not vec:
                            continue
                        self._ids.append(r[0])
                        self._vecs.append(vec)
                        cat = ""
                        if has_cat and len(r) > 2 and r[2]:
                            try:
                                cat = str((json.loads(r[2]).get("visual") or {}).get("product_category") or "")
                            except (json.JSONDecodeError, AttributeError):
                                cat = ""
                        self._cats.append(cat.lower())
            except sqlite3.Error:
                pass
            finally:
                conn.close()
        self._loaded = True
        return self

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query_vec: list[float], top_k: int = 40,
               category: str | None = None) -> list[tuple[str, float]]:
        if not query_vec or not self._ids:
            return []
        cat = (category or "").strip().lower()
        scored: list[tuple[str, float]] = []
        for i, vec in enumerate(self._vecs):
            if cat and self._cats[i] and cat not in self._cats[i] and self._cats[i] not in cat:
                continue  # 品类护栏:限制在同品类内
            scored.append((self._ids[i], cosine(query_vec, vec)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


_INDEX_CACHE: dict[str, VisualIndex] = {}


def get_index(db_path: str = DEFAULT_DB) -> VisualIndex:
    idx = _INDEX_CACHE.get(db_path)
    if idx is None:
        idx = VisualIndex(db_path).load()
        _INDEX_CACHE[db_path] = idx
    return idx


def visual_recall(image_bytes: bytes, *, mime: str = "image/jpeg",
                  db_path: str = DEFAULT_DB, top_k: int | None = None,
                  category: str | None = None) -> list[dict]:
    """在线视觉召回:用户图 → 向量 → 余弦 top-K → 返回商品 dict(带 _visual_score)。"""
    idx = get_index(db_path)
    if idx.size == 0:
        return []
    qvec = embed_query_image_bytes(image_bytes, mime)
    hits = idx.search(qvec, top_k=top_k or CONFIG.visual_top_k, category=category)
    if not hits:
        return []
    id_order = [h[0] for h in hits]
    score_map = {h[0]: h[1] for h in hits}
    products = fetch_products_by_ids(db_path, id_order)
    for p in products:
        p["_visual_score"] = round(float(score_map.get(p.get("id"), 0.0)), 4)
    return products


def embed_query_image_bytes(image_bytes: bytes, mime: str = "image/jpeg") -> list[float]:
    return embed_image_bytes(image_bytes, mime=mime)


def embed_text(text: str, provider: str | None = None) -> list[float]:
    """把一段文本(商品富集文本 / 评论聚合 / 用户查询)编码成向量。使用与图像
    相同的多模态 embedding(共享空间,便于跨模态),无网时退化为确定性 hash。
    供 reviewer.py 的评论向量与文本语义检索复用。"""
    prov = provider or CONFIG.embedding_provider
    clean = (text or "").strip()
    if prov == "hash":
        return _hash_embed(f"text:{clean}")
    if not clean:
        return []
    return _dashscope_multimodal_embed({"text": clean[:2000]}) or []


def attach_visual_scores(candidates: list[dict], image_bytes: bytes, *,
                         mime: str = "image/jpeg", db_path: str = DEFAULT_DB) -> list[dict]:
    """给一批候选(可能来自文本召回)统一补上与用户图的视觉相似度 `_visual_score`,
    使排序阶段能公平地把视觉信号纳入融合(创新点三的融合排序)。"""
    idx = get_index(db_path)
    if idx.size == 0 or not candidates or not image_bytes:
        return candidates
    qvec = embed_query_image_bytes(image_bytes, mime)
    pos = {pid: i for i, pid in enumerate(idx._ids)}  # noqa: SLF001
    for c in candidates:
        i = pos.get(c.get("id"))
        if i is not None and "_visual_score" not in c:
            c["_visual_score"] = round(cosine(qvec, idx._vecs[i]), 4)  # noqa: SLF001
    return candidates


def visual_score_for(image_bytes: bytes, product_id: str, *, mime: str = "image/jpeg",
                     db_path: str = DEFAULT_DB) -> float:
    """给定用户图与某商品 id,返回其视觉相似度(供排序融合)。"""
    idx = get_index(db_path)
    if idx.size == 0:
        return 0.0
    qvec = embed_query_image_bytes(image_bytes, mime)
    for i, pid in enumerate(idx._ids):  # noqa: SLF001 - 内部索引
        if pid == product_id:
            return cosine(qvec, idx._vecs[i])
    return 0.0


# --------------------------------------------------------------------------- #
# 离线富集 CLI
# --------------------------------------------------------------------------- #
def enrich_catalog(db_path: str = DEFAULT_DB, *, limit: int | None = None,
                   with_vl: bool = True, provider: str | None = None,
                   only_missing: bool = True, verbose: bool = True) -> dict:
    """遍历商品,写回 image_embedding / visual_attrs / enriched_text。
    图像向量走 DashScope multimodal-embedding,视觉属性默认用 Qwen-VL 抽
    (--no-vl 关闭);二者失败则对应字段留空,不兜底。VL 较贵,建议配 --limit 子集。"""
    prov = provider or CONFIG.embedding_provider
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)
    where = "WHERE image_url IS NOT NULL AND image_url<>''"
    if only_missing:
        where += " AND (image_embedding IS NULL OR image_embedding='')"
    sql = f"SELECT * FROM laptops {where}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    done = 0
    emb_ok = 0
    vl_ok = 0
    for r in rows:
        row = _row_to_dict(r)
        url = (row.get("image_url") or "").strip()
        vec = embed_image_url(url, provider=prov)
        if vec:
            emb_ok += 1
        visual = extract_visual_attributes(url) if with_vl else {}
        if visual:
            vl_ok += 1
        physical = extract_physical_attributes(row)
        textual = _textual_attributes(row)
        attrs = {"visual": visual, "physical": physical, "textual": textual}
        enriched = build_enriched_text(row, visual, physical)
        # 空值不覆盖:向量拿到才写 image_embedding,否则保留旧值(避免限流失败冲掉数据)。
        # visual_attrs / enriched_text 由 metadata + 本次 visual 合成,始终写(至少含物理/文字属性)。
        if vec:
            conn.execute(
                "UPDATE laptops SET image_embedding=?, visual_attrs=?, enriched_text=? WHERE id=?",
                (encode_vec(vec), json.dumps(attrs, ensure_ascii=False), enriched, row["id"]),
            )
        else:
            conn.execute(
                "UPDATE laptops SET visual_attrs=?, enriched_text=? WHERE id=?",
                (json.dumps(attrs, ensure_ascii=False), enriched, row["id"]),
            )
        done += 1
        if verbose and done % 100 == 0:
            print(f"  ... {done}/{len(rows)} (emb {emb_ok}, vl {vl_ok})")
            conn.commit()
    conn.commit()
    conn.close()
    _INDEX_CACHE.pop(db_path, None)  # 使缓存失效
    stats = {"processed": done, "embeddings": emb_ok, "vl_attrs": vl_ok, "provider": prov}
    if verbose:
        print(f"Done. {stats}")
    return stats


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="离线富集 catalog.db(创新点二)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条(子集实验)")
    ap.add_argument("--no-vl", dest="with_vl", action="store_false",
                    help="关闭 Qwen-VL 视觉属性抽取(默认开启)")
    ap.add_argument("--provider", default=None, choices=["dashscope", "hash"],
                    help="向量来源;默认 dashscope。hash 仅供离线测试")
    ap.add_argument("--all", action="store_true", help="重算全部(默认只补缺失)")
    args = ap.parse_args()
    enrich_catalog(args.db, limit=args.limit, with_vl=args.with_vl,
                   provider=args.provider, only_missing=not args.all)


if __name__ == "__main__":
    _main()
