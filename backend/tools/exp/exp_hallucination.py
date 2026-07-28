"""实验 E-hallucination:属性来源分层能否抑制物理属性幻觉。

要证明的命题:视觉模型(Qwen-VL)只应报告"图片里看得见"的属性;一旦让它去报
**非视觉的物理/数值属性**(净重、尺寸、容量、成分、件数、续航),它会大量"脑补"
出图片根本无法确定、且商品文字资料也不支持的数值 → 幻觉。来源分层(物理属性只走
metadata、VL 只抽可见属性)从根上消除这一类幻觉。

两种模式(对同一批商品、同一张图):
  L) 分层(layered,当前系统):VL 只抽可见属性(enrichment._VL_ATTR_PROMPT),
     物理数值一律不问 → VL 产生的物理数值断言应≈0。
  N) 不分层(non-layered,消融):额外要求 VL 报 净重/尺寸/容量/成分/件数/续航。

验证金标准(可复现,无需人工):把 VL 的每条物理断言与该商品**自身的文字资料**
  (name/summary/performance/display/battery/weakness)比对:
    - 断言含数字 → 该数字出现在资料里 = 可佐证,否则 = 不可佐证(幻觉);
    - 断言不含数字(如 "stainless steel")→ 关键词出现在资料里 = 可佐证,否则幻觉。

指标(output):
    hallucination_rate = 不可佐证的物理断言数 / 物理断言总数
  - 分层 L:物理断言总数应≈0 → 无从幻觉(rate 记为 0 / NA)。
  - 不分层 N:大量物理断言,其中高比例不可佐证 → 幻觉率高。

落盘:results/hallucination_*.csv / .md / .png / _examples.json
依赖:需要 DASHSCOPE_API_KEY(Qwen-VL)。
用法:
    python tools/exp/exp_hallucination.py --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request

import common as C
from engine import enrichment as e


_NUM = re.compile(r"\d+(?:\.\d+)?")

_PHYSICAL_PROMPT = (
    "You are a product spec extractor. Look at the product image and return ONLY a "
    "JSON object with these PHYSICAL/QUANTITATIVE attributes as best you can: "
    "net_weight, dimensions, capacity_or_volume, item_count, material_composition, "
    "battery_life. Give concrete values with units (e.g. \"250 ml\", \"1.2 kg\", "
    "\"100% cotton\", \"pack of 3\"). If you must estimate, still provide your best "
    "single value. Return only the JSON object."
)


def _vl_physical(url: str, timeout: float = 40.0) -> dict:
    """非分层:让 VL 直接报物理/数值属性(故意越界,用于测幻觉)。"""
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not api_key or not url:
        return {}
    base = (os.environ.get("QWEN_CHAT_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = (os.environ.get("QWEN_VL_MODEL") or "qwen-vl-plus").strip()
    payload = {
        "model": model, "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _PHYSICAL_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Extract physical/quantitative specs as JSON."},
            ]},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return e._parse_json_object(raw)  # noqa: SLF001
    except Exception:
        return {}


def _metadata_haystack(item: dict) -> tuple[str, set[str]]:
    parts = [str(item.get(k) or "") for k in
             ("name", "summary", "performance", "display", "battery", "weakness")]
    reasons = item.get("reasons") or ""
    if isinstance(reasons, list):
        parts.append(" ".join(str(x) for x in reasons))
    else:
        parts.append(str(reasons))
    hay = " ".join(parts).lower()
    return hay, set(_NUM.findall(hay))


def _claims_from(physical: dict) -> list[tuple[str, str]]:
    """把 VL 的物理属性字典拍平成 [(key, value_str)] 断言列表(去空)。"""
    out: list[tuple[str, str]] = []
    for k, v in (physical or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        s = str(v).strip()
        if s and s.lower() not in ("", "unknown", "n/a", "none", "[]", "{}"):
            out.append((str(k), s))
    return out


def _verify_claim(value: str, hay: str, hay_nums: set[str]) -> bool:
    """该物理断言能否被商品文字资料佐证。"""
    nums = _NUM.findall(value)
    if nums:
        return any(n in hay_nums for n in nums)
    # 无数字的成分/材质类:关键词是否出现在资料里
    words = [w for w in C.tokenize(value) if len(w) >= 4]
    return any(w in hay for w in words) if words else False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if not (os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        print("!! 需要 DASHSCOPE_API_KEY(Qwen-VL)。请先设置。")
        return

    catalog = C.load_catalog()
    pool = [it for it in catalog if (it.get("image_url") or "").strip()]
    random.shuffle(pool)
    sample = pool[:args.n]
    print(f"抽样 {len(sample)} 个商品做幻觉对比 ...")

    # 分层 L:直接看当前库里 visual_attrs 里 VL 抽的东西是否包含物理数值断言
    #   (当前 prompt 明确禁止 weight/battery/size,预期物理数值断言≈0)
    layered_claims = 0
    layered_unsupported = 0
    # 不分层 N:重新调用 VL 要物理属性
    non_claims = 0
    non_unsupported = 0
    per_query: list[dict] = []
    examples: list[dict] = []

    for qi, it in enumerate(sample, 1):
        url = (it.get("image_url") or "").strip()
        hay, hay_nums = _metadata_haystack(it)

        # L:从库里已存的 visual_attrs.visual 里找"物理/数值"字段(应几乎没有)
        vis = (it.get("_visual") or {}).get("visual") or {}
        # 当前分层 prompt 只产出 category/colors/style/shape/material_look/pattern/keywords,
        # 其中可能带数字的极少;把带数字的可见属性也算作"物理数值断言"以从严评估。
        l_claims = [(k, str(v)) for k, v in vis.items()
                    if _NUM.findall(str(v))]
        for _, val in l_claims:
            layered_claims += 1
            if not _verify_claim(val, hay, hay_nums):
                layered_unsupported += 1

        # N:调用 VL 要物理属性
        phys = _vl_physical(url)
        n_claims = _claims_from(phys)
        bad_here = []
        for key, val in n_claims:
            non_claims += 1
            ok = _verify_claim(val, hay, hay_nums)
            if not ok:
                non_unsupported += 1
                bad_here.append(f"{key}={val}")

        per_query.append({
            "id": it["id"],
            "name": (it.get("name") or "")[:50],
            "layered_phys_claims": len(l_claims),
            "nonlayered_phys_claims": len(n_claims),
            "nonlayered_unsupported": len(bad_here),
        })
        if len(examples) < 6 and bad_here:
            examples.append({
                "id": it["id"],
                "name": (it.get("name") or "")[:80],
                "metadata_summary": (it.get("summary") or "")[:160],
                "VL_physical_claims": dict(n_claims),
                "unsupported_claims": bad_here,
            })
        if qi % 5 == 0:
            print(f"  ... {qi}/{len(sample)}")

    l_rate = (layered_unsupported / layered_claims) if layered_claims else 0.0
    n_rate = (non_unsupported / non_claims) if non_claims else 0.0

    summary = [
        {"mode": "Layered (VL visible-only, physical<-metadata)",
         "phys_claims_by_VL": layered_claims,
         "unsupported": layered_unsupported,
         "hallucination_rate": l_rate},
        {"mode": "Non-layered (VL also asked for physical)",
         "phys_claims_by_VL": non_claims,
         "unsupported": non_unsupported,
         "hallucination_rate": n_rate},
    ]

    out = C.ensure_results_dir()
    C.write_csv(os.path.join(out, "hallucination_perquery.csv"), per_query)
    headers = ["mode", "phys_claims_by_VL", "unsupported", "hallucination_rate"]
    md = C.write_markdown_table(
        os.path.join(out, "hallucination_summary.md"), summary, headers,
        title=f"Source Layering:物理属性幻觉对比(N={len(sample)})")
    md += ("\n说明:分层版让 VL 只抽可见属性、物理数值走 metadata,故 VL 几乎不产生"
           "物理数值断言(claims≈0),从根上没有可幻觉的空间;不分层版让 VL 直接报"
           "物理数值,其中高比例无法被商品文字资料佐证 = 幻觉。\n")
    with open(os.path.join(out, "hallucination_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    if examples:
        C.dump_json(os.path.join(out, "hallucination_examples.json"), examples)

    C.grouped_bar(
        os.path.join(out, "hallucination_summary.png"),
        categories=["phys claims / product", "hallucination rate"],
        series={"Layered": [layered_claims / max(1, len(sample)), l_rate],
                "Non-layered": [non_claims / max(1, len(sample)), n_rate]},
        ylabel="value", title="Source layering: physical-attribute hallucination")

    print(md)
    print(f"\n结果已落盘到 {out}/hallucination_*")


if __name__ == "__main__":
    main()
