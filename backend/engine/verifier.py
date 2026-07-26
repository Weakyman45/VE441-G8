"""创新点三:约束感知的校验-重排序之"校验闸门"(Constraint-Aware Verifier)。

对应《创新点与实验设计.md》§3。在召回与排序之间插入独立的 Verifier / Reject
Agent,把"硬约束校验"(品类一致 + must-have 满足)与"软偏好排序"解耦:

  - 二值判定(LLM-as-Judge):对候选判 品类是否一致、must-have 是否满足;
  - 规则粗筛:预算 / 平台等可靠的结构化硬约束直接用规则剔除(便宜且确定);
  - unknown 容错:信息缺失时按"不违反"处理(保留而非硬拒),避免过度拒绝导致空结果。

消融(VS_VERIFIER):
  off        → 不校验,全部保留;
  rule       → 仅规则(品类 token + must-have token 命中,unknown 容错);
  llm_strict → LLM 判定,但 unknown 即拒;
  llm        → LLM 判定 + unknown 容错(完整版,默认)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .exp_config import ExpConfig, CONFIG
from .llm.qwen_client import chat_json, qwen_configured

# 与 recommend_worker 一致的通用词(不作为品类/命中的有效信号)
_GENERIC = {
    "the", "and", "for", "with", "your", "new", "best", "set", "pack",
    "white", "black", "red", "blue", "green", "gray", "grey", "pink", "silver",
    "color", "colors", "size", "women", "womens", "men", "mens", "unisex",
    "premium", "quality", "durable", "classic", "style", "design",
}

_MAX_LLM_CANDIDATES = 30  # 单次 LLM 批量判的候选上限(控成本);其余用规则


@dataclass
class VerifyResult:
    kept: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)   # {id, name, reason}
    method: str = "off"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": len(self.kept),
            "rejected": self.rejected[:12],
            "method": self.method,
        }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in (text or "").lower().replace("-", " ").replace("/", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) >= 3 and tok not in _GENERIC:
            out.add(tok)
    return out


def _item_haystack(item: dict) -> str:
    parts = [str(item.get(k) or "") for k in
             ("name", "summary", "enriched_text", "display", "performance", "platform")]
    reasons = item.get("reasons")
    if isinstance(reasons, list):
        parts.append(" ".join(str(x) for x in reasons))
    elif isinstance(reasons, str):
        parts.append(reasons)
    va = item.get("visual_attrs")
    if isinstance(va, str) and va:
        parts.append(va)
    # 评论关键词 / pros / 摘要:让"只在评论里出现的软需求"(安静/透气/正码…)可被命中
    ra = item.get("review_aspects")
    if isinstance(ra, str) and ra:
        try:
            d = json.loads(ra)
            if isinstance(d, dict):
                parts.append(" ".join(str(x) for x in (d.get("keywords") or [])))
                parts.append(" ".join(str(x) for x in (d.get("pros") or [])))
                if d.get("summary"):
                    parts.append(str(d.get("summary")))
        except (json.JSONDecodeError, TypeError):
            pass
    return " ".join(parts).lower()


def _cmp_budget(budget: int | None) -> int | None:
    if not budget:
        return None
    return budget if budget < 3000 else max(800, int(budget / 7))  # 粗略 ¥→$


# --------------------------------------------------------------------------- #
# 规则粗筛(所有 verifier-on 模式共用:可靠的结构化硬约束)
# --------------------------------------------------------------------------- #
def _rule_prefilter(profile, item: dict) -> str | None:
    """返回拒绝理由;通过返回 None。只用最可靠的数值/枚举字段。"""
    price = int(item.get("price") or 0)
    cmp_b = _cmp_budget(getattr(profile, "budget", None))
    if cmp_b and price > 0 and price > cmp_b * 1.35:
        return f"over budget (price {price} > ~{cmp_b})"
    plat = str(item.get("platform") or "").strip()
    want_plat = getattr(profile, "platform", "No preference")
    if want_plat in ("Windows", "macOS") and plat and plat != want_plat:
        return f"platform mismatch ({plat} != {want_plat})"
    return None


# --------------------------------------------------------------------------- #
# 规则版品类 + must-have 判定(unknown 容错)
# --------------------------------------------------------------------------- #
def _rule_category_ok(profile, hay_tokens: set[str]) -> tuple[bool, str]:
    cat = (getattr(profile, "category", "") or "").strip()
    if not cat or not cat.isascii():
        return True, ""  # 无明确英文品类 → 无从判,unknown 容错保留
    cat_toks = _tokens(cat)
    if not cat_toks:
        return True, ""
    if cat_toks & hay_tokens:
        return True, ""
    return False, f"category mismatch (want {cat})"


def _rule_musthave_ok(profile, hay_tokens: set[str], hay: str) -> tuple[bool, str]:
    # 只对"可 token 化的英文 must-have"做判定;中文/预算类交给规则粗筛或 LLM
    for h in getattr(profile, "hard", []) or []:
        h = str(h)
        if h.lower().startswith("budget") or "ram" in h.lower():
            continue  # 预算/内存等由粗筛或其它字段负责
        toks = _tokens(h)
        toks = {t for t in toks if t.isascii()}
        if not toks:
            continue  # 非英文/无有效 token → unknown 容错,不拒
        if not (toks & hay_tokens):
            return False, f"missing must-have ({h})"
    return True, ""


# --------------------------------------------------------------------------- #
# LLM-as-Judge 批量判定
# --------------------------------------------------------------------------- #
def _llm_judge_batch(profile, items: list[dict], unknown_tolerant: bool) -> dict[str, dict]:
    """返回 id -> {keep: bool, reason: str}。失败返回空 dict(调用方回退规则)。"""
    if not qwen_configured() or not items:
        return {}
    cat = (getattr(profile, "category", "") or "").strip()
    musts = [str(x) for x in (getattr(profile, "hard", []) or [])
             if not str(x).lower().startswith("budget")]
    cand_payload = []
    for it in items:
        cand_payload.append({
            "id": str(it.get("id") or ""),
            "name": str(it.get("name") or "")[:120],
            "info": (str(it.get("enriched_text") or "") or str(it.get("summary") or ""))[:220],
        })
    policy = ("If an attribute cannot be determined from the product info, mark it "
              "\"unknown\" and DO NOT reject on that ground." if unknown_tolerant
              else "If an attribute cannot be determined, treat it as NOT satisfied.")
    system = (
        "You are the Verifier/Reject agent of a shopping search system. For EACH "
        "candidate product decide whether it should be KEPT for recommendation, based "
        "on two hard constraints: (1) same product category as the user wants; "
        "(2) all user must-haves are satisfied. " + policy + " "
        "Return ONLY a JSON object: {\"results\":[{\"id\":\"..\",\"category_match\":"
        "\"yes|no|unknown\",\"must_haves_met\":\"yes|no|unknown\",\"keep\":true|false,"
        "\"reason\":\"short\"}]}"
    )
    user = json.dumps({
        "want_category": cat,
        "must_haves": musts,
        "candidates": cand_payload,
    }, ensure_ascii=False)
    try:
        data = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
        )
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for r in (data.get("results") or []):
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "")
        if not rid:
            continue
        cat_m = str(r.get("category_match") or "unknown").lower()
        must_m = str(r.get("must_haves_met") or "unknown").lower()
        # 依据策略决定 keep(以模型的 keep 为辅,以显式 no 为主)
        reject = (cat_m == "no") or (must_m == "no")
        if not unknown_tolerant:
            reject = reject or (cat_m == "unknown") or (must_m == "unknown")
        keep = (not reject)
        reason = str(r.get("reason") or "")[:80]
        out[rid] = {"keep": keep, "reason": reason or ("category/must-have not satisfied" if not keep else "")}
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def verify_candidates(profile, candidates: list[dict],
                      config: ExpConfig | None = None) -> VerifyResult:
    cfg = config or CONFIG
    if not cfg.verifier_enabled or not candidates:
        return VerifyResult(kept=list(candidates), rejected=[], method="off")

    kept: list[dict] = []
    rejected: list[dict] = []

    # 1) 规则粗筛(预算/平台)—— 所有开启模式共用
    survivors: list[dict] = []
    for it in candidates:
        reason = _rule_prefilter(profile, it)
        if reason:
            rejected.append({"id": str(it.get("id") or ""),
                             "name": str(it.get("name") or ""), "reason": reason})
        else:
            survivors.append(it)

    # 2) 品类 + must-have 判定
    if cfg.verifier_uses_llm:
        head = survivors[:_MAX_LLM_CANDIDATES]
        tail = survivors[_MAX_LLM_CANDIDATES:]
        verdicts = _llm_judge_batch(profile, head, cfg.verifier_unknown_tolerant)
        method = "llm" if verdicts else "rule(llm_fallback)"
        for it in head:
            rid = str(it.get("id") or "")
            v = verdicts.get(rid)
            if v is None:
                # LLM 未覆盖该条(或整体失败)→ 回退规则判定
                _rule_decide(profile, it, kept, rejected, cfg)
            elif v["keep"]:
                it["_verify"] = {"method": "llm", "reason": v.get("reason") or "ok"}
                kept.append(it)
            else:
                rejected.append({"id": rid, "name": str(it.get("name") or ""),
                                 "reason": v.get("reason") or "rejected by judge"})
        # 超出 LLM 预算的尾部用规则判定
        for it in tail:
            _rule_decide(profile, it, kept, rejected, cfg)
    else:
        method = "rule"
        for it in survivors:
            _rule_decide(profile, it, kept, rejected, cfg)

    return VerifyResult(kept=kept, rejected=rejected, method=method)


def _rule_decide(profile, item: dict, kept: list[dict], rejected: list[dict],
                 cfg: ExpConfig) -> None:
    hay = _item_haystack(item)
    hay_tokens = _tokens(hay)
    ok_cat, r1 = _rule_category_ok(profile, hay_tokens)
    if not ok_cat:
        rejected.append({"id": str(item.get("id") or ""),
                         "name": str(item.get("name") or ""), "reason": r1})
        return
    ok_must, r2 = _rule_musthave_ok(profile, hay_tokens, hay)
    if not ok_must:
        rejected.append({"id": str(item.get("id") or ""),
                         "name": str(item.get("name") or ""), "reason": r2})
        return
    item["_verify"] = {"method": "rule", "reason": "ok"}
    kept.append(item)
