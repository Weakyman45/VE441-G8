"""创新点三:约束感知的校验-重排序之"校验闸门"(Constraint-Aware Verifier)。

对应《创新点与实验设计.md》§3。在召回与排序之间插入独立的 Verifier / Reject
Agent,把"硬约束校验"(品类一致 + must-have 满足)与"软偏好排序"解耦:

  - 二值判定(LLM-as-Judge):对候选判 品类是否一致、must-have 是否满足;
  - 规则粗筛:预算 / 平台等可靠的结构化硬约束直接用规则剔除(便宜且确定);
  - unknown 容错:信息缺失时按"不违反"处理(保留而非硬拒),避免过度拒绝导致空结果;
  - ConflictPacket:结构化冲突,供 Talker 说明违规 / 未知 / 权衡。

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

from .conflicts import ConflictPacket, conflicts_to_rejected, packet_from_reject
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
    rejected: list[dict] = field(default_factory=list)   # legacy {id, name, reason}
    conflicts: list[ConflictPacket] = field(default_factory=list)
    unresolved: list[ConflictPacket] = field(default_factory=list)
    method: str = "off"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": len(self.kept),
            "rejected": self.rejected[:12],
            "conflicts": [c.to_dict() for c in self.conflicts[:12]],
            "unresolved": [c.to_dict() for c in self.unresolved[:8]],
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


def _evidence_snip(item: dict, field: str = "name") -> list[dict[str, Any]]:
    val = str(item.get(field) or item.get("summary") or "")[:160]
    return [{"source": field, "snippet": val, "confidence": 0.7}] if val else []


# --------------------------------------------------------------------------- #
# 规则粗筛(所有 verifier-on 模式共用:可靠的结构化硬约束)
# --------------------------------------------------------------------------- #
def _rule_prefilter(profile, item: dict) -> ConflictPacket | None:
    """返回冲突包;通过返回 None。只用最可靠的数值/枚举字段。"""
    price = int(item.get("price") or 0)
    cmp_b = _cmp_budget(getattr(profile, "budget", None))
    if cmp_b and price > 0 and price > cmp_b * 1.35:
        return ConflictPacket(
            product_id=str(item.get("id") or ""),
            product_name=str(item.get("name") or ""),
            conflict_type="budget",
            constraint=f"over budget (price {price} > ~{cmp_b})",
            status="violated",
            evidence=_evidence_snip(item, "price") or [{"source": "price", "snippet": str(price)}],
            user_action="relax_constraint",
        )
    plat = str(item.get("platform") or "").strip()
    want_plat = getattr(profile, "platform", "No preference")
    if want_plat in ("Windows", "macOS") and plat and plat != want_plat:
        return ConflictPacket(
            product_id=str(item.get("id") or ""),
            product_name=str(item.get("name") or ""),
            conflict_type="platform",
            constraint=f"platform mismatch ({plat} != {want_plat})",
            status="violated",
            evidence=[{"source": "platform", "snippet": plat, "confidence": 0.95}],
            user_action="relax_constraint",
        )
    return None


# --------------------------------------------------------------------------- #
# 规则版品类 + must-have 判定(unknown 容错)
# --------------------------------------------------------------------------- #
def _rule_category_ok(profile, hay_tokens: set[str]) -> tuple[str, str]:
    """Return (yes|no|unknown, reason)."""
    cat = (getattr(profile, "category", "") or "").strip()
    if not cat or not cat.isascii():
        return "unknown", ""
    cat_toks = _tokens(cat)
    if not cat_toks:
        return "unknown", ""
    if cat_toks & hay_tokens:
        return "yes", ""
    return "no", f"category mismatch (want {cat})"


def _rule_musthave_status(profile, hay_tokens: set[str], hay: str) -> list[tuple[str, str, str]]:
    """List of (constraint, yes|no|unknown, reason).

    Non-ASCII / non-tokenizable must-haves are skipped (legacy rule behavior):
    they cannot be judged reliably without LLM.
    """
    out: list[tuple[str, str, str]] = []
    for h in getattr(profile, "hard", []) or []:
        h = str(h)
        if h.lower().startswith("budget") or "ram" in h.lower():
            continue
        toks = _tokens(h)
        toks = {t for t in toks if t.isascii()}
        if not toks:
            continue
        if toks & hay_tokens or any(t in hay for t in toks):
            out.append((h, "yes", ""))
        else:
            out.append((h, "no", f"missing must-have ({h})"))
    return out


def _reverse_verify_packet(profile, item: dict) -> ConflictPacket | None:
    """Image→catalog reverse check: candidate must align with image-derived category."""
    cat = (getattr(profile, "category", "") or "").strip().lower()
    if not cat:
        return None
    hay = _item_haystack(item)
    cat_toks = _tokens(cat)
    if not cat_toks:
        return None
    if cat_toks & _tokens(hay):
        return None
    # Soft visual cue mismatch → unresolved unknown rather than hard reject when
    # the catalog text is sparse; still emit a packet for Talker.
    return ConflictPacket(
        product_id=str(item.get("id") or ""),
        product_name=str(item.get("name") or ""),
        conflict_type="visual_mismatch",
        constraint=f"weak visual/category alignment with photo category '{cat}'",
        status="unknown",
        evidence=_evidence_snip(item),
        user_action="clarify",
    )


# --------------------------------------------------------------------------- #
# LLM-as-Judge 批量判定
# --------------------------------------------------------------------------- #
def _llm_judge_batch(profile, items: list[dict], unknown_tolerant: bool) -> dict[str, dict]:
    """返回 id -> {keep, reason, category_match, must_haves_met}。"""
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
        reject = (cat_m == "no") or (must_m == "no")
        if not unknown_tolerant:
            reject = reject or (cat_m == "unknown") or (must_m == "unknown")
        keep = not reject
        reason = str(r.get("reason") or "")[:80]
        out[rid] = {
            "keep": keep,
            "reason": reason or ("category/must-have not satisfied" if not keep else ""),
            "category_match": cat_m,
            "must_haves_met": must_m,
        }
    return out


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def verify_candidates(
    profile,
    candidates: list[dict],
    config: ExpConfig | None = None,
    *,
    reverse_verify: bool = False,
) -> VerifyResult:
    cfg = config or CONFIG
    if not cfg.verifier_enabled or not candidates:
        return VerifyResult(kept=list(candidates), rejected=[], method="off")

    kept: list[dict] = []
    conflicts: list[ConflictPacket] = []
    unresolved: list[ConflictPacket] = []

    # 1) 规则粗筛(预算/平台)
    survivors: list[dict] = []
    for it in candidates:
        pkt = _rule_prefilter(profile, it)
        if pkt:
            conflicts.append(pkt)
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
                _rule_decide(profile, it, kept, conflicts, unresolved, cfg)
            elif v["keep"]:
                it["_verify"] = {"method": "llm", "reason": v.get("reason") or "ok"}
                kept.append(it)
                if v.get("category_match") == "unknown" or v.get("must_haves_met") == "unknown":
                    unresolved.append(
                        ConflictPacket(
                            product_id=rid,
                            product_name=str(it.get("name") or ""),
                            conflict_type="evidence_unknown",
                            constraint=v.get("reason") or "some constraints unverified",
                            status="unknown",
                            evidence=_evidence_snip(it),
                            user_action="clarify",
                        )
                    )
            else:
                conflicts.append(
                    packet_from_reject(
                        product_id=rid,
                        product_name=str(it.get("name") or ""),
                        reason=v.get("reason") or "rejected by judge",
                        evidence=_evidence_snip(it),
                    )
                )
        for it in tail:
            _rule_decide(profile, it, kept, conflicts, unresolved, cfg)
    else:
        method = "rule"
        for it in survivors:
            _rule_decide(profile, it, kept, conflicts, unresolved, cfg)

    # 3) reverse verify (image-only path): mark weak visual/category alignment
    if reverse_verify:
        still_kept: list[dict] = []
        for it in kept:
            pkt = _reverse_verify_packet(profile, it)
            if pkt is None:
                still_kept.append(it)
                continue
            # Keep candidate but surface uncertainty to Talker (unknown-tolerant).
            if cfg.verifier_unknown_tolerant:
                unresolved.append(pkt)
                still_kept.append(it)
            else:
                pkt.status = "violated"
                pkt.user_action = "keep_searching"
                conflicts.append(pkt)
        kept = still_kept

    rejected = conflicts_to_rejected(conflicts)
    return VerifyResult(
        kept=kept,
        rejected=rejected,
        conflicts=conflicts,
        unresolved=unresolved,
        method=method,
    )


def _rule_decide(
    profile,
    item: dict,
    kept: list[dict],
    conflicts: list[ConflictPacket],
    unresolved: list[ConflictPacket],
    cfg: ExpConfig,
) -> None:
    hay = _item_haystack(item)
    hay_tokens = _tokens(hay)
    cat_status, r1 = _rule_category_ok(profile, hay_tokens)
    if cat_status == "no":
        conflicts.append(
            ConflictPacket(
                product_id=str(item.get("id") or ""),
                product_name=str(item.get("name") or ""),
                conflict_type="category_mismatch",
                constraint=r1,
                status="violated",
                evidence=_evidence_snip(item),
                user_action="clarify",
            )
        )
        return
    if cat_status == "unknown" and (getattr(profile, "category", "") or "").strip():
        unresolved.append(
            ConflictPacket(
                product_id=str(item.get("id") or ""),
                product_name=str(item.get("name") or ""),
                conflict_type="evidence_unknown",
                constraint=f"could not confirm category '{profile.category}'",
                status="unknown",
                evidence=_evidence_snip(item),
                user_action="clarify",
            )
        )

    for constraint, status, reason in _rule_musthave_status(profile, hay_tokens, hay):
        if status == "no":
            conflicts.append(
                ConflictPacket(
                    product_id=str(item.get("id") or ""),
                    product_name=str(item.get("name") or ""),
                    conflict_type="must_have_violated",
                    constraint=reason or constraint,
                    status="violated",
                    evidence=_evidence_snip(item),
                    user_action="keep_searching",
                )
            )
            return
        if status == "unknown":
            unresolved.append(
                ConflictPacket(
                    product_id=str(item.get("id") or ""),
                    product_name=str(item.get("name") or ""),
                    conflict_type="evidence_unknown",
                    constraint=reason or constraint,
                    status="unknown",
                    evidence=_evidence_snip(item),
                    user_action="clarify",
                )
            )
            if not cfg.verifier_unknown_tolerant:
                conflicts.append(
                    ConflictPacket(
                        product_id=str(item.get("id") or ""),
                        product_name=str(item.get("name") or ""),
                        conflict_type="must_have_violated",
                        constraint=reason or constraint,
                        status="violated",
                        evidence=_evidence_snip(item),
                        user_action="clarify",
                    )
                )
                return

    item["_verify"] = {"method": "rule", "reason": "ok"}
    kept.append(item)
