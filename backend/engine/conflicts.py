"""Structured conflict packets for verify/recommend → Talker speech."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_CONFLICT_TYPES = frozenset(
    {
        "category_mismatch",
        "must_have_violated",
        "budget",
        "platform",
        "evidence_unknown",
        "soft_tradeoff",
        "source_disagree",
        "visual_mismatch",
    }
)

VALID_STATUS = frozenset({"violated", "unknown", "soft_tradeoff"})

VALID_ACTIONS = frozenset(
    {
        "relax_constraint",
        "keep_searching",
        "clarify",
        "accept_tradeoff",
    }
)


@dataclass
class ConflictPacket:
    product_id: str = ""
    product_name: str = ""
    conflict_type: str = "must_have_violated"
    constraint: str = ""
    status: str = "violated"  # violated | unknown | soft_tradeoff
    evidence: list[dict[str, Any]] = field(default_factory=list)
    user_action: str = "keep_searching"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConflictPacket":
        data = data or {}
        ctype = str(data.get("conflict_type") or "must_have_violated")
        if ctype not in VALID_CONFLICT_TYPES:
            ctype = "must_have_violated"
        status = str(data.get("status") or "violated")
        if status not in VALID_STATUS:
            status = "violated"
        action = str(data.get("user_action") or "keep_searching")
        if action not in VALID_ACTIONS:
            action = "keep_searching"
        evidence = data.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        return cls(
            product_id=str(data.get("product_id") or ""),
            product_name=str(data.get("product_name") or ""),
            conflict_type=ctype,
            constraint=str(data.get("constraint") or "")[:160],
            status=status,
            evidence=[e for e in evidence if isinstance(e, dict)][:6],
            user_action=action,
        )


def packet_from_reject(
    *,
    product_id: str,
    product_name: str,
    reason: str,
    evidence: list[dict[str, Any]] | None = None,
) -> ConflictPacket:
    """Map a legacy reject reason string into a typed ConflictPacket."""
    low = (reason or "").lower()
    if "budget" in low or "over budget" in low:
        ctype, action = "budget", "relax_constraint"
    elif "platform" in low:
        ctype, action = "platform", "relax_constraint"
    elif "category" in low:
        ctype, action = "category_mismatch", "clarify"
    elif "visual" in low:
        ctype, action = "visual_mismatch", "keep_searching"
    elif "unknown" in low or "insufficient" in low:
        ctype, action = "evidence_unknown", "clarify"
        return ConflictPacket(
            product_id=product_id,
            product_name=product_name,
            conflict_type=ctype,
            constraint=reason[:160],
            status="unknown",
            evidence=list(evidence or []),
            user_action=action,
        )
    else:
        ctype, action = "must_have_violated", "keep_searching"
    return ConflictPacket(
        product_id=product_id,
        product_name=product_name,
        conflict_type=ctype,
        constraint=reason[:160],
        status="violated",
        evidence=list(evidence or []),
        user_action=action,
    )


def build_tradeoffs(ranked: list[Any], limit: int = 2) -> list[dict[str, Any]]:
    """Cheap pairwise trade-off notes for the top products."""
    if len(ranked) < 2:
        return []
    a, b = ranked[0], ranked[1]
    axes: list[str] = []
    pa = int(getattr(a, "price", 0) or 0)
    pb = int(getattr(b, "price", 0) or 0)
    if pa and pb and pa != pb:
        axes.append("price")
    ra = float(getattr(a, "rating", 0) or 0)
    rb = float(getattr(b, "rating", 0) or 0)
    if ra and rb and abs(ra - rb) >= 0.2:
        axes.append("rating")
    if not axes:
        axes = ["overall_fit"]
    return [
        {
            "a": getattr(a, "id", ""),
            "a_name": getattr(a, "name", ""),
            "b": getattr(b, "id", ""),
            "b_name": getattr(b, "name", ""),
            "axes": axes[:limit],
        }
    ]


def build_talker_brief(
    *,
    summary: str,
    ranked: list[Any],
    conflicts: list[ConflictPacket],
    tradeoffs: list[dict[str, Any]],
    open_questions: list[str],
) -> str:
    """One short spoken script: recommendation + conflict/tradeoff + optional clarify."""
    parts: list[str] = []
    if summary:
        parts.append(summary.strip())
    elif ranked:
        top = ranked[0]
        parts.append(f"Top pick is {getattr(top, 'name', 'this product')}.")

    violated = [c for c in conflicts if c.status == "violated"][:2]
    if violated:
        bits = []
        for c in violated:
            name = c.product_name or "one option"
            bits.append(f"{name}: {c.constraint}")
        parts.append("I filtered some options — " + "; ".join(bits) + ".")

    unknowns = [c for c in conflicts if c.status == "unknown"][:2]
    if unknowns and not open_questions:
        open_questions = [
            f"I could not verify '{c.constraint}' for {c.product_name or 'a candidate'}. "
            f"Should that stay a hard requirement?"
            for c in unknowns[:1]
        ]

    if tradeoffs:
        t = tradeoffs[0]
        axes = ", ".join(t.get("axes") or ["fit"])
        parts.append(
            f"Compared with {t.get('b_name') or 'the alternative'}, "
            f"the main trade-off is {axes}."
        )

    if open_questions:
        parts.append(str(open_questions[0]).strip())

    spoken = " ".join(p for p in parts if p)
    return spoken[:900] if spoken else "I updated your matches."


def conflicts_to_rejected(conflicts: list[ConflictPacket]) -> list[dict[str, str]]:
    """Backward-compatible rejected[] rows for older clients/tests."""
    out: list[dict[str, str]] = []
    for c in conflicts:
        if c.status == "soft_tradeoff":
            continue
        out.append(
            {
                "id": c.product_id,
                "name": c.product_name,
                "reason": c.constraint or c.conflict_type,
            }
        )
    return out
