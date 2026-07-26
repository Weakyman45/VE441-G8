"""Plan / PlanHints contracts and deterministic DAG compilation.

Topology is always built in code from RoutePlan + intent short-circuit flags.
LLM may only produce PlanHints (query rewrite, focus, relax ops, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..modality_router import RoutePlan

AGENT_TEXT_RECALL = "text_recall"
AGENT_VISUAL_RECALL = "visual_recall"
AGENT_VERIFY = "verify"
AGENT_RECOMMEND = "recommend"
AGENT_RERANK_EXISTING = "rerank_existing"

VALID_AGENTS = frozenset(
    {
        AGENT_TEXT_RECALL,
        AGENT_VISUAL_RECALL,
        AGENT_VERIFY,
        AGENT_RECOMMEND,
        AGENT_RERANK_EXISTING,
    }
)


@dataclass
class PlanHints:
    query_hint: str = ""
    focus: list[str] = field(default_factory=list)
    fuse_weights: dict[str, float] = field(default_factory=dict)
    relax_ops: list[str] = field(default_factory=list)
    intent: str = ""
    memory_summary_delta: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlanHints":
        data = data or {}
        fuse = data.get("fuse_weights") or {}
        if not isinstance(fuse, dict):
            fuse = {}
        focus = data.get("focus") or []
        if not isinstance(focus, list):
            focus = []
        relax = data.get("relax_ops") or []
        if not isinstance(relax, list):
            relax = []
        return cls(
            query_hint=str(data.get("query_hint") or "").strip(),
            focus=[str(x) for x in focus[:12]],
            fuse_weights={str(k): float(v) for k, v in fuse.items() if _is_number(v)},
            relax_ops=[str(x) for x in relax[:8]],
            intent=str(data.get("intent") or "").strip(),
            memory_summary_delta=str(data.get("memory_summary_delta") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
        )


@dataclass
class Stage:
    agent: str
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "depends_on": list(self.depends_on),
            "params": dict(self.params),
        }


@dataclass
class Plan:
    plan_id: str
    stages: list[Stage] = field(default_factory=list)
    loop: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    writeback: dict[str, Any] = field(default_factory=dict)
    hints: PlanHints = field(default_factory=PlanHints)
    refused: bool = False
    message: str = ""
    planner: str = "rules"  # llm | rules | light
    # Backward-compatible fields consumed by older traces / UIs
    query_hint: str = ""
    reason: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "stages": [s.to_dict() for s in self.stages],
            "loop": dict(self.loop),
            "decision": dict(self.decision),
            "writeback": dict(self.writeback),
            "hints": self.hints.to_dict(),
            "refused": self.refused,
            "message": self.message,
            "planner": self.planner,
            "query_hint": self.query_hint or self.hints.query_hint,
            "reason": self.reason or self.hints.reason,
            "tasks": self.tasks
            or [{"name": s.agent, "depends_on": s.depends_on} for s in self.stages],
            "focus": list(self.hints.focus),
        }


def build_dag(
    *,
    route_plan: RoutePlan | None,
    shortcircuit: bool,
    do_verify: bool,
    focus_product_ids: list[str] | None = None,
    query_hint: str = "",
    visual_top_k: int | None = None,
) -> list[Stage]:
    """Compile a legal stage DAG. Never trusts LLM for topology."""
    params_common: dict[str, Any] = {}
    if query_hint:
        params_common["query_hint"] = query_hint
    if focus_product_ids:
        params_common["focus_product_ids"] = list(focus_product_ids)
    if visual_top_k is not None:
        params_common["visual_top_k"] = int(visual_top_k)

    if shortcircuit:
        return [
            Stage(
                agent=AGENT_RERANK_EXISTING,
                depends_on=[],
                params=dict(params_common),
            )
        ]

    rp = route_plan
    stages: list[Stage] = []
    recall_names: list[str] = []

    if rp is None or rp.do_text_recall:
        stages.append(
            Stage(agent=AGENT_TEXT_RECALL, depends_on=[], params=dict(params_common))
        )
        recall_names.append(AGENT_TEXT_RECALL)

    if rp is not None and rp.do_visual_recall:
        vparams = dict(params_common)
        vparams["infer_category_from_image"] = bool(rp.infer_category_from_image)
        stages.append(
            Stage(agent=AGENT_VISUAL_RECALL, depends_on=[], params=vparams)
        )
        recall_names.append(AGENT_VISUAL_RECALL)

    if not recall_names:
        # Safety net: never emit an empty retrieval plan.
        stages.append(
            Stage(agent=AGENT_TEXT_RECALL, depends_on=[], params=dict(params_common))
        )
        recall_names = [AGENT_TEXT_RECALL]

    prev = recall_names
    verify_enabled = do_verify if rp is None else bool(rp.do_verify and do_verify)
    if verify_enabled:
        vparams = {
            "reverse_verify": bool(rp.reverse_verify) if rp else False,
        }
        stages.append(
            Stage(agent=AGENT_VERIFY, depends_on=list(prev), params=vparams)
        )
        prev = [AGENT_VERIFY]

    stages.append(
        Stage(agent=AGENT_RECOMMEND, depends_on=list(prev), params=dict(params_common))
    )
    return stages


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
