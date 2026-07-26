from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PreferenceProfile:
    category: str = ""
    budget: int | None = None
    use_case: str = ""
    platform: str = "No preference"  # No preference | Windows | macOS
    touch: str = "Not required"
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    visual_context: str = ""
    raw_query: str = ""
    # English keywords for matching the (English) product catalog. Filled by the
    # LLM brief so a Chinese/voice request still yields catalog-searchable terms.
    search_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RankedProduct:
    id: str
    name: str
    price: int
    score: int
    reasons: list[str]
    summary: str = ""
    rating: float = 0.0
    platform: str = "Windows"
    display: str = ""
    performance: str = ""
    weight_kg: float = 0.0
    image_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationBundle:
    plan_id: str
    ranked: list[RankedProduct] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    status: str = "ready"
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    tradeoffs: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    talker_brief: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "ranked": [r.to_dict() for r in self.ranked],
            "excluded": self.excluded,
            "summary": self.summary,
            "status": self.status,
            "conflicts": self.conflicts,
            "tradeoffs": self.tradeoffs,
            "open_questions": self.open_questions,
            "talker_brief": self.talker_brief,
        }


@dataclass
class WorkerState:
    plan_id: str = ""
    status: str = "idle"  # idle | planning | searching | recommending | ready | cancelled
    last_bundle: RecommendationBundle | None = None
    message: str = ""
    # Monotonic id for the latest pipeline run; Talker drops stale recommendation injects.
    run_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status,
            "message": self.message,
            "run_id": self.run_id,
            "last_bundle": self.last_bundle.to_dict() if self.last_bundle else None,
        }


@dataclass
class SessionState:
    session_id: str
    conversation: list[dict[str, str]] = field(default_factory=list)
    preference: PreferenceProfile = field(default_factory=PreferenceProfile)
    worker: WorkerState = field(default_factory=WorkerState)
    image_refs: list[dict[str, str]] = field(default_factory=list)
    memory_summary: str = ""
    risk_category: str = "normal"
    risk_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation": self.conversation[-20:],
            "preference": self.preference.to_dict(),
            "worker": self.worker.to_dict(),
            "image_refs": self.image_refs[-10:],
            "memory_summary": self.memory_summary,
            "risk_category": self.risk_category,
            "risk_reasons": self.risk_reasons[-5:],
        }
