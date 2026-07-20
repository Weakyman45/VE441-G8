"""Session Engine domain events (process-local EventBus).

These are NOT OpenAI Realtime wire types. Realtime frames stay in
`engine/realtime_map.py` / TalkerBridge; domain payloads may include
`source: { protocol: "openai.realtime", type, event_id, ... }`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class EventType:
    USER_UTTERANCE = "user.utterance"
    USER_INTENT_UPDATED = "user.intent_updated"
    WORKER_PLAN_CREATED = "worker.plan_created"
    WORKER_CANDIDATES_READY = "worker.candidates_ready"
    WORKER_RECOMMENDATION_READY = "worker.recommendation_ready"
    WORKER_STATUS = "worker.status"
    SESSION_INTERRUPTED = "session.interrupted"


@dataclass
class Event:
    type: str
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
