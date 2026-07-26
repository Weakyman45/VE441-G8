from __future__ import annotations

import threading
import uuid
from typing import Any

from .models import PreferenceProfile, RecommendationBundle, SessionState


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str | None = None) -> SessionState:
        sid = session_id or uuid.uuid4().hex[:16]
        with self._lock:
            state = SessionState(session_id=sid)
            self._sessions[sid] = state
            return state

    def create_or_get(self, session_id: str | None = None) -> SessionState:
        sid = session_id or uuid.uuid4().hex[:16]
        with self._lock:
            state = self._sessions.get(sid)
            if state is None:
                state = SessionState(session_id=sid)
                self._sessions[sid] = state
            return state

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> SessionState:
        state = self.get(session_id)
        if state is None:
            return self.create(session_id)
        return state

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        with self._lock:
            state = self.require(session_id)
            state.conversation.append({"role": role, "text": text})
            if len(state.conversation) > 40:
                state.conversation = state.conversation[-40:]

    def update_preference(self, session_id: str, preference: PreferenceProfile) -> PreferenceProfile:
        with self._lock:
            state = self.require(session_id)
            state.preference = preference
            return state.preference

    def set_worker_status(self, session_id: str, status: str, message: str = "", plan_id: str = "") -> None:
        with self._lock:
            state = self.require(session_id)
            state.worker.status = status
            state.worker.message = message
            if plan_id:
                state.worker.plan_id = plan_id

    def set_bundle(self, session_id: str, bundle: RecommendationBundle) -> None:
        with self._lock:
            state = self.require(session_id)
            state.worker.last_bundle = bundle
            state.worker.status = "ready"
            state.worker.plan_id = bundle.plan_id
            state.worker.message = bundle.summary

    def add_image_ref(self, session_id: str, image_ref: dict[str, str]) -> None:
        with self._lock:
            state = self.require(session_id)
            state.image_refs.append(image_ref)
            if len(state.image_refs) > 20:
                state.image_refs = state.image_refs[-20:]

    def update_risk(self, session_id: str, category: str, reasons: list[str]) -> None:
        with self._lock:
            state = self.require(session_id)
            state.risk_category = category
            state.risk_reasons = reasons[-5:]

    def snapshot(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self.get(session_id)
            return state.to_dict() if state else None
