from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from ..bus import EventBus
from ..events import Event, EventType
from ..intent import extract_preference
from ..logging_store import LoggingStore
from ..realtime_map import (
    RT_INPUT_TRANSCRIPT_DONE,
    RT_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    RT_OUTPUT_TEXT_DONE,
    RT_SPEECH_STARTED,
    realtime_source,
)
from ..session import SessionStore

SendUpstream = Callable[[dict], None]


class TalkerBridge:
    """
    Side-car on the Realtime WebSocket proxy.
    - Parses official Realtime frames → domain Events (with payload.source)
    - Emits Session events for the Worker runtime
    - Injects recommendation summaries back via Realtime item.create + response.create
    """

    def __init__(
        self,
        session_id: str,
        bus: EventBus,
        sessions: SessionStore,
        logs: LoggingStore,
        *,
        protocol: str = "openai.realtime",
    ) -> None:
        self.session_id = session_id
        self.bus = bus
        self.sessions = sessions
        self.logs = logs
        self.protocol = protocol
        self._send_upstream: SendUpstream | None = None
        self._lock = threading.Lock()
        self._last_interrupt_at = 0.0
        self._inject_timer: threading.Timer | None = None
        self.sessions.require(session_id)
        bus.subscribe(EventType.WORKER_RECOMMENDATION_READY, self._on_recommendation)

    def bind_sender(self, send_upstream: SendUpstream) -> None:
        self._send_upstream = send_upstream

    def on_client_text(self, text: str) -> None:
        """Frames from the Android app toward OpenAI (usually control / audio append)."""
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        et = frame.get("type")
        if et == "conversation.item.create":
            item = frame.get("item") or {}
            content = item.get("content") or []
            for part in content:
                if part.get("type") == "input_text" and part.get("text"):
                    self._handle_user_text(
                        part["text"],
                        source=realtime_source(
                            frame,
                            protocol=self.protocol,
                            extra={"content_type": "input_text", "item_id": item.get("id")},
                        ),
                    )

    def on_upstream_text(self, text: str) -> None:
        """Frames from OpenAI toward the app (official Realtime server events)."""
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        et = frame.get("type")
        if et == RT_SPEECH_STARTED:
            self._last_interrupt_at = time.time()
            if self._inject_timer is not None:
                self._inject_timer.cancel()
                self._inject_timer = None
            self.bus.emit(
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=self.session_id,
                    payload={
                        "reason": "barge_in",
                        "source": realtime_source(frame, protocol=self.protocol),
                    },
                )
            )
            return
        if et == RT_INPUT_TRANSCRIPT_DONE:
            transcript = (frame.get("transcript") or frame.get("text") or "").strip()
            if transcript:
                self._handle_user_text(
                    transcript,
                    source=realtime_source(frame, protocol=self.protocol),
                )
            return
        # Qwen Omni also emits response.audio_transcript.done
        if et in (
            RT_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            RT_OUTPUT_TEXT_DONE,
            "response.audio_transcript.done",
            "response.text.done",
        ):
            spoken = (frame.get("transcript") or frame.get("text") or "").strip()
            if spoken:
                self.sessions.append_turn(self.session_id, "assistant", spoken)
                self.logs.log_conversation(self.session_id, "assistant", spoken)

    def _handle_user_text(self, text: str, *, source: dict[str, Any] | None = None) -> None:
        text = text.strip()
        if len(text) < 2:
            return
        state = self.sessions.require(self.session_id)
        self.sessions.append_turn(self.session_id, "user", text)
        self.logs.log_conversation(self.session_id, "user", text)
        preference = extract_preference(text, state.preference)
        self.sessions.update_preference(self.session_id, preference)
        self.logs.log_trace(
            self.session_id,
            "talker",
            "intent_updated",
            {**preference.to_dict(), **({"source": source} if source else {})},
        )
        utterance_payload: dict[str, Any] = {"text": text}
        if source:
            utterance_payload["source"] = source
        intent_payload: dict[str, Any] = preference.to_dict()
        if source:
            intent_payload["source"] = source
        self.bus.emit(
            Event(
                type=EventType.USER_UTTERANCE,
                session_id=self.session_id,
                payload=utterance_payload,
            )
        )
        self.bus.emit(
            Event(
                type=EventType.USER_INTENT_UPDATED,
                session_id=self.session_id,
                payload=intent_payload,
            )
        )

    def _on_recommendation(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        bundle = event.payload or {}
        summary = bundle.get("summary") or "I updated your matches."
        ranked = bundle.get("ranked") or []
        lines = [summary]
        for i, item in enumerate(ranked[:3], 1):
            lines.append(
                f"{i}. {item.get('name')} — score {item.get('score')}, "
                f"about {item.get('price')}."
            )
        spoken = " ".join(lines)
        # Delay inject so we don't stomp an in-progress reply / barge-in window.
        if self._inject_timer is not None:
            self._inject_timer.cancel()
        timer = threading.Timer(2.5, lambda: self._inject_assistant_speak(spoken))
        timer.daemon = True
        self._inject_timer = timer
        timer.start()

    def _inject_assistant_speak(self, spoken: str) -> None:
        send = self._send_upstream
        if not send:
            return
        # Skip if the user barged in recently — let them finish their turn.
        if time.time() - self._last_interrupt_at < 4.0:
            return
        # Official Realtime client events
        send({"type": "response.cancel"})
        send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "[Worker recommendation ready — speak this naturally in one short turn, "
                                "do not invent other products]\n" + spoken
                            ),
                        }
                    ],
                },
            }
        )
        send({"type": "response.create"})
