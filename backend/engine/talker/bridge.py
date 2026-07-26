from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

from ..bus import EventBus
from ..events import Event, EventType
from ..intent import classify_preference_change, extract_preference
from ..logging_store import LoggingStore
from ..realtime_map import (
    RT_INPUT_TRANSCRIPT_DONE,
    RT_OUTPUT_AUDIO_TRANSCRIPT_DONE,
    RT_OUTPUT_TEXT_DONE,
    RT_SPEECH_STARTED,
    realtime_source,
)
from ..session import SessionStore
from .shopping_safety import (
    RiskAssessment,
    assess_shopping_risk,
    risk_context_message,
    shopping_safety_enabled,
    shopping_safety_log_trace_enabled,
    shopping_safety_talker_hint_enabled,
)

SendUpstream = Callable[[dict], None]

INTERRUPTED_MARKER = "[INTERRUPTED]"
UNSPOKEN_MARKER = "[UNSPOKEN]"


def interruption_handling_enabled() -> bool:
    raw = os.environ.get("INTERRUPTION_HANDLING_ENABLED", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def interrupted_assistant_history_mode() -> str:
    raw = os.environ.get("INTERRUPTED_ASSISTANT_HISTORY_MODE", "truncate")
    mode = raw.strip().lower().replace("-", "_")
    if mode in ("retain", "keep", "keep_unspoken", "retain_unspoken", "preserve_unspoken"):
        return "retain_unspoken"
    return "truncate"


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
        self._assistant_buffer = ""
        self._current_assistant_item_id: str | None = None
        self._truncated_assistant_item_ids: set[str] = set()
        self._ignore_next_assistant_done = False
        self._inject_timer: threading.Timer | None = None
        self._assistant_speaking = False
        self._pending_worker_speak: str | None = None
        self._pending_run_id: int | None = None
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
        elif et == "conversation.item.truncate":
            item_id = frame.get("item_id")
            if interruption_handling_enabled():
                self._record_truncated_assistant(
                    item_id=item_id,
                    audio_end_ms=frame.get("audio_end_ms"),
                    source=realtime_source(frame, protocol=self.protocol),
                )
            elif item_id:
                self._truncated_assistant_item_ids.add(str(item_id))
                self._ignore_next_assistant_done = True
                self._assistant_buffer = ""

    def on_upstream_text(self, text: str) -> None:
        """Frames from OpenAI toward the app (official Realtime server events)."""
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        et = frame.get("type")
        if et == RT_SPEECH_STARTED:
            self._last_interrupt_at = time.time()
            self._pending_worker_speak = None
            self._pending_run_id = None
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
        if et == "response.created":
            self._assistant_speaking = True
            return
        if et in (
            "response.done",
            "response.cancelled",
            "response.failed",
            "response.completed",
        ):
            self._assistant_speaking = False
            self._schedule_pending_worker_speak(delay_s=0.15)
            return
        if et in ("response.output_item.added", "response.output_item.created"):
            item = frame.get("item") or {}
            item_id = item.get("id") or frame.get("item_id")
            if item_id:
                self._current_assistant_item_id = str(item_id)
                self._assistant_buffer = ""
            return
        if et == RT_INPUT_TRANSCRIPT_DONE:
            transcript = (frame.get("transcript") or frame.get("text") or "").strip()
            if transcript:
                self._handle_user_text(
                    transcript,
                    source=realtime_source(frame, protocol=self.protocol),
                )
            return
        if et in (
            "response.output_text.delta",
            "response.text.delta",
            "response.output_audio_transcript.delta",
            "response.audio_transcript.delta",
        ):
            item_id = frame.get("item_id") or self._current_assistant_item_id
            if item_id and str(item_id) in self._truncated_assistant_item_ids:
                return
            delta = frame.get("delta") or ""
            if delta:
                self._assistant_buffer += str(delta)
            return
        # Qwen Omni also emits response.audio_transcript.done
        if et in (
            RT_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            RT_OUTPUT_TEXT_DONE,
            "response.audio_transcript.done",
            "response.text.done",
        ):
            item_id = frame.get("item_id") or self._current_assistant_item_id
            if item_id and str(item_id) in self._truncated_assistant_item_ids:
                self._assistant_buffer = ""
                self._current_assistant_item_id = None
                return
            if self._ignore_next_assistant_done:
                self._ignore_next_assistant_done = False
                self._assistant_buffer = ""
                self._current_assistant_item_id = None
                return
            spoken = (frame.get("transcript") or frame.get("text") or "").strip()
            if spoken:
                self.sessions.append_turn(self.session_id, "assistant", spoken)
                self.logs.log_conversation(self.session_id, "assistant", spoken)
            self._assistant_buffer = ""
            self._current_assistant_item_id = None

    def record_local_barge_in_cancel(self, *, source: dict[str, Any] | None = None) -> None:
        """Mark the current assistant response as interrupted by local VAD.

        Local VAD can cancel a response before an upstream
        conversation.item.truncate event exists. This keeps local conversation
        history aligned with the spoken barge-in path.
        """
        if not interruption_handling_enabled():
            return
        item_id = self._current_assistant_item_id
        if item_id and item_id in self._truncated_assistant_item_ids:
            return
        if not item_id and not self._assistant_buffer.strip():
            return
        self._record_truncated_assistant(
            item_id=item_id,
            audio_end_ms=None,
            source=source or {"type": "local_vad_barge_in"},
        )

    def _handle_user_text(self, text: str, *, source: dict[str, Any] | None = None) -> None:
        text = text.strip()
        if len(text) < 2:
            return
        state = self.sessions.require(self.session_id)
        self.sessions.append_turn(self.session_id, "user", text)
        self.logs.log_conversation(self.session_id, "user", text)
        prior = state.preference
        preference = extract_preference(text, prior)
        change_kind = classify_preference_change(prior, preference)
        self.sessions.update_preference(self.session_id, preference)
        risk = assess_shopping_risk(text) if shopping_safety_enabled() else RiskAssessment()
        self.sessions.update_risk(self.session_id, risk.category, risk.reasons)
        self.logs.log_trace(
            self.session_id,
            "talker",
            "intent_updated",
            {
                **preference.to_dict(),
                "change_kind": change_kind,
                "risk_category": risk.category,
                "risk_reasons": risk.reasons,
                **({"source": source} if source else {}),
            },
        )
        if shopping_safety_log_trace_enabled():
            self.logs.log_trace(
                self.session_id,
                "talker",
                "risk_assessed",
                {
                    **risk.to_dict(),
                    "text": text[:500],
                    **({"source": source} if source else {}),
                },
            )
        if shopping_safety_talker_hint_enabled():
            self._inject_safety_context(risk_context_message(risk))
        utterance_payload: dict[str, Any] = {"text": text}
        if source:
            utterance_payload["source"] = source
        intent_payload: dict[str, Any] = preference.to_dict()
        if source:
            intent_payload["source"] = source
        intent_payload["risk_category"] = risk.category
        intent_payload["risk_reasons"] = risk.reasons
        intent_payload["change_kind"] = change_kind
        intent_payload["raw_query"] = text
        intent_payload["utterance"] = text
        self.bus.emit(
            Event(
                type=EventType.USER_UTTERANCE,
                session_id=self.session_id,
                payload=utterance_payload,
            )
        )
        # Soft-only followups with no material change still refresh conversation,
        # but skip kicking a new Worker run (avoids cancel storms while Talker chats).
        if change_kind == "none" and state.worker.last_bundle:
            self.logs.log_trace(
                self.session_id,
                "talker",
                "intent_skipped_worker",
                {"change_kind": change_kind, "reason": "no_material_preference_change"},
            )
            return
        self.bus.emit(
            Event(
                type=EventType.USER_INTENT_UPDATED,
                session_id=self.session_id,
                payload=intent_payload,
            )
        )

    def _inject_safety_context(self, text: str) -> None:
        send = self._send_upstream
        if not send:
            return
        send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def _record_truncated_assistant(
        self,
        *,
        item_id: Any,
        audio_end_ms: Any,
        source: dict[str, Any] | None = None,
    ) -> None:
        text = self._interrupted_history_text(audio_end_ms).strip()
        if not text:
            text = INTERRUPTED_MARKER
        elif INTERRUPTED_MARKER not in text:
            text = f"{text} {INTERRUPTED_MARKER}"
        if item_id:
            self._truncated_assistant_item_ids.add(str(item_id))
        self._ignore_next_assistant_done = True
        self.sessions.append_turn(self.session_id, "assistant", text)
        self.logs.log_conversation(self.session_id, "assistant", text)
        self.logs.log_trace(
            self.session_id,
            "talker",
            "assistant_truncated",
            {
                "item_id": item_id,
                "audio_end_ms": audio_end_ms,
                "marker": INTERRUPTED_MARKER,
                "history_mode": interrupted_assistant_history_mode(),
                **({"source": source} if source else {}),
            },
        )
        self._assistant_buffer = ""

    def _interrupted_history_text(self, audio_end_ms: Any) -> str:
        if interrupted_assistant_history_mode() == "retain_unspoken":
            heard, unspoken = self._split_heard_unspoken_estimate(audio_end_ms)
            if heard and unspoken:
                return f"{heard} {INTERRUPTED_MARKER} {UNSPOKEN_MARKER} {unspoken}"
            if unspoken:
                return f"{INTERRUPTED_MARKER} {UNSPOKEN_MARKER} {unspoken}"
            return heard
        return self._heard_text_estimate(audio_end_ms)

    def _split_heard_unspoken_estimate(self, audio_end_ms: Any) -> tuple[str, str]:
        text = self._assistant_buffer.strip()
        if not text:
            return "", ""
        try:
            played_ms = max(0, int(audio_end_ms))
        except (TypeError, ValueError):
            return "", text
        if played_ms <= 0:
            return "", text

        words = text.split()
        if len(words) <= 1:
            return text, ""
        heard_words = max(1, round(played_ms / 375))
        split_at = min(heard_words, len(words))
        return " ".join(words[:split_at]), " ".join(words[split_at:])

    def _heard_text_estimate(self, audio_end_ms: Any) -> str:
        text = self._assistant_buffer.strip()
        if not text:
            return ""
        try:
            played_ms = max(0, int(audio_end_ms))
        except (TypeError, ValueError):
            return ""
        if played_ms <= 0:
            return ""

        words = text.split()
        if len(words) <= 1:
            return text
        # Approximate spoken English at about 160 words/minute. Realtime's
        # upstream transcript can arrive ahead of audio playback, so keep local
        # logs aligned with the audio_end_ms truncate point as closely as we can.
        heard_words = max(1, round(played_ms / 375))
        return " ".join(words[: min(heard_words, len(words))])

    def _on_recommendation(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        bundle = event.payload or {}
        run_id = bundle.get("run_id")
        state = self.sessions.require(self.session_id)
        if run_id is not None and int(run_id) != int(state.worker.run_id):
            self.logs.log_trace(
                self.session_id,
                "talker",
                "stale_recommendation_dropped",
                {"run_id": run_id, "current_run_id": state.worker.run_id},
            )
            return
        # Prefer structured Talker brief (recommendation + conflicts/tradeoffs).
        spoken = (bundle.get("talker_brief") or "").strip()
        if not spoken:
            summary = bundle.get("summary") or "I updated your matches."
            ranked = bundle.get("ranked") or []
            lines = [summary]
            for i, item in enumerate(ranked[:3], 1):
                lines.append(
                    f"{i}. {item.get('name')} — score {item.get('score')}, "
                    f"about {item.get('price')}."
                )
            conflicts = bundle.get("conflicts") or []
            violated = [
                c for c in conflicts
                if isinstance(c, dict) and c.get("status") == "violated"
            ][:2]
            if violated:
                bits = [
                    f"{c.get('product_name') or 'one option'}: {c.get('constraint')}"
                    for c in violated
                ]
                lines.append("I filtered some options — " + "; ".join(bits) + ".")
            questions = bundle.get("open_questions") or []
            if questions:
                lines.append(str(questions[0]))
            spoken = " ".join(lines)
        self.logs.log_trace(
            self.session_id,
            "talker",
            "recommendation_brief",
            {
                "run_id": run_id,
                "conflicts": len(bundle.get("conflicts") or []),
                "open_questions": len(bundle.get("open_questions") or []),
                "brief": spoken[:300],
            },
        )
        self._pending_worker_speak = spoken
        self._pending_run_id = int(run_id) if run_id is not None else state.worker.run_id
        # Wait until the Talker finishes its current utterance so Worker notes
        # do not response.cancel mid-sentence (that sounded choppy on web/Android).
        delay = 0.2 if not self._assistant_speaking else 0.05
        self._schedule_pending_worker_speak(delay_s=delay)

    def _schedule_pending_worker_speak(self, *, delay_s: float) -> None:
        if self._inject_timer is not None:
            self._inject_timer.cancel()
            self._inject_timer = None
        if not self._pending_worker_speak:
            return
        timer = threading.Timer(delay_s, self._flush_pending_worker_speak)
        timer.daemon = True
        self._inject_timer = timer
        timer.start()

    def _flush_pending_worker_speak(self) -> None:
        spoken = self._pending_worker_speak
        run_id = self._pending_run_id
        if not spoken:
            return
        if self._assistant_speaking:
            # Still talking — retry shortly instead of cancelling mid-utterance.
            self._schedule_pending_worker_speak(delay_s=0.35)
            return
        if time.time() - self._last_interrupt_at < 2.0:
            self._schedule_pending_worker_speak(delay_s=0.5)
            return
        state = self.sessions.require(self.session_id)
        if run_id is not None and int(run_id) != int(state.worker.run_id):
            self._pending_worker_speak = None
            self._pending_run_id = None
            return
        self._pending_worker_speak = None
        self._pending_run_id = None
        self._inject_assistant_speak(spoken)

    def _inject_assistant_speak(self, spoken: str) -> None:
        send = self._send_upstream
        if not send:
            return
        # Skip if the user barged in recently — let them finish their turn.
        if time.time() - self._last_interrupt_at < 2.0:
            return
        # Only cancel if a response somehow started again; prefer idle inject.
        if self._assistant_speaking:
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
                                "[Worker recommendation ready — speak this naturally in one short turn. "
                                "Include any constraint conflicts, trade-offs, or clarifying questions. "
                                "Do not invent other products]\n" + spoken
                            ),
                        }
                    ],
                },
            }
        )
        send({"type": "response.create"})
        self._assistant_speaking = True
