from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def half_duplex_mic_gate_requested() -> bool:
    return env_bool("HALF_DUPLEX_MIC_GATE", False)


class ExperimentEventLogger:
    """Small JSONL logger for duplex experiments.

    Business logs stay in SQLite. Experiment traces are append-only JSONL so
    metric scripts can scan them without coupling to the live app schema.
    """

    def __init__(self, log_dir: str | os.PathLike[str] | None) -> None:
        self.log_dir = Path(log_dir).expanduser() if log_dir else None
        self._lock = threading.Lock()
        self._path: Path | None = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.log_dir / "events.jsonl"

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def log(
        self,
        *,
        session_id: str,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return
        entry = {
            "ts": time.time(),
            "session_id": session_id,
            "action": action,
            "detail": detail or {},
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_frame(self, *, session_id: str, direction: str, text: str) -> None:
        if self._path is None:
            return
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            self.log(
                session_id=session_id,
                action="realtime_frame",
                detail={"direction": direction, "type": "invalid_json"},
            )
            return
        summary = summarize_realtime_frame(frame)
        summary["direction"] = direction
        self.log(session_id=session_id, action="realtime_frame", detail=summary)


class HalfDuplexMicGate:
    """Drop user mic/control frames while the assistant is responding."""

    def __init__(
        self,
        *,
        session_id: str,
        input_sample_rate: int,
        logger: ExperimentEventLogger | None = None,
    ) -> None:
        self.session_id = session_id
        self.input_sample_rate = input_sample_rate
        self.logger = logger
        self._assistant_active = False
        self._dropped_audio_bytes = 0
        self._dropped_chunks = 0

    @property
    def assistant_active(self) -> bool:
        return self._assistant_active

    def process_client_text(self, text: str) -> list[str] | None:
        """Return [] to drop, None to forward unchanged."""
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return None

        frame_type = frame.get("type")
        if frame_type in ("response.cancel", "conversation.item.truncate"):
            self._log_drop(frame_type, reason="barge_in_disabled")
            return []

        if (
            self._assistant_active
            and frame_type == "conversation.item.create"
            and (frame.get("item") or {}).get("role") == "user"
        ):
            self._log_drop(frame_type, reason="assistant_speaking_user_text")
            return []

        if self._assistant_active and frame_type == "response.create":
            self._log_drop(frame_type, reason="assistant_speaking_response_create")
            return []

        if frame_type != "input_audio_buffer.append" or not self._assistant_active:
            return None

        audio_bytes = _decoded_audio_bytes(frame.get("audio"))
        self._dropped_audio_bytes += audio_bytes
        self._dropped_chunks += 1
        self._log_drop(
            frame_type,
            reason="assistant_speaking",
            audio_bytes=audio_bytes,
            audio_ms=_pcm16_duration_ms(audio_bytes, self.input_sample_rate),
        )
        return []

    def process_upstream_text(self, text: str) -> None:
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        frame_type = frame.get("type")
        if frame_type == "response.created":
            self._assistant_active = True
            self._log_state("assistant_active")
        elif frame_type in (
            "response.done",
            "response.cancelled",
            "response.failed",
            "response.completed",
        ):
            if self._assistant_active:
                self._log_state(
                    "assistant_inactive",
                    dropped_audio_bytes=self._dropped_audio_bytes,
                    dropped_chunks=self._dropped_chunks,
                    dropped_audio_ms=_pcm16_duration_ms(
                        self._dropped_audio_bytes,
                        self.input_sample_rate,
                    ),
                )
            self._assistant_active = False
            self._dropped_audio_bytes = 0
            self._dropped_chunks = 0

    def _log_drop(self, frame_type: str, **detail: Any) -> None:
        if self.logger is None:
            return
        self.logger.log(
            session_id=self.session_id,
            action="half_duplex_drop",
            detail={"frame_type": frame_type, **detail},
        )

    def _log_state(self, state: str, **detail: Any) -> None:
        if self.logger is None:
            return
        self.logger.log(
            session_id=self.session_id,
            action="half_duplex_state",
            detail={"state": state, **detail},
        )


def summarize_realtime_frame(frame: dict[str, Any]) -> dict[str, Any]:
    frame_type = str(frame.get("type") or "")
    summary: dict[str, Any] = {"type": frame_type}
    for key in ("event_id", "item_id", "response_id"):
        if frame.get(key):
            summary[key] = frame[key]

    audio_value = frame.get("audio")
    if isinstance(audio_value, str):
        summary["audio_base64_chars"] = len(audio_value)
        summary["audio_decoded_bytes"] = _decoded_audio_bytes(audio_value)

    delta_value = frame.get("delta")
    if isinstance(delta_value, str):
        if "audio" in frame_type:
            summary["delta_base64_chars"] = len(delta_value)
            summary["delta_decoded_bytes"] = _decoded_audio_bytes(delta_value)
        elif delta_value:
            summary["delta_chars"] = len(delta_value)

    transcript = frame.get("transcript") or frame.get("text")
    if isinstance(transcript, str) and transcript:
        summary["text_chars"] = len(transcript)
    return summary


def _decoded_audio_bytes(value: Any) -> int:
    if not isinstance(value, str) or not value:
        return 0
    try:
        return len(base64.b64decode(value, validate=False))
    except Exception:
        return 0


def _pcm16_duration_ms(num_bytes: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (num_bytes / 2.0) * 1000.0 / sample_rate
