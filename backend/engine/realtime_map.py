"""Map OpenAI Realtime wire events ↔ Session Engine domain events.

Realtime frames keep official `type` names and JSON shapes.
Domain EventBus uses `user.*` / `worker.*` / `session.*` and may attach a
compact `source` envelope pointing at the originating Realtime event.
"""

from __future__ import annotations

from typing import Any

# Official Realtime event types we consume (server → client unless noted).
RT_SPEECH_STARTED = "input_audio_buffer.speech_started"
RT_INPUT_TRANSCRIPT_DONE = "conversation.item.input_audio_transcription.completed"
RT_CLIENT_ITEM_CREATE = "conversation.item.create"  # client → server
RT_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
RT_OUTPUT_TEXT_DONE = "response.output_text.done"
RT_RESPONSE_CREATE = "response.create"

# Domain EventType values (see events.py) ← Realtime triggers
DOMAIN_FROM_REALTIME: dict[str, str] = {
    RT_SPEECH_STARTED: "session.interrupted",
    RT_INPUT_TRANSCRIPT_DONE: "user.utterance",
    RT_CLIENT_ITEM_CREATE: "user.utterance",  # when content has input_text
}


def realtime_source(
    frame: dict[str, Any],
    *,
    protocol: str = "openai.realtime",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact provenance block for domain Event.payload['source']."""
    src: dict[str, Any] = {
        "protocol": protocol,
        "type": frame.get("type"),
    }
    for key in ("event_id", "item_id", "response_id", "previous_item_id"):
        if frame.get(key):
            src[key] = frame[key]
    if extra:
        src.update(extra)
    return src
