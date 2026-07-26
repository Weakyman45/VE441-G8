from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any


class LocalRealtimeVadGate:
    """Gate Realtime audio append events through the local 16 kHz streaming VAD.

    The browser and Android clients send PCM16 audio in the selected realtime
    provider's input rate. GPT Realtime uses 24 kHz, while the local VAD models
    are 16 kHz. This adapter resamples a copy down for detection, forwards only
    detected speech chunks, then resamples those speech chunks back to the
    provider input rate before sending them upstream.
    """

    def __init__(
        self,
        *,
        input_sample_rate: int,
        threshold: float = 0.5,
        two_pass_threshold: float = 0.4,
        min_silence_duration_ms: int = 500,
        min_pause_duration_ms: int = 300,
        enable_two_pass: bool = True,
        use_new_two_pass: bool = False,
    ) -> None:
        self.input_sample_rate = input_sample_rate
        self.vad_sample_rate = 16_000

        try:
            import numpy as np  # noqa: PLC0415

            from .local_vad.local_vad import (  # noqa: PLC0415
                AudioChunk,
                AudioEvent,
                AudioEventType,
                StreamingVAD,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Local VAD requires numpy and onnxruntime. Install them before enabling LOCAL_VAD_ENABLED=1."
            ) from exc

        self._np = np
        self._AudioChunk = AudioChunk
        self._AudioEvent = AudioEvent
        self._AudioEventType = AudioEventType

        model_dir = Path(__file__).with_name("local_vad")
        self.vad = StreamingVAD(
            model_path=str(model_dir / "silero_vad_16k.onnx"),
            two_pass_model_path=str(model_dir / "two_pass_barge_in.onnx"),
            threshold=float(os.environ.get("LOCAL_VAD_THRESHOLD", threshold)),
            two_pass_threshold=float(os.environ.get("LOCAL_VAD_TWO_PASS_THRESHOLD", two_pass_threshold)),
            min_silence_duration_ms=int(os.environ.get("LOCAL_VAD_MIN_SILENCE_MS", min_silence_duration_ms)),
            min_pause_duration_ms=int(os.environ.get("LOCAL_VAD_MIN_PAUSE_MS", min_pause_duration_ms)),
            enable_two_pass=_env_bool("LOCAL_VAD_TWO_PASS", enable_two_pass),
            use_new_two_pass=_env_bool("LOCAL_VAD_NEW_TWO_PASS", use_new_two_pass),
            return_raw_audio=True,
        )
        self.speech_started_count = 0
        self.speech_ended_count = 0
        self.min_commit_audio_ms = int(os.environ.get("LOCAL_VAD_MIN_COMMIT_AUDIO_MS", "100"))
        self._pending_upstream_audio_ms = 0.0
        self._response_in_progress = False
        self._current_speech_started_ms: float | None = None
        self._last_text_tail = ""
        self._delayed_commit_deadline_s: float | None = None
        self._delayed_commit_audio_ms = 0.0
        self._barge_in_cancel_pending = False
        self.incomplete_delay_enabled = _env_bool("LOCAL_VAD_INCOMPLETE_DELAY_ENABLED", True)
        self.incomplete_delay_ms = int(os.environ.get("LOCAL_VAD_INCOMPLETE_DELAY_MS", "500"))
        self.incomplete_short_utterance_ms = int(
            os.environ.get("LOCAL_VAD_INCOMPLETE_SHORT_UTTERANCE_MS", "1600")
        )
        self.incomplete_tail_pattern = re.compile(
            r"(?:\b(?:i\s+(?:want|need|would\s+like)|maybe|between|either|from|with|for|"
            r"like|under|around|about|and|or|to|than|versus|vs)\b|[,;:—-])\s*$",
            re.IGNORECASE,
        )

    def process_client_text(self, text: str) -> list[str] | None:
        """Transform one client text frame.

        Returns:
            None to keep the original frame unchanged, or a list of replacement
            JSON frames. An empty list drops the original frame.
        """
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return None
        frame_type = frame.get("type")
        if frame_type == "response.cancel":
            self._response_in_progress = False
            self._clear_delayed_commit()
            return None
        if frame_type == "response.create":
            if self._response_in_progress:
                print("[local-vad] dropped duplicate response.create while response is active")
                return []
            self._response_in_progress = True
            self._clear_delayed_commit()
            return None
        if frame_type != "input_audio_buffer.append":
            self._capture_text_tail(frame)
            ready = self._flush_delayed_commit_if_ready()
            if ready:
                return [*ready, text]
            return None
        audio_b64 = frame.get("audio")
        if not audio_b64:
            return self._flush_delayed_commit_if_ready() or []
        try:
            pcm = base64.b64decode(audio_b64)
        except Exception:  # noqa: BLE001
            return []

        vad_pcm = self._resample_pcm16_bytes(pcm, self.input_sample_rate, self.vad_sample_rate)
        vad_audio = self._np.frombuffer(vad_pcm, dtype=self._np.int16).copy()
        chunk = self._AudioChunk(data=vad_audio, raw_data=vad_pcm)

        out: list[str] = []
        for item in asyncio.run(_collect_async(self.vad.run(chunk))):
            if isinstance(item, self._AudioEvent):
                if item.type == self._AudioEventType.START:
                    if self._delayed_commit_deadline_s is not None:
                        print("[local-vad] cancelled delayed commit; speech resumed")
                        self._clear_delayed_commit()
                    if self._response_in_progress:
                        print("[local-vad] speech_started while response active; cancelling response")
                        out.append(json.dumps({"type": "response.cancel"}))
                        self._response_in_progress = False
                        self._barge_in_cancel_pending = True
                    self.speech_started_count += 1
                    self._current_speech_started_ms = item.timestamp_ms
                    self._last_text_tail = ""
                    print(f"[local-vad] speech_started t={item.timestamp_ms}ms id={item.id}")
                elif item.type == self._AudioEventType.END:
                    self.speech_ended_count += 1
                    print(f"[local-vad] speech_stopped t={item.timestamp_ms}ms id={item.id}")
                    if self._pending_upstream_audio_ms < self.min_commit_audio_ms:
                        print(
                            "[local-vad] dropped short commit "
                            f"audio_ms={self._pending_upstream_audio_ms:.2f}"
                        )
                        self._pending_upstream_audio_ms = 0.0
                    elif self._should_delay_commit(item.timestamp_ms):
                        self._delayed_commit_deadline_s = time.monotonic() + self.incomplete_delay_ms / 1000.0
                        self._delayed_commit_audio_ms = self._pending_upstream_audio_ms
                        print(
                            "[local-vad] delayed commit for possible incomplete utterance "
                            f"audio_ms={self._pending_upstream_audio_ms:.2f} "
                            f"delay_ms={self.incomplete_delay_ms}"
                        )
                    else:
                        out.extend(self._commit_frames())
                continue
            speech_pcm = item.tobytes()
            upstream_pcm = self._resample_pcm16_bytes(speech_pcm, self.vad_sample_rate, self.input_sample_rate)
            self._pending_upstream_audio_ms += _pcm16_duration_ms(upstream_pcm, self.input_sample_rate)
            out.append(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(upstream_pcm).decode("ascii"),
                    }
                )
            )
        ready = self._flush_delayed_commit_if_ready()
        if ready:
            out.extend(ready)
        return out

    def process_upstream_text(self, text: str) -> None:
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        frame_type = frame.get("type")
        if frame_type == "response.created":
            self._response_in_progress = True
        elif frame_type in (
            "response.done",
            "response.cancelled",
            "response.failed",
            "response.completed",
        ):
            self._response_in_progress = False
        elif frame_type in (
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.done",
        ):
            transcript = (frame.get("transcript") or frame.get("text") or "").strip()
            if transcript:
                self._last_text_tail = transcript[-120:]

    def consume_barge_in_cancel_pending(self) -> bool:
        pending = self._barge_in_cancel_pending
        self._barge_in_cancel_pending = False
        return pending

    def _capture_text_tail(self, frame: dict[str, Any]) -> None:
        if frame.get("type") != "conversation.item.create":
            return
        item = frame.get("item") or {}
        content = item.get("content") or []
        if not isinstance(content, list):
            return
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                value = part.get("text")
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        if texts:
            self._last_text_tail = " ".join(texts)[-120:]

    def _should_delay_commit(self, speech_end_ms: float) -> bool:
        if not self.incomplete_delay_enabled or self.incomplete_delay_ms <= 0:
            return False
        if self._last_text_tail and self.incomplete_tail_pattern.search(self._last_text_tail.strip()):
            return True
        if self._current_speech_started_ms is None:
            return False
        utterance_ms = max(0.0, float(speech_end_ms) - float(self._current_speech_started_ms))
        if utterance_ms <= 0:
            utterance_ms = self._pending_upstream_audio_ms
        return self.min_commit_audio_ms <= utterance_ms <= self.incomplete_short_utterance_ms

    def _flush_delayed_commit_if_ready(self) -> list[str] | None:
        if self._delayed_commit_deadline_s is None:
            return None
        if time.monotonic() < self._delayed_commit_deadline_s:
            return None
        if self._response_in_progress:
            print("[local-vad] dropped delayed commit because response is active")
            self._pending_upstream_audio_ms = 0.0
            self._clear_delayed_commit()
            return []
        print(
            "[local-vad] flushing delayed commit "
            f"audio_ms={self._delayed_commit_audio_ms:.2f}"
        )
        return self._commit_frames()

    def _commit_frames(self) -> list[str]:
        self._response_in_progress = True
        self._pending_upstream_audio_ms = 0.0
        self._clear_delayed_commit()
        self._current_speech_started_ms = None
        return [
            json.dumps({"type": "input_audio_buffer.commit"}),
            json.dumps({"type": "response.create"}),
        ]

    def _clear_delayed_commit(self) -> None:
        self._delayed_commit_deadline_s = None
        self._delayed_commit_audio_ms = 0.0

    def _resample_pcm16_bytes(self, pcm: bytes, from_rate: int, to_rate: int) -> bytes:
        if from_rate == to_rate or not pcm:
            return pcm
        audio = self._np.frombuffer(pcm, dtype=self._np.int16)
        if audio.size == 0:
            return b""
        if audio.size == 1:
            return audio.astype(self._np.int16).tobytes()
        new_len = max(1, int(round(audio.size * to_rate / from_rate)))
        old_x = self._np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        new_x = self._np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        resampled = self._np.interp(new_x, old_x, audio.astype(self._np.float32))
        return self._np.clip(resampled, -32768, 32767).astype(self._np.int16).tobytes()


def _pcm16_duration_ms(pcm: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (len(pcm) / 2) * 1000.0 / sample_rate


async def _collect_async(iterator: Any) -> list[Any]:
    items: list[Any] = []
    async for item in iterator:
        items.append(item)
    return items


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")
