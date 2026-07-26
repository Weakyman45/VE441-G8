from __future__ import annotations

import asyncio
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
try:
    from enum import StrEnum, auto
except ImportError:  # Python 3.9 compatibility
    from enum import Enum, auto

    class StrEnum(str, Enum):
        pass
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import numpy.typing as npt
import logging
from .vad_rt import VADRT
from .two_pass_rt import TwoPassRT
from .vad_utils import make_confidence_strategy

logger = logging.getLogger(__name__)

def generate_new_id(prefix: str = "") -> str:
    b62uid = uuid.uuid4().hex
    if prefix:
        return f"{prefix}_{b62uid}"
    return b62uid


@dataclass
class AudioChunk:
    data: npt.NDArray[np.int16] | bytes
    raw_data: bytes | None = None  # quick workaround for ASR to get the raw data
    samples_per_ms: int = 16
    speculative: bool = False
    id: str | None = None
    latency_info: dict = field(default_factory=dict)

    def tobytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        return self.data.tobytes()

    def tonumpy(self) -> npt.NDArray[np.int16]:
        if isinstance(self.data, bytes):
            return np.frombuffer(self.data, dtype=np.int16)
        return self.data

    def duration_ms(self) -> float:
        if isinstance(self.data, bytes):
            return (len(self.data) // 2) / self.samples_per_ms
        return len(self.data) / self.samples_per_ms



class AudioEventType(StrEnum):
    START = auto()
    END = auto()
    PAUSE = auto()
    RESUME = auto()
    SPEECH = auto()  # ASR returned non-empty transcript.
    NOSPEECH = auto()  # ASR returned empty transcript.
    BARGEIN = auto()  # to be deprecated


@dataclass(frozen=True)
class AudioEvent:
    type: AudioEventType
    timestamp_ms: int  # in relative to the start of the audio stream
    id: str = field(default_factory=generate_new_id)

    wall_clock_time: float = field(default_factory=time.time)
    latency_info: dict = field(default_factory=dict)

class StreamingVAD:
    """Streaming VAD, responsible for processing audio chunks and returning chunks with speech and audio events"""

    def __init__(
        self,
        model_path: str,
        two_pass_model_path: str,
        threshold: float = 0.5,
        two_pass_threshold: float = 0.4,
        neg_threshold: float | None = None,
        two_pass_window_ms: int = 100,
        two_pass_context_ms: int = 500,
        speech_pad_ms: int = 300,
        min_silence_duration_ms: int = 300,
        min_pause_duration_ms: int = 300,  # has to be smaller than min_silence_duration_ms
        enable_two_pass: bool = True,
        reset_on_long_silence_ms: int = 0,
        return_raw_audio: bool = True,
        # New two pass config
        use_new_two_pass: bool = False,
        detection_inference_interval_ms: int = 32,
        two_pass_min_pending_ms: int = 100,
        two_pass_max_pending_ms: int = 500,
        two_pass_tail_ms_count: int = 200,
        two_pass_non_silence_weight: float = 0.5,
    ):
        super().__init__()
        neg_threshold = neg_threshold or max(threshold - 0.15, 0.15)
        assert neg_threshold <= threshold, f"check failed: neg_threshold <= threshold, got {neg_threshold}, {threshold}"
        self.speech_prob_threshold = threshold
        self.two_pass_speech_prob_threshold = two_pass_threshold
        self.neg_speech_prob_threshold = neg_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_pause_duration_ms = min_pause_duration_ms
        # Baseline (configured) values. Runtime adjustments via
        # `increase_timeout_duration` are clamped so the live values can never
        # drop below these baselines. There is no upper bound.
        self._base_min_silence_duration_ms = min_silence_duration_ms
        self._base_min_pause_duration_ms = min_pause_duration_ms
        assert 0.0 < neg_threshold < threshold < 1.0, f"check failed: 0 < neg_threshold < threshold < 1.0, got {neg_threshold}, {threshold}"
        self.enable_two_pass = enable_two_pass
        self.use_new_two_pass = use_new_two_pass
        self.reset_on_long_silence_ms = reset_on_long_silence_ms
        self.return_raw_audio = return_raw_audio

        assert two_pass_window_ms > 32, "confidence_window_ms must be at least 32ms"
        assert speech_pad_ms > 0, "speech_pad_ms must be at least 0ms"
        self.confidence_window_ms = prev_multiple_of(two_pass_window_ms, 32)
        self.speech_pad_ms = next_multiple_of(speech_pad_ms, 32)

        self.detection_inference_interval_ms = detection_inference_interval_ms
        self.two_pass_min_pending_ms = two_pass_min_pending_ms
        self.two_pass_max_pending_ms = two_pass_max_pending_ms
        self._two_pass_tail_ms_count = two_pass_tail_ms_count
        self._two_pass_non_silence_weight = two_pass_non_silence_weight

        self.set_strategy = make_confidence_strategy("last")

        self.vad_rt = VADRT(model_path)
        self.two_pass_rt = TwoPassRT(two_pass_model_path, self.confidence_window_ms)
        assert self.vad_rt.sample_rate == 16000, "VAD model sample rate must be 16000"
        assert self.vad_rt.chunk_size == 512, "VAD model chunk size must be 512"
        self._vad_chunk_ms = self.vad_rt.chunk_size * 1000 // self.vad_rt.sample_rate  # 32 ms

        # Derive frame counts from the two-pass model's chunk size (10 ms).
        two_pass_chunk_ms = self.two_pass_rt.chunk_ms
        self._two_pass_tail_frame_count = two_pass_tail_ms_count // two_pass_chunk_ms
        # Upper bound on how far back any future compute call could look:
        # a PENDING window is at most ``two_pass_max_pending_ms`` and each
        # inference writes ``context_ms`` of history.  Anything older than this
        # can never influence a decision, so we prune it.
        self._frame_history_limit = (self.two_pass_rt.context_ms + two_pass_max_pending_ms) // two_pass_chunk_ms

        self._processed_audio_ms: int = 0
        self._remaining_audio: npt.NDArray[np.int16] = np.zeros(0, dtype=np.int16)
        self._remaining_raw_audio: bytes = b""

        self._max_buffer_chunks = self.speech_pad_ms // 32
        self._buffer: deque[npt.NDArray[np.int16]] = deque(maxlen=self._max_buffer_chunks)

        self._max_confidence_chunks = self.confidence_window_ms // 32
        self._sliding_window_speech_probs: deque[float] = deque(maxlen=self._max_confidence_chunks)

        self._triggered: bool = False
        self._paused: bool = False
        self._temp_end: int = 0  # in miliseconds
        self._current_segment_id: str | None = None
        self._last_trigger_ms: int = 0

        # Pending / sliding state
        self._vad_pending: bool = False
        self._pending_vad_ms: int = 0
        self._last_inference_ms: int = 0
        self._pending_silence_start_ms: int = 0
        self._two_pass_frame_timeline: dict[int, float] = {}
        self._two_pass_frame_top_ids: dict[int, int] = {}

        logger.info(
            "VAD initialized with: threshold: %.3f, two-pass threshold: %.3f, neg_threshold: %.3f, "
            "window_ms: %d, speech_pad_ms: %d, min_silence_duration_ms: %d, min_pause_duration_ms: %d, "
            "reset_on_long_silence_ms: %d, detection_inference_interval_ms: %d, two_pass_min_pending_ms: %d, two_pass_max_pending_ms: %d",
            threshold,
            two_pass_threshold,
            neg_threshold,
            self.confidence_window_ms,
            self.speech_pad_ms,
            min_silence_duration_ms,
            min_pause_duration_ms,
            reset_on_long_silence_ms,
            detection_inference_interval_ms,
            two_pass_min_pending_ms,
            two_pass_max_pending_ms,
        )

        self.__prev_bargein: float = 0.0  # to be deprecated

    def reset(self) -> None:
        self._triggered = False
        self._paused = False
        self._temp_end = 0
        self._last_trigger_ms = 0
        self._processed_audio_ms = 0
        self._current_segment_id = None
        self._remaining_audio = np.zeros(0, dtype=np.int16)
        self._remaining_raw_audio = b""
        self._buffer.clear()
        self._sliding_window_speech_probs.clear()
        self.vad_rt.reset()
        self._vad_pending = False
        self._pending_vad_ms = 0
        self._last_inference_ms = 0
        self._pending_silence_start_ms = 0
        self._two_pass_frame_timeline.clear()
        self._two_pass_frame_top_ids.clear()
        logger.info("VAD reset")

    def _clear_pending_state(self) -> None:
        # Deliberately leaves ``_two_pass_frame_timeline`` alone: the ratio
        # computation already filters to ``[T_vad, now]``, and _update_frame_timeline
        # prunes to ``_frame_history_limit`` so the dicts cannot grow without bound.
        self._vad_pending = False
        self._pending_vad_ms = 0
        self._last_inference_ms = 0
        self._pending_silence_start_ms = 0

    def _update_frame_timeline(self, t_now_ms: int, frame_probs: list[float], frame_top_ids: list[int]) -> None:
        """Write inference output into the frame timeline.

        The model covers ``[t_now - context_ms, t_now]`` in ``chunk_ms`` steps;
        later inferences overwrite overlapping frames because they have more
        right-side context. Entries older than ``_frame_history_limit`` frames
        are dropped so the timeline size stays bounded.

        Args:
            t_now_ms: Timestamp (ms) at the end of the inference window.
            frame_probs: Non-silence probabilities, one per ``chunk_ms`` frame.
            frame_top_ids: Argmax phoneme id per frame, aligned to ``frame_probs``.
        """
        chunk_ms = self.two_pass_rt.chunk_ms
        context_ms = self.two_pass_rt.context_ms
        t_start_ms = t_now_ms - context_ms
        latest_frame = (t_now_ms // chunk_ms) - 1
        for i, (prob, top_id) in enumerate(zip(frame_probs, frame_top_ids)):
            abs_frame = (t_start_ms + i * chunk_ms) // chunk_ms
            self._two_pass_frame_timeline[abs_frame] = prob
            self._two_pass_frame_top_ids[abs_frame] = int(top_id)

        cutoff = latest_frame - self._frame_history_limit
        if cutoff > 0:
            stale = [k for k in self._two_pass_frame_timeline if k < cutoff]
            for k in stale:
                del self._two_pass_frame_timeline[k]
                self._two_pass_frame_top_ids.pop(k, None)

    def _compute_two_pass_speech_ratio(self, t_vad_ms: int, t_now_ms: int) -> float:
        """Fraction of frames in ``[t_vad_ms, t_now_ms]`` classified as speech.

        Args:
            t_vad_ms: Start of the evaluation window (VAD trigger time, ms).
            t_now_ms: End of the evaluation window (current time, ms).

        Returns:
            Float in ``[0.0, 1.0]``.  Returns ``0.0`` if no frames are available.
        """
        chunk_ms = self.two_pass_rt.chunk_ms
        vad_frame = t_vad_ms // chunk_ms
        now_frame = t_now_ms // chunk_ms
        # Explicit sort: dict insertion order does not guarantee chronological
        # key order after overwrites, and the tail slice below assumes it.
        relevant = sorted((k, v, self._two_pass_frame_top_ids[k]) for k, v in self._two_pass_frame_timeline.items() if vad_frame <= k <= now_frame)
        if not relevant:
            return 0.0
        if len(relevant) > self._two_pass_tail_frame_count:
            relevant = relevant[-self._two_pass_tail_frame_count :]
        n = len(relevant)
        non_silence_scores = sum(v for _, v, _ in relevant) / n
        phoneme_scores = sum(1 for _, _, top_id in relevant if top_id != self.two_pass_rt.silence_id) / n
        return max(non_silence_scores * self._two_pass_non_silence_weight, phoneme_scores)

    async def run(self, audio_chunk: AudioChunk) -> AsyncIterator[AudioChunk | AudioEvent]:
        self._remaining_audio = np.concatenate([self._remaining_audio, audio_chunk.tonumpy()])
        self._remaining_raw_audio += audio_chunk.raw_data if audio_chunk.raw_data is not None else b""
        while len(self._remaining_audio) >= self.vad_rt.chunk_size:
            vad_chunk = self._remaining_audio[: self.vad_rt.chunk_size]
            raw_chunk = np.frombuffer(self._remaining_raw_audio[: self.vad_rt.chunk_size * 2], dtype=np.int16)
            self._remaining_audio = self._remaining_audio[self.vad_rt.chunk_size :]
            self._remaining_raw_audio = self._remaining_raw_audio[self.vad_rt.chunk_size * 2 :]
            self.two_pass_rt.append_audio(vad_chunk.tobytes())
            self._processed_audio_ms += 32

            speech_prob = await asyncio.to_thread(self.vad_rt.process_audio_chunk, vad_chunk)
            self._sliding_window_speech_probs.append(speech_prob)
            if self.return_raw_audio and raw_chunk.size > 0:
                self._buffer.append(raw_chunk)
            else:
                self._buffer.append(vad_chunk)
            confidence = self.set_strategy.calc_set_confidence(self._sliding_window_speech_probs, self._max_confidence_chunks)
            # logger.debug("Confidence: %.3f at %s ms", confidence, self._processed_audio_ms)

            if confidence >= self.speech_prob_threshold and self._temp_end:
                self._temp_end = 0
                if self._paused:
                    self._paused = False
                    logger.info("Resume triggered at %s ms", self._processed_audio_ms)
                    assert self._current_segment_id is not None, "Must have a current segment id"
                    yield AudioEvent(type=AudioEventType.RESUME, timestamp_ms=self._processed_audio_ms - 32, id=self._current_segment_id)

            if not self.enable_two_pass or not self.use_new_two_pass:
                if confidence >= self.speech_prob_threshold and not self._triggered:
                    logger.info("T: %d Silero: %.3f >= %.3f", self._processed_audio_ms, confidence, self.speech_prob_threshold)
                    two_pass_confidence = await asyncio.to_thread(self.two_pass_rt.get_speech_confidence) if self.enable_two_pass else 1.0
                    if two_pass_confidence >= self.two_pass_speech_prob_threshold:
                        logger.info(
                            "T: %d Two-pass: %.3f >= %.3f", self._processed_audio_ms, two_pass_confidence, self.two_pass_speech_prob_threshold
                        )
                        self._triggered = True
                        self.__prev_bargein = 0.0
                        self._current_segment_id = generate_new_id()
                        logger.info("Start triggered at %s ms", self._processed_audio_ms)
                        yield AudioEvent(
                            type=AudioEventType.START,
                            timestamp_ms=self._processed_audio_ms - (len(self._buffer) * 32),
                            id=self._current_segment_id,
                        )
                        yield AudioChunk(data=np.concatenate(self._buffer), id=self._current_segment_id)
                        continue
            else:
                if not self._triggered:
                    if self._vad_pending:
                        pending_duration = self._processed_audio_ms - self._pending_vad_ms
                        if self._processed_audio_ms - self._last_inference_ms >= self.detection_inference_interval_ms:
                            frame_probs, frame_top_ids = await asyncio.to_thread(self.two_pass_rt.get_all_frame_data)
                            self._update_frame_timeline(self._processed_audio_ms, frame_probs, frame_top_ids)
                            self._last_inference_ms = self._processed_audio_ms
                            two_pass_confidence = self._compute_two_pass_speech_ratio(self._pending_vad_ms, self._processed_audio_ms)
                            # Only trigger after two_pass_min_pending_ms; earlier inferences
                            # still update the timeline for retrospective refinement.
                            can_trigger = pending_duration >= self.two_pass_min_pending_ms
                            ratio_ok = two_pass_confidence >= self.two_pass_speech_prob_threshold
                            if can_trigger and ratio_ok:
                                logger.info(
                                    "T: %d Two-pass: %.3f >= %.3f", self._processed_audio_ms, two_pass_confidence, self.two_pass_speech_prob_threshold
                                )
                                self._triggered = True
                                self.__prev_bargein = 0.0
                                self._current_segment_id = generate_new_id()
                                logger.info("Start triggered at %s ms", self._processed_audio_ms)
                                self._clear_pending_state()
                                yield AudioEvent(
                                    type=AudioEventType.START,
                                    timestamp_ms=self._processed_audio_ms - (len(self._buffer) * self._vad_chunk_ms),
                                    id=self._current_segment_id,
                                )

                                yield AudioChunk(data=np.concatenate(self._buffer), id=self._current_segment_id)
                                continue
                        if speech_prob < self.neg_speech_prob_threshold:
                            if not self._pending_silence_start_ms:
                                self._pending_silence_start_ms = self._processed_audio_ms
                            silence_duration = self._processed_audio_ms - self._pending_silence_start_ms
                            if silence_duration >= self.min_silence_duration_ms:
                                logger.info(
                                    "PENDING cancelled (silence %d ms) at %d ms",
                                    silence_duration,
                                    self._processed_audio_ms,
                                )
                                self._clear_pending_state()
                        else:
                            self._pending_silence_start_ms = 0

                        if self._vad_pending and pending_duration >= self.two_pass_max_pending_ms:
                            logger.info(
                                "PENDING timed out after %d ms at %d ms",
                                pending_duration,
                                self._processed_audio_ms,
                            )
                            self._clear_pending_state()
                            pending_duration = 0
                    elif confidence >= self.speech_prob_threshold:
                        # VAD fired on the chunk that just ended at ``_processed_audio_ms``,
                        # so the speech onset sits one VAD chunk earlier.
                        onset_ms = max(0, self._processed_audio_ms - self._vad_chunk_ms)
                        self._vad_pending = True
                        self._pending_vad_ms = onset_ms
                        self._last_inference_ms = onset_ms

            if self._triggered:
                self._last_trigger_ms = self._processed_audio_ms
                if time.time() - self.__prev_bargein > 0.5:  # Barge in signal interval: 500ms
                    self.__prev_bargein = time.time()
                    yield AudioEvent(type=AudioEventType.BARGEIN, timestamp_ms=self._processed_audio_ms, id="bargein")
                if self.return_raw_audio and raw_chunk.size > 0:
                    yield AudioChunk(data=raw_chunk, id=self._current_segment_id)
                else:
                    yield AudioChunk(data=vad_chunk, id=self._current_segment_id)
            elif (
                self.reset_on_long_silence_ms > 0
                and self._last_trigger_ms > 0
                and (self._processed_audio_ms - self._last_trigger_ms) >= self.reset_on_long_silence_ms
            ):
                logger.info("Reset silero vad state due to long silence at %s ms", self._processed_audio_ms)
                # Keep tracking from current position to enable periodic resets during long silence
                self._last_trigger_ms = self._processed_audio_ms
                self.vad_rt.reset()

            if speech_prob < self.neg_speech_prob_threshold and self._triggered:
                if not self._temp_end:
                    self._temp_end = self._processed_audio_ms
                if self._processed_audio_ms - self._temp_end >= self.min_pause_duration_ms and not self._paused:
                    self._paused = True
                    logger.info("Pause triggered at %s ms", self._processed_audio_ms)
                    assert self._current_segment_id is not None, "Must have a current segment id"
                    yield AudioEvent(type=AudioEventType.PAUSE, timestamp_ms=self._processed_audio_ms, id=self._current_segment_id)
                if self._processed_audio_ms - self._temp_end >= self.min_silence_duration_ms:
                    self._triggered = False
                    self._paused = False
                    self._temp_end = 0
                    self._buffer.clear()
                    logger.info("End triggered at %s ms", self._processed_audio_ms)
                    assert self._current_segment_id is not None, "Must have a current segment id"
                    yield AudioEvent(type=AudioEventType.END, timestamp_ms=self._processed_audio_ms, id=self._current_segment_id)

    async def close(self) -> AsyncIterator[AudioChunk | AudioEvent]:
        if self._triggered and self._current_segment_id is not None:
            if self._remaining_audio.size > 0:
                if self.return_raw_audio and self._remaining_raw_audio:
                    remaining_audio = AudioChunk(
                        data=self._remaining_raw_audio,
                        raw_data=self._remaining_raw_audio,
                        id=self._current_segment_id,
                    )
                else:
                    remaining_audio = AudioChunk(data=self._remaining_audio, id=self._current_segment_id)
                self._processed_audio_ms += int(remaining_audio.duration_ms())
                yield remaining_audio
            if not self._paused:
                logger.info("PAUSE triggered at %s ms", self._processed_audio_ms)
                yield AudioEvent(type=AudioEventType.PAUSE, timestamp_ms=self._processed_audio_ms, id=self._current_segment_id)
            logger.info("End triggered at %s ms", self._processed_audio_ms)
            yield AudioEvent(type=AudioEventType.END, timestamp_ms=self._processed_audio_ms, id=self._current_segment_id)
        self._triggered = False
        self._paused = False
        self._temp_end = 0
        self._current_segment_id = None
        self._remaining_audio = np.zeros(0, dtype=np.int16)
        self._remaining_raw_audio = b""
        self._buffer.clear()



    def increase_timeout_duration(self, delta_ms: int = 500) -> None:
        """Adjust the VAD endpointing windows by ``delta_ms`` milliseconds.

        Both ``min_pause_duration_ms`` (drives the PAUSE event) and
        ``min_silence_duration_ms`` (drives the END event) are shifted by the
        same delta, preserving their relative gap. The resulting values are
        clamped so they never drop below the baseline values captured at
        construction time. There is no upper bound; callers are responsible
        for not extending the windows beyond what their use case needs.

        Args:
            delta_ms: Milliseconds to add to the current timeouts. May be
                negative to shrink the windows back toward the baseline; the
                result is clamped to the baseline.
        """
        prev_pause = self.min_pause_duration_ms
        prev_silence = self.min_silence_duration_ms
        requested_pause = prev_pause + delta_ms
        requested_silence = prev_silence + delta_ms
        new_pause = max(self._base_min_pause_duration_ms, requested_pause)
        new_silence = max(self._base_min_silence_duration_ms, requested_silence)

        self.min_pause_duration_ms = new_pause
        self.min_silence_duration_ms = new_silence

        pause_clamped = new_pause != requested_pause
        silence_clamped = new_silence != requested_silence
        if pause_clamped or silence_clamped:
            logger.info(
                "VAD timeout adjustment clamped to baseline: delta_ms=%d, "
                "requested min_pause_duration_ms=%d (clamped to %d, base=%d), "
                "requested min_silence_duration_ms=%d (clamped to %d, base=%d)",
                delta_ms,
                requested_pause,
                new_pause,
                self._base_min_pause_duration_ms,
                requested_silence,
                new_silence,
                self._base_min_silence_duration_ms,
            )
        else:
            logger.info(
                "VAD timeout adjusted: delta_ms=%d, min_pause_duration_ms %d -> %d (base=%d), min_silence_duration_ms %d -> %d (base=%d)",
                delta_ms,
                prev_pause,
                new_pause,
                self._base_min_pause_duration_ms,
                prev_silence,
                new_silence,
                self._base_min_silence_duration_ms,
            )


def next_multiple_of(x: int, n: int) -> int:
    """Calculate the next multiple of n that is greater than or equal to x"""
    return ((x + n - 1) // n) * n


def prev_multiple_of(x: int, n: int) -> int:
    """Calculate the previous multiple of n that is less than or equal to x"""
    return (x // n) * n


async def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / "test_case.wav"
    output_prefix = script_dir / "test_case_segment"
    vad = StreamingVAD(
        model_path="/Users/sylvan.huang/Project/unified-ai/bargeIn/silero_vad_16k.onnx",
        two_pass_model_path="/Users/sylvan.huang/Project/unified-ai/bargeIn/two_pass_barge_in.onnx",
    )

    chunk_size_ms: int = 100
    sampling_rate: int = 16000
    with wave.open(str(input_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1, "Only mono WAV is supported"
        assert wav_file.getsampwidth() == 2, "Only 16-bit PCM WAV is supported"
        assert wav_file.getframerate() == sampling_rate, f"WAV sample rate must be {sampling_rate}"
        input_pcm_bytes = wav_file.readframes(wav_file.getnframes())

    _chunk_size_samples = chunk_size_ms * (sampling_rate // 1000)

    input_audio = np.frombuffer(input_pcm_bytes, dtype=np.int16).copy()
    chunk_size_samples = _chunk_size_samples
    segment_order: list[str] = []
    segment_chunks: dict[str, list[bytes]] = {}

    def handle_output(item: AudioChunk | AudioEvent) -> None:
        print(item)
        if isinstance(item, AudioEvent) and item.type == AudioEventType.START and item.id not in segment_chunks:
            segment_order.append(item.id)
            segment_chunks[item.id] = []
            return
        if isinstance(item, AudioChunk) and item.id is not None:
            segment_chunks.setdefault(item.id, []).append(item.tobytes())

    for start_idx in range(0, len(input_audio), chunk_size_samples):
        chunk = input_audio[start_idx : start_idx + chunk_size_samples]
        original_chunk_len = len(chunk)

        # Keep fixed-size chunks for the streaming demo and let `close()` flush
        # the true tail audio after the last full VAD frame is processed.
        if original_chunk_len < chunk_size_samples:
            padded_chunk = np.pad(chunk, (0, chunk_size_samples - original_chunk_len), mode="constant").astype(np.int16)
        else:
            padded_chunk = chunk
        data = AudioChunk(data=padded_chunk, raw_data=chunk.tobytes())
        async for event in vad.run(data):
            handle_output(event)

    async for event in vad.close():
        handle_output(event)

    for index, segment_id in enumerate(segment_order, start=1):
        output_path = output_prefix.with_name(f"{output_prefix.name}_{index:02d}.wav")
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(b"".join(segment_chunks.get(segment_id, [])))
        print(f"Saved VAD segment {index} to: {output_path.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
