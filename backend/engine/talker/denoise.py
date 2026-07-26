from __future__ import annotations

import base64
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any


class TalkerDenoiser:
    """Optional Krisp denoise transform for Realtime PCM16 append frames."""

    def __init__(
        self,
        *,
        input_sample_rate: int,
        model_path: str,
        wheel_path: str = "",
        suppression_level: int = 100,
    ) -> None:
        self.input_sample_rate = input_sample_rate
        self.denoise_sample_rate = 16_000
        self.suppression_level = suppression_level
        self.frame_duration_ms = 20
        self.chunk_duration_ms = 100
        self._pending_denoise_pcm = b""

        if input_sample_rate <= 0:
            raise ValueError("input_sample_rate must be positive")
        if not model_path:
            raise ValueError("TALKER_DENOISE_MODEL is required when denoise is enabled")
        self.model_path = str(Path(model_path).expanduser())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"denoise model not found: {self.model_path}")

        if wheel_path:
            self._prepare_wheel(wheel_path)

        try:
            import krisp_audio  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Krisp denoise requires krisp_audio and numpy in the Python runtime that starts the backend. "
                "Set TALKER_DENOISE_WHEEL for krisp_audio and install numpy for that interpreter."
            ) from exc

        self._krisp_audio = krisp_audio
        self._np = np
        self._ensure_krisp_initialized()
        self._session = self._create_session()
        self._chunk_samples = self.chunk_duration_ms * (self.denoise_sample_rate // 1000)
        self._frame_samples = self.frame_duration_ms * (self.denoise_sample_rate // 1000)

    def process_client_text(self, text: str) -> list[str] | None:
        """Replace input_audio_buffer.append audio with denoised audio."""
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return None
        if frame.get("type") != "input_audio_buffer.append":
            return None
        audio_b64 = frame.get("audio")
        if not audio_b64:
            return []
        try:
            pcm = base64.b64decode(audio_b64)
        except Exception:  # noqa: BLE001
            return []

        denoised = self.process_pcm16(pcm)
        if not denoised:
            return []
        frame["audio"] = base64.b64encode(denoised).decode("ascii")
        return [json.dumps(frame)]

    def process_pcm16(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        denoise_pcm = self._resample_pcm16_bytes(
            pcm,
            self.input_sample_rate,
            self.denoise_sample_rate,
        )
        self._pending_denoise_pcm += denoise_pcm
        chunk_bytes = self._chunk_samples * 2
        out_chunks: list[bytes] = []
        while len(self._pending_denoise_pcm) >= chunk_bytes:
            chunk = self._pending_denoise_pcm[:chunk_bytes]
            self._pending_denoise_pcm = self._pending_denoise_pcm[chunk_bytes:]
            out_chunks.append(self._process_100ms_chunk(chunk))
        if not out_chunks:
            return b""
        processed = b"".join(out_chunks)
        return self._resample_pcm16_bytes(
            processed,
            self.denoise_sample_rate,
            self.input_sample_rate,
        )

    def _process_100ms_chunk(self, pcm_16k: bytes) -> bytes:
        audio = self._np.frombuffer(pcm_16k, dtype=self._np.int16)
        frames = []
        for start in range(0, self._chunk_samples, self._frame_samples):
            frame = audio[start:start + self._frame_samples]
            if len(frame) < self._frame_samples:
                frame = self._np.pad(frame, (0, self._frame_samples - len(frame)), mode="constant")
            frames.append(self._session.process(frame.astype(self._np.int16), self.suppression_level))
        return self._np.concatenate(frames).astype(self._np.int16).tobytes()

    def _create_session(self) -> Any:
        krisp_audio = self._krisp_audio
        model_info = krisp_audio.ModelInfo()
        model_info.path = self.model_path

        cfg = krisp_audio.NcSessionConfig()
        cfg.inputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd20ms
        cfg.outputSampleRate = krisp_audio.SamplingRate.Sr16000Hz
        cfg.modelInfo = model_info
        return krisp_audio.NcInt16.create(cfg)

    def _ensure_krisp_initialized(self) -> None:
        krisp_audio = self._krisp_audio
        if getattr(TalkerDenoiser, "_krisp_initialized", False):
            return
        krisp_audio.globalInit("", _krisp_log_silent, krisp_audio.LogLevel.Off)
        TalkerDenoiser._krisp_initialized = True

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

    def _prepare_wheel(self, wheel_path: str) -> None:
        wheel = Path(wheel_path).expanduser()
        if not wheel.exists():
            raise FileNotFoundError(f"denoise wheel not found: {wheel}")
        target = Path(os.environ.get("TALKER_DENOISE_WHEEL_EXTRACT_DIR", "/private/tmp/voiceshop-krisp-audio"))
        marker = target / wheel.name
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(target)
            marker.write_text("extracted\n", encoding="utf-8")
        if str(target) not in sys.path:
            sys.path.insert(0, str(target))


def talker_denoise_requested() -> bool:
    raw = os.environ.get("TALKER_DENOISE_ENABLED")
    if raw is None:
        return False
    return raw.strip().lower() not in ("0", "false", "no", "off")


def build_talker_denoiser(input_sample_rate: int) -> TalkerDenoiser:
    return TalkerDenoiser(
        input_sample_rate=input_sample_rate,
        model_path=os.environ.get("TALKER_DENOISE_MODEL", "").strip(),
        wheel_path=os.environ.get("TALKER_DENOISE_WHEEL", "").strip(),
        suppression_level=int(os.environ.get("TALKER_DENOISE_SUPPRESSION_LEVEL", "100")),
    )


def _krisp_log_silent(msg: str, level: Any) -> None:
    _ = (msg, level)
