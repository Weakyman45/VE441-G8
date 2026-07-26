from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import onnxruntime
import logging
logger = logging.getLogger(__name__)


class BargeInTwoPassOnnxModel:
    # NOTE: This currently is not a streaming model. If cost too high, we can train a streaming version of it.
    def __init__(self, model_path: str, confidence_window_ms: int = 100, force_onnx_cpu: bool = True):
        self.sample_rate = 16000
        self.chunk_ms = 10
        self.context_ms = 500
        self.confidence_window_ms = confidence_window_ms
        self.batch_size = 1
        self.num_phonemes = 364
        self.silence_id = 1
        assert self.confidence_window_ms <= self.context_ms, "confidence_window_ms must be less than or equal to context_ms"

        # Calculate derived parameters
        assert self.sample_rate % 1000 == 0, "sample_rate must be divisible by 1000"
        self.chunk_size = int(self.chunk_ms * self.sample_rate / 1000)
        self.context_size = int(self.context_ms * self.sample_rate / 1000)
        self.confidence_window_size = int(self.confidence_window_ms * self.sample_rate / 1000)
        assert self.context_ms % self.chunk_ms == 0, "context_ms must be divisible by chunk_ms"
        self.context_chunks = self.context_ms // self.chunk_ms
        assert self.confidence_window_ms % self.chunk_ms == 0, "confidence_window_ms must be divisible by chunk_ms"
        self.confidence_chunks = self.confidence_window_ms // self.chunk_ms

        # ONNX session setup
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(model_path, sess_options=opts)

        # Initialize context buffer for stateless operation (not used but kept for derived calculations)
        self._last_batch_size = self.batch_size

        # Check sample rate compatibility
        if self.sample_rate not in [8000, 16000] and not (self.sample_rate % 16000 == 0):
            warnings.warn(f"Sample rate {self.sample_rate} may not be compatible with the model", stacklevel=2)
        logger.info("barge in two pass onnx model load success")

    def _process_chunks_and_get_confidence(self, audio_1s: npt.NDArray[np.float32]) -> float:
        """Process 1s audio in chunks and calculate confidence from last 300ms"""
        try:
            # Add batch dimension
            if audio_1s.ndim == 1:
                audio_1s = audio_1s[np.newaxis, :]

            # Prepare inputs for ONNX model
            lengths = np.array([1.0], dtype=np.float32)  # Relative length

            # Run ONNX inference
            ort_inputs = {"audio": audio_1s, "lengths": lengths}
            ort_outs = self.session.run(None, ort_inputs)
            logits, output_lengths = ort_outs

            # Convert logits to probabilities
            # Apply softmax manually
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Get the actual output length
            actual_length = int(output_lengths[0])

            # Take probabilities for the valid length
            valid_probs = probs[0, :actual_length, :]  # [time, num_classes]

            # Calculate confidence from last confidence_chunks frames
            if len(valid_probs) >= self.confidence_chunks:
                last_probs = valid_probs[-self.confidence_chunks :]  # Last 30 chunks (300ms)
            else:
                last_probs = valid_probs  # Use all available if less than 300ms

            # Calculate non-silence probability
            silence_probs = last_probs[:, self.silence_id]  # [confidence_chunks]
            non_silence_probs = 1.0 - silence_probs

            # Return mean confidence
            confidence = float(np.mean(non_silence_probs))
            return confidence

        except Exception as e:
            logger.info(f"Error in chunk processing: {e}")
            return 0.0

    def _process_chunks(self, audio_array: npt.NDArray[np.float32]) -> npt.NDArray[np.float32] | None:
        """Process audio and get all frame data for sliding detection"""
        try:
            # Add batch dimension
            if audio_array.ndim == 1:
                audio_array = audio_array[np.newaxis, :]

            # Prepare inputs for ONNX model
            lengths = np.array([1.0], dtype=np.float32)  # Relative length

            # Run ONNX inference
            ort_inputs = {"audio": audio_array, "lengths": lengths}
            ort_outs = self.session.run(None, ort_inputs)
            logits, output_lengths = ort_outs

            # Convert logits to probabilities
            # Apply softmax manually
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Get the actual output length
            actual_length = int(output_lengths[0])

            # Take probabilities for the valid length
            valid_probs = probs[0, :actual_length, :]  # [time, num_classes]

            return valid_probs

        except Exception as e:
            logger.info(f"Error in chunk processing: {e}")
            return None

    def get_speech_confidence(self, audio_chunk: npt.NDArray[np.float32]) -> float:
        """
        Process audio and get speech probability

        Args:
            audio_chunk: Raw audio array of 1 second of audio

        Returns:
            Speech probability (float: 0.0 = silence, 1.0 = speech)
        """
        try:
            # Process and get confidence from last 300ms
            confidence = self._process_chunks_and_get_confidence(audio_chunk)
            return confidence
        except Exception as e:
            logger.info(f"Error analyzing audio with ONNX two-pass model: {e}")
            return 0.0


class TwoPassRT:
    two_pass_model: BargeInTwoPassOnnxModel = None  # type: ignore[assignment]

    def __init__(self, model_path: str, confidence_window_ms: int = 100, consecutive_threshold: int = 1):
        # Run two pass model every `consecutive_threshold` chunks
        # larger value means less frequent two pass model execution, thus lower cost and higher latency
        # TODO: make this exponential backoff
        assert consecutive_threshold > 0, "consecutive_threshold must be greater than 0"
        self._consecutive_threshold = consecutive_threshold
        self._confidence_window_ms = confidence_window_ms
        self._chunk_idx = 0
        self._prev_processed_chunk_idx = 0

        self._load_model(model_path)
        self._buffer: bytes = b"\x00" * 2 * self.two_pass_model.context_size
        self._buffer_size: int = 2 * self.two_pass_model.context_size
        self.chunk_ms = self.two_pass_model.chunk_ms
        self.context_ms = self.two_pass_model.context_ms
        self.silence_id = self.two_pass_model.silence_id

        logger.info(
            "Two pass runtime loaded successfully. consecutive_threshold: %s, confidence_window_ms: %s", consecutive_threshold, confidence_window_ms
        )

    @classmethod
    def _load_model(cls, model_path: str) -> None:
        if cls.two_pass_model is not None:
            return
        cls.two_pass_model = BargeInTwoPassOnnxModel(model_path)

    def append_audio(self, audio_chunk: bytes) -> None:
        self._chunk_idx += 1
        self._buffer += audio_chunk
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size :]

    def get_speech_confidence(self) -> float:
        if self._chunk_idx - self._prev_processed_chunk_idx < self._consecutive_threshold:
            return 0.0
        self._prev_processed_chunk_idx = self._chunk_idx
        audio_array = np.frombuffer(self._buffer, np.int16).astype(np.float32) / 32768.0
        return self.two_pass_model.get_speech_confidence(audio_array)

    def get_all_frame_data(self) -> tuple[list[float], npt.NDArray[np.intp]]:
        """Run inference; return non-silence probabilities AND top phonemes per frame.

        Returns:
            Tuple ``(non_silence_probs, frame_top_ids)`` where:

            * ``non_silence_probs``: list of floats (one per 10 ms frame).
            * ``frame_top_ids``: list of ints (one per 10 ms frame).
        """
        assert self.two_pass_model is not None
        audio_array = np.frombuffer(self._buffer, np.int16).astype(np.float32) / 32768.0
        valid_probs = self.two_pass_model._process_chunks(audio_array)  # [time, num_classes]
        if valid_probs is None:
            return [], []
        non_silence_probs = (1.0 - valid_probs[:, self.two_pass_model.silence_id]).tolist()  # [time]
        valid_top_ids: npt.NDArray[np.intp] = np.argmax(valid_probs, axis=-1)  # [time]
        return non_silence_probs, valid_top_ids


if __name__ == "__main__":
    model_path = "/Users/JiachengLuo/Code/unified-ai/bargeIn/two_pass_barge_in.onnx"
    model = BargeInTwoPassOnnxModel(model_path)
    audio_chunk = np.random.rand(16000).astype(np.float32)
    confidence = model.get_speech_confidence(audio_chunk)
    print(confidence)
