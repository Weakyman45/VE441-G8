"""VAD runtime"""

import warnings

import numpy as np
import numpy.typing as npt
import onnxruntime
import logging
logger = logging.getLogger(__name__)


class BargeInOnnxModel:
    def __init__(self, model_path: str, force_onnx_cpu: bool = True) -> None:
        self.sample_rate = 16000
        self.chunk_size = 512
        self.context_size = 64
        self.batch_size = 1

        # ONNX session setup
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(model_path, sess_options=opts)

        # Initialize states
        self._state = np.zeros((2, self.batch_size, 128), dtype=np.float32)
        self._context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        self._last_batch_size = self.batch_size

        # Check sample rate compatibility
        if self.sample_rate not in [8000, 16000] and not (self.sample_rate % 16000 == 0):
            warnings.warn(f"Sample rate {self.sample_rate} may not be compatible with the model", stacklevel=2)
        logger.info("barge in onnx model load success")

    def get_init_states(self) -> npt.NDArray[np.float32]:
        """Get initial states array"""
        self._state = np.zeros((2, self.batch_size, 128), dtype=np.float32)
        return self._state.copy()

    def get_init_context(self) -> npt.NDArray[np.float32]:
        """Get initial context array"""
        self._context = np.zeros((self.batch_size, self.context_size), dtype=np.float32)
        return self._context.copy()

    def get_speech_prob(
        self, audio_chunk: npt.NDArray[np.float32], context: npt.NDArray[np.float32], state: npt.NDArray[np.float32]
    ) -> tuple[float, npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Process audio and get speech probability

        Args:
            audio_chunk: Raw audio array
            context: Context array
            state: State array

        Returns:
            Tuple of (probability, new_context, new_state)
        """
        try:
            # Add batch dimension if needed
            if audio_chunk.ndim == 1:
                audio_chunk = audio_chunk[np.newaxis, :]

            if audio_chunk.shape[-1] != self.chunk_size:
                raise ValueError(
                    f"Provided number of samples is {audio_chunk.shape[-1]} "
                    f"(Supported values: {self.chunk_size} for {self.sample_rate} sample rate)"
                )
            if context.shape[-1] != self.context_size:
                raise ValueError(
                    f"Provided number of context samples is {context.shape[-1]} "
                    f"(Supported values: {self.context_size} for {self.sample_rate} sample rate)"
                )

            # For ONNX we'll concatenate instead of using an input buffer
            x = np.concatenate([context, audio_chunk], axis=1)

            # Run ONNX inference
            ort_inputs = {"input": x.astype(np.float32), "state": state.astype(np.float32), "sr": np.array(self.sample_rate, dtype=np.int64)}

            ort_outs = self.session.run(None, ort_inputs)
            out, new_state = ort_outs
            new_state = new_state.astype(np.float32)

            # Extract probability as a Python scalar
            prob = float(out.item())

            # Get new context from the input tensor (last context_size elements)
            new_context = x[:, -self.context_size :].astype(np.float32)

            return prob, new_context, new_state

        except Exception as e:
            print(f"Error analyzing audio with ONNX model: {e}")
            return 0.0, self.get_init_context(), self.get_init_states()


class VADRT:
    """VAD runtime, responsible for processing audio chunks and returning speech probability"""

    silero_vad_onnx_model: BargeInOnnxModel = None  # type: ignore[assignment]

    def __init__(self, model_path: str) -> None:
        self.states: npt.NDArray[np.float32]
        self.context: npt.NDArray[np.float32]
        self._load_model(model_path)
        self.reset()

        self.chunk_size = 512
        self.sample_rate = 16000
        self.context_size = 64

        logger.info("VAD runtime loaded successfully")

    @classmethod
    def _load_model(cls, model_path: str) -> None:
        if cls.silero_vad_onnx_model is not None:
            return
        cls.silero_vad_onnx_model = BargeInOnnxModel(model_path, force_onnx_cpu=True)

    def process_audio_chunk(self, audio_chunk: npt.NDArray[np.int16]) -> float:
        """
        Process audio chunk and return speech probability.

        Args:
            audio_chunk (npt.NDArray[np.int16]): Audio chunk to process, must be 32ms or 512 samples of 16000Hz audio

        Returns:
            float: Speech probability
        """
        audio_chunk_float: npt.NDArray[np.float32] = audio_chunk.astype(np.float32) / 32768.0
        prob, new_context, new_states = self.silero_vad_onnx_model.get_speech_prob(audio_chunk_float, self.context, self.states)
        self.context = new_context
        self.states = new_states
        return prob

    def reset(self) -> None:
        """Reset the VAD state and context"""
        self.states = self.silero_vad_onnx_model.get_init_states()
        self.context = self.silero_vad_onnx_model.get_init_context()
