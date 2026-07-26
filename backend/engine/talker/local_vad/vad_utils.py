from collections import deque
from typing import Literal, Protocol


class ConfidenceStrategy(Protocol):
    def calc_set_confidence(self, sliding_window_probs: deque[float], max_chunks: int) -> float:
        """Calculate the confidence from the sliding window of speech probabilities for setting the VAD state"""
        raise NotImplementedError()


class AverageConfidenceStrategy(ConfidenceStrategy):
    def calc_set_confidence(self, sliding_window_probs: deque[float], max_chunks: int) -> float:
        if len(sliding_window_probs) < max_chunks:
            return 0.0
        return sum(sliding_window_probs) / max_chunks


class MinConfidenceStrategy(ConfidenceStrategy):
    def calc_set_confidence(self, sliding_window_probs: deque[float], max_chunks: int) -> float:
        if len(sliding_window_probs) < max_chunks:
            return 0.0
        return min(sliding_window_probs)


class LastConfidenceStrategy(ConfidenceStrategy):
    def calc_set_confidence(self, sliding_window_probs: deque[float], max_chunks: int) -> float:
        return sliding_window_probs[-1]


def make_confidence_strategy(strategy: Literal["average", "min", "last"]) -> ConfidenceStrategy:
    if strategy == "average":
        return AverageConfidenceStrategy()
    elif strategy == "min":
        return MinConfidenceStrategy()
    elif strategy == "last":
        return LastConfidenceStrategy()
    raise ValueError(f"Invalid strategy: {strategy}")
