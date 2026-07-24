"""Catalog retrieval workers used before the Recommend Agent."""

from .merge import merge_candidates
from .text_retrieval import retrieve_text
from .visual_retrieval import retrieve_visual

__all__ = ["merge_candidates", "retrieve_text", "retrieve_visual"]
