"""Compatibility import for the standalone Recommend Agent.

New code should import :mod:`engine.recommend_agent` directly.
"""

from ..recommend_agent import RANKING_WEIGHTS, rank_products

__all__ = ["RANKING_WEIGHTS", "rank_products"]
