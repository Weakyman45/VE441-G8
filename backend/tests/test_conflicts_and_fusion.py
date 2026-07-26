"""Tests for ConflictPacket talker briefs and fuse-weight ranking."""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.conflicts import (  # noqa: E402
    ConflictPacket,
    build_talker_brief,
    packet_from_reject,
)
from engine.models import PreferenceProfile, RankedProduct  # noqa: E402
from engine.query_fusion import default_fuse_weights, normalize_fuse_weights  # noqa: E402
from engine.worker.recommend_worker import rank_products  # noqa: E402


class TestConflicts(unittest.TestCase):
    def test_packet_from_budget(self):
        p = packet_from_reject(
            product_id="1",
            product_name="Shoe",
            reason="over budget (price 900 > ~500)",
        )
        self.assertEqual(p.conflict_type, "budget")
        self.assertEqual(p.user_action, "relax_constraint")

    def test_talker_brief_includes_conflict_and_question(self):
        ranked = [
            RankedProduct(id="a", name="Alpha Shoe", price=80, score=90, reasons=["fit"]),
            RankedProduct(id="b", name="Beta Shoe", price=120, score=85, reasons=["fit"]),
        ]
        conflicts = [
            ConflictPacket(
                product_id="c",
                product_name="Gamma Clip",
                conflict_type="category_mismatch",
                constraint="category mismatch (want shoes)",
                status="violated",
                user_action="clarify",
            )
        ]
        brief = build_talker_brief(
            summary="Top pick is Alpha Shoe.",
            ranked=ranked,
            conflicts=conflicts,
            tradeoffs=[{"a": "a", "a_name": "Alpha Shoe", "b": "b", "b_name": "Beta Shoe", "axes": ["price"]}],
            open_questions=["Is waterproof a hard requirement?"],
        )
        self.assertIn("Alpha Shoe", brief)
        self.assertIn("Gamma Clip", brief)
        self.assertIn("trade-off", brief.lower())
        self.assertIn("waterproof", brief)


class TestFusionWeights(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(default_fuse_weights("text")["visual"], 0.0)
        self.assertGreater(default_fuse_weights("image")["visual"], 0.5)
        w = normalize_fuse_weights({"text": 2, "visual": 2}, modality="text_image")
        self.assertAlmostEqual(w["text"] + w["visual"], 1.0)

    def test_rank_uses_visual_score(self):
        profile = PreferenceProfile(category="shoes", search_keywords=["running", "shoes"])
        cands = [
            {
                "id": "1",
                "name": "random gadget",
                "price": 40,
                "rating": 4.9,
                "_visual_score": 0.05,
            },
            {
                "id": "2",
                "name": "trail running shoes",
                "price": 90,
                "rating": 4.2,
                "_visual_score": 0.82,
            },
        ]
        with patch("engine.worker.recommend_worker.qwen_configured", return_value=False):
            bundle = rank_products(
                "p1",
                profile,
                cands,
                fuse_weights={"text": 0.4, "visual": 0.6},
                modality="text_image",
            )
        self.assertTrue(bundle.ranked)
        self.assertEqual(bundle.ranked[0].id, "2")
        self.assertTrue(bundle.talker_brief)


if __name__ == "__main__":
    unittest.main()
