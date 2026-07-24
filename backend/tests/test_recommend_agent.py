from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from engine.models import PreferenceProfile
from engine.recommend_agent import RANKING_WEIGHTS, rank_products


ROOT = Path(__file__).resolve().parents[2]


def product(product_id: str, **overrides):
    item = {
        "id": product_id,
        "name": f"Product {product_id}",
        "price": 80,
        "rating": 4.2,
        "rating_number": 100,
        "summary": "A reliable everyday product",
        "review_aspects": {
            "pros": ["reliable"],
            "cons": [],
            "issues": [],
            "aspects": {"quality": "positive"},
            "summary": "Generally reliable.",
        },
    }
    item.update(overrides)
    return item


class RecommendAgentTests(unittest.TestCase):
    def test_weights_match_architecture(self):
        self.assertEqual(
            RANKING_WEIGHTS,
            {
                "visual_similarity": 0.30,
                "nice_to_have": 0.25,
                "quality": 0.20,
                "price": 0.15,
                "preference_history": 0.10,
            },
        )

    def test_visual_similarity_changes_final_order(self):
        profile = PreferenceProfile()
        bundle = rank_products(
            "visual-plan",
            profile,
            [
                product("low", visual_similarity=0.15),
                product("high", visual_similarity=0.92),
            ],
        )

        self.assertEqual([item.id for item in bundle.ranked], ["high", "low"])
        self.assertEqual(bundle.ranked[0].score_breakdown["visual_similarity"], 92.0)

    def test_nice_to_have_price_quality_and_history_are_explainable(self):
        profile = PreferenceProfile(
            budget=100,
            soft=["gentle charcoal cleanser"],
            preference_history=["acne skincare"],
        )
        enriched = product(
            "match",
            price=75,
            rating=4.8,
            rating_number=500,
            enriched_text="gentle charcoal acne skincare cleanser",
            visual_attrs={"visual": {"product_category": "skincare"}},
            review_aspects=json.dumps(
                {
                    "pros": ["gentle", "effective for acne"],
                    "cons": ["small bottle"],
                    "issues": [],
                    "aspects": {"effectiveness": "positive"},
                    "summary": "Effective and gentle for acne-prone skin.",
                }
            ),
        )
        other = product("other", price=130, rating=3.8, rating_number=5)

        bundle = rank_products("preference-plan", profile, [other, enriched])
        top = bundle.ranked[0]

        self.assertEqual(top.id, "match")
        self.assertIn("nice_to_have", top.score_breakdown)
        self.assertIn("quality", top.score_breakdown)
        self.assertIn("price", top.score_breakdown)
        self.assertIn("preference_history", top.score_breakdown)
        self.assertEqual(top.review_pros[0], "gentle")
        self.assertEqual(top.review_cons[0], "small bottle")

    def test_explicit_upstream_rejection_is_not_reverified(self):
        bundle = rank_products(
            "verified-plan",
            PreferenceProfile(),
            [
                product("accepted", verified=True),
                product("rejected", verified=False, rejection_reason="Wrong category"),
            ],
        )

        self.assertEqual([item.id for item in bundle.ranked], ["accepted"])
        self.assertEqual(bundle.excluded, [{"id": "rejected", "reason": "Wrong category"}])

    def test_new_catalog_enrichment_can_be_ranked(self):
        db_path = ROOT / "catalog.db"
        self.assertTrue(db_path.exists())
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT id, name, price, rating, rating_number, summary, display,
                       performance, battery, weight_kg, image_url, platform,
                       visual_attrs, enriched_text, review_aspects, review_count_used
                  FROM laptops
                 WHERE review_aspects IS NOT NULL AND review_aspects <> ''
                 LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        bundle = rank_products("catalog-plan", PreferenceProfile(), [dict(row)])
        self.assertEqual(len(bundle.ranked), 1)
        self.assertIn("quality", bundle.ranked[0].score_breakdown)
        self.assertTrue(bundle.ranked[0].review_summary)


if __name__ == "__main__":
    unittest.main()
