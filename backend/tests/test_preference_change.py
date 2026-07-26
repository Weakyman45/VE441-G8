"""Unit tests for preference change classification (soft vs hard replan)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.intent import classify_preference_change, extract_preference  # noqa: E402
from engine.models import PreferenceProfile  # noqa: E402


class TestPreferenceChange(unittest.TestCase):
    def test_hard_budget_change(self):
        prior = PreferenceProfile(category="laptop", budget=1500, hard=["Budget up to 1500"])
        cur = extract_preference("actually under 800", prior)
        self.assertEqual(classify_preference_change(prior, cur), "hard")

    def test_soft_use_case(self):
        prior = PreferenceProfile(
            category="laptop",
            budget=1500,
            search_keywords=["laptop"],
            soft=["Use case: study"],
        )
        cur = PreferenceProfile(
            category="laptop",
            budget=1500,
            search_keywords=["laptop"],
            soft=["Use case: study", "Portable / lightweight"],
            use_case="study",
        )
        self.assertEqual(classify_preference_change(prior, cur), "soft")

    def test_recall_keywords(self):
        prior = PreferenceProfile(category="shoes", search_keywords=["running", "shoes"])
        cur = PreferenceProfile(category="shoes", search_keywords=["trail", "shoes"])
        self.assertEqual(classify_preference_change(prior, cur), "recall")

    def test_none(self):
        p = PreferenceProfile(category="phone", budget=900, search_keywords=["phone"])
        self.assertEqual(classify_preference_change(p, p), "none")


if __name__ == "__main__":
    unittest.main()
