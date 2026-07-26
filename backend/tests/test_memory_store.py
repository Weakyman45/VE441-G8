"""memory_store.py unit tests: prefill, ref resolve, opt-in user writes."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.exp_config import load_config  # noqa: E402
from engine.memory_store import MemoryStore  # noqa: E402
from engine.models import (  # noqa: E402
    PreferenceProfile,
    RankedProduct,
    RecommendationBundle,
    SessionState,
)


def _cfg(**overrides):
    os.environ.setdefault("VS_MEMORY", "1")
    os.environ.setdefault("VS_PLANNER_LLM", "0")
    c = load_config()
    # ExpConfig is frozen — rebuild via env for tests that need toggles.
    return c


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("VS_")}
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ["VS_MEMORY"] = "1"
        os.environ["VS_PLANNER_LLM"] = "0"
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = MemoryStore(db_path=os.path.join(self.tmp.name, "memory.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ.update(self._saved)

    def test_prefill_does_not_override(self):
        session = SessionState(session_id="s1")
        session.preference = PreferenceProfile(budget=1000, category="")
        mem = self.store.get_session("s1")
        mem.preference_slots = {"budget": 500, "category": "shoes"}
        meta = self.store.prefill(session, config=load_config())
        self.assertEqual(session.preference.budget, 1000)
        self.assertEqual(session.preference.category, "shoes")
        self.assertIn("category", meta["prefilled_keys"])
        self.assertNotIn("budget", meta["prefilled_keys"])

    def test_resolve_ordinal_rules(self):
        session = SessionState(session_id="s2")
        session.worker.last_bundle = RecommendationBundle(
            plan_id="p",
            ranked=[
                RankedProduct(id="a", name="Alpha Shoe", price=10, score=90, reasons=[]),
                RankedProduct(id="b", name="Beta Shoe", price=20, score=80, reasons=[]),
            ],
        )
        refs = self.store.resolve_refs("我要第二个", session, config=load_config(), use_llm=False)
        self.assertEqual(refs["product_ids"], ["b"])
        self.assertEqual(refs["method"], "rules")

    def test_user_write_requires_opt_in(self):
        session = SessionState(session_id="s3")
        writes = [{"key": "category", "value": "bags", "scope": "user"}]
        meta = self.store.apply_writes(
            session=session,
            writes=writes,
            user_id="u1",
            opted_in_memory=False,
            config=load_config(),
        )
        self.assertEqual(meta["user_writes"], 0)
        self.assertEqual(self.store.load_user_facts("u1"), {})

        meta2 = self.store.apply_writes(
            session=session,
            writes=writes,
            user_id="u1",
            opted_in_memory=True,
            config=load_config(),
        )
        self.assertEqual(meta2["user_writes"], 1)
        self.assertEqual(self.store.load_user_facts("u1").get("category"), "bags")

    def test_memory_disabled(self):
        os.environ["VS_MEMORY"] = "0"
        session = SessionState(session_id="s4")
        meta = self.store.prefill(session, config=load_config())
        self.assertFalse(meta["enabled"])

    def test_llm_resolve_fallback(self):
        os.environ["VS_PLANNER_LLM"] = "1"
        session = SessionState(session_id="s5")
        session.worker.last_bundle = RecommendationBundle(
            plan_id="p",
            ranked=[
                RankedProduct(id="x", name="Red Bag", price=1, score=1, reasons=[]),
            ],
        )
        with patch("engine.memory_store.qwen_configured", return_value=True), patch(
            "engine.memory_store.chat_json",
            return_value={"product_ids": ["x"]},
        ):
            refs = self.store.resolve_refs(
                "the reddish one please",
                session,
                config=load_config(),
                use_llm=True,
            )
        self.assertEqual(refs["product_ids"], ["x"])
        self.assertEqual(refs["method"], "llm")


if __name__ == "__main__":
    unittest.main()
