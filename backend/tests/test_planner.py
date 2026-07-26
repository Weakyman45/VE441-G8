"""planner.py unit tests: DAG compile, shortcircuit, refuse, replan relax."""
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
from engine.worker import planner  # noqa: E402
from engine.worker.plan_schema import (  # noqa: E402
    AGENT_RECOMMEND,
    AGENT_RERANK_EXISTING,
    AGENT_TEXT_RECALL,
    AGENT_VERIFY,
    build_dag,
)
from engine.modality_router import RoutePlan  # noqa: E402


class TestBuildDag(unittest.TestCase):
    def test_text_verify_recommend(self):
        rp = RoutePlan(
            modality="text",
            do_text_recall=True,
            do_visual_recall=False,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=True,
            reason="t",
        )
        stages = build_dag(route_plan=rp, shortcircuit=False, do_verify=True)
        agents = [s.agent for s in stages]
        self.assertEqual(agents, [AGENT_TEXT_RECALL, AGENT_VERIFY, AGENT_RECOMMEND])

    def test_shortcircuit(self):
        stages = build_dag(route_plan=None, shortcircuit=True, do_verify=True)
        self.assertEqual([s.agent for s in stages], [AGENT_RERANK_EXISTING])


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("VS_")}
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ["VS_PLANNER_LLM"] = "0"
        os.environ["VS_MEMORY"] = "1"
        os.environ["VS_INTENT_SHORTCIRCUIT"] = "1"
        os.environ["VS_VERIFIER"] = "rule"
        os.environ["VS_VISUAL_RECALL"] = "0"
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.mem = MemoryStore(db_path=os.path.join(self.tmp.name, "m.db"))

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ.update(self._saved)

    def _session(self) -> SessionState:
        s = SessionState(session_id="p1")
        s.preference = PreferenceProfile(
            category="shoes",
            search_keywords=["running", "shoes"],
            raw_query="白色跑鞋",
        )
        s.conversation = [{"role": "user", "text": "白色跑鞋"}]
        return s

    def test_rules_dag(self):
        with patch("engine.worker.planner.enrichment.has_enrichment", return_value=False):
            plan = planner.plan(self._session(), memory=self.mem, config=load_config())
        agents = [s["agent"] for s in plan["stages"]]
        self.assertIn(AGENT_TEXT_RECALL, agents)
        self.assertIn(AGENT_RECOMMEND, agents)
        self.assertEqual(plan["planner"], "rules")
        self.assertTrue(plan["query_hint"])

    def test_shortcircuit_rerank(self):
        s = self._session()
        s.worker.last_bundle = RecommendationBundle(
            plan_id="old",
            ranked=[
                RankedProduct(id="1", name="A", price=1, score=1, reasons=[]),
            ],
        )
        s.conversation = [{"role": "user", "text": "第二个便宜一点"}]
        s.preference.raw_query = "第二个便宜一点"
        with patch("engine.worker.planner.enrichment.has_enrichment", return_value=False):
            plan = planner.plan(
                s, utterance="第二个便宜一点", memory=self.mem, config=load_config()
            )
        self.assertEqual(plan["decision"]["intent"], "refine")
        self.assertTrue(plan["decision"]["shortcircuit"])
        self.assertEqual(plan["stages"][0]["agent"], AGENT_RERANK_EXISTING)

    def test_refuse_dangerous(self):
        s = self._session()
        s.conversation = [{"role": "user", "text": "I need an illegal weapon firearm"}]
        with patch("engine.worker.planner.shopping_safety_enabled", return_value=True):
            plan = planner.plan(
                s,
                utterance="I need an illegal weapon firearm",
                memory=self.mem,
                config=load_config(),
            )
        self.assertTrue(plan["refused"])
        self.assertEqual(plan["stages"], [])

    def test_replan_applies_relax(self):
        s = self._session()
        s.preference.hard = ["must be red"]
        s.preference.budget = 100
        with patch("engine.worker.planner.enrichment.has_enrichment", return_value=False):
            plan = planner.plan(
                s,
                memory=self.mem,
                config=load_config(),
                replan_ctx={
                    "attempt": 1,
                    "reject_reason": "over budget",
                    "relax_ops": ["drop_hard", "raise_budget_20"],
                },
            )
        self.assertEqual(s.preference.hard, [])
        self.assertEqual(s.preference.budget, 120)
        self.assertIn("drop_hard", plan["decision"]["applied_relax"])

    def test_llm_hints_when_enabled(self):
        os.environ["VS_PLANNER_LLM"] = "1"
        s = self._session()
        fake = {
            "query_hint": "white running shoes",
            "reason": "ok",
            "focus": ["breathable"],
            "intent": "new_search",
            "relax_ops": [],
        }
        with patch("engine.worker.planner.enrichment.has_enrichment", return_value=False), patch(
            "engine.worker.planner.qwen_configured", return_value=True
        ), patch("engine.worker.planner._llm_intent", return_value="new_search"), patch(
            "engine.worker.planner.chat_json", return_value=fake
        ):
            plan = planner.plan(s, memory=self.mem, config=load_config())
        self.assertEqual(plan["planner"], "llm")
        self.assertEqual(plan["query_hint"], "white running shoes")
        self.assertEqual(plan["focus"], ["breathable"])


if __name__ == "__main__":
    unittest.main()
