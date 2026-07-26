"""executor + runtime wiring tests with mocked search / verifier."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.bus import EventBus  # noqa: E402
from engine.events import Event, EventType  # noqa: E402
from engine.exp_config import load_config  # noqa: E402
from engine.logging_store import LoggingStore  # noqa: E402
from engine.memory_store import MemoryStore  # noqa: E402
from engine.models import PreferenceProfile, SessionState  # noqa: E402
from engine.session import SessionStore  # noqa: E402
from engine.worker.executor import Executor  # noqa: E402
from engine.worker.plan_schema import (  # noqa: E402
    AGENT_RECOMMEND,
    AGENT_TEXT_RECALL,
    AGENT_VERIFY,
    build_dag,
)
from engine.worker.runtime import WorkerRuntime  # noqa: E402
from engine.modality_router import RoutePlan  # noqa: E402


CATALOG = [
    {
        "id": "p1",
        "name": "White Running Shoes",
        "price": 80,
        "rating": 4.6,
        "platform": "Windows",
        "summary": "breathable runners",
        "display": "",
        "performance": "",
        "weight_kg": 0.8,
        "image_url": "",
    },
    {
        "id": "p2",
        "name": "Office Chair",
        "price": 200,
        "rating": 4.2,
        "platform": "Windows",
        "summary": "ergonomic",
        "display": "",
        "performance": "",
        "weight_kg": 12,
        "image_url": "",
    },
]


def _search(params: dict) -> list[dict]:
    q = (params.get("q") or [""])[0].lower()
    hits = [c for c in CATALOG if q in c["name"].lower() or any(
        t in c["name"].lower() for t in q.split() if len(t) > 2
    )]
    return hits or list(CATALOG)


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("VS_")}
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ["VS_VERIFIER"] = "off"
        os.environ["VS_PLANNER_LLM"] = "0"
        os.environ["VS_MEMORY"] = "1"
        os.environ["VS_PLANNER_REPLAN"] = "1"

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ.update(self._saved)

    def test_recall_recommend_order(self):
        rp = RoutePlan(
            modality="text",
            do_text_recall=True,
            do_visual_recall=False,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=False,
            reason="t",
        )
        stages = [s.to_dict() for s in build_dag(route_plan=rp, shortcircuit=False, do_verify=False)]
        plan = {
            "plan_id": "abc",
            "stages": stages,
            "query_hint": "running shoes",
            "hints": {"query_hint": "running shoes", "relax_ops": []},
            "loop": {"max_replans": 1},
            "decision": {"replan_attempt": 0, "memory": {}},
            "writeback": {"memory_write": []},
        }
        state = SessionState(session_id="e1")
        state.preference = PreferenceProfile(
            search_keywords=["running", "shoes"], category="shoes"
        )
        ex = Executor(_search)
        result = ex.run(plan, state, config=load_config())
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.bundle)
        self.assertGreaterEqual(len(result.bundle.ranked), 1)
        agents = [t["agent"] for t in result.stage_trace]
        self.assertEqual(agents[0], AGENT_TEXT_RECALL)
        self.assertEqual(agents[-1], AGENT_RECOMMEND)

    def test_verify_reject_triggers_replan_flag(self):
        os.environ["VS_VERIFIER"] = "rule"
        rp = RoutePlan(
            modality="text",
            do_text_recall=True,
            do_visual_recall=False,
            infer_category_from_image=False,
            reverse_verify=False,
            do_verify=True,
            reason="t",
        )
        stages = [s.to_dict() for s in build_dag(route_plan=rp, shortcircuit=False, do_verify=True)]
        plan = {
            "plan_id": "abc2",
            "stages": stages,
            "query_hint": "shoes",
            "hints": {"relax_ops": ["drop_hard"]},
            "loop": {"max_replans": 2},
            "decision": {"replan_attempt": 0, "memory": {}},
            "writeback": {},
        }
        state = SessionState(session_id="e2")
        # Force category mismatch so rule verifier rejects all.
        state.preference = PreferenceProfile(
            search_keywords=["shoes"], category="zzzznotacategory"
        )
        ex = Executor(_search)
        with patch(
            "engine.worker.executor.verify_candidates",
            return_value=type(
                "VR",
                (),
                {
                    "kept": [],
                    "rejected": [{"id": "p1", "name": "x", "reason": "category"}],
                    "method": "rule",
                },
            )(),
        ):
            result = ex.run(plan, state, config=load_config())
        self.assertTrue(result.need_replan)
        self.assertIn("category", result.reject_reason)


class TestRuntimeIntegration(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("VS_")}
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ["VS_VERIFIER"] = "off"
        os.environ["VS_PLANNER_LLM"] = "0"
        os.environ["VS_MEMORY"] = "1"
        os.environ["VS_VISUAL_RECALL"] = "0"
        os.environ["VS_INTENT_SHORTCIRCUIT"] = "0"
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.bus = EventBus()
        self.sessions = SessionStore()
        self.logs = LoggingStore(os.path.join(self.tmp.name, "logs.db"))
        self.mem = MemoryStore(db_path=os.path.join(self.tmp.name, "mem.db"))
        self.ready = threading.Event()
        self.mem_writes = []

        def on_ready(event: Event):
            self.ready.set()

        def on_mem(event: Event):
            self.mem_writes.append(event.payload)

        self.bus.subscribe(EventType.WORKER_RECOMMENDATION_READY, on_ready)
        self.bus.subscribe(EventType.MEMORY_WRITE_REQUESTED, on_mem)
        self.runtime = WorkerRuntime(
            self.bus, self.sessions, self.logs, _search, memory=self.mem
        )
        self.runtime.start()

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()
        for k in list(os.environ):
            if k.startswith("VS_"):
                del os.environ[k]
        os.environ.update(self._saved)

    def test_pipeline_emits_recommendation_and_memory(self):
        state = self.sessions.create("rt1")
        state.preference = PreferenceProfile(
            search_keywords=["running", "shoes"],
            raw_query="running shoes",
            category="shoes",
        )
        state.conversation = [{"role": "user", "text": "running shoes"}]
        with patch("engine.worker.planner.enrichment.has_enrichment", return_value=False):
            self.bus.emit(
                Event(
                    type=EventType.USER_INTENT_UPDATED,
                    session_id="rt1",
                    payload=state.preference.to_dict(),
                )
            )
            ok = self.ready.wait(5.0)
        self.assertTrue(ok)
        snap = self.sessions.snapshot("rt1")
        self.assertIsNotNone(snap)
        self.assertTrue(snap["worker"]["last_bundle"]["ranked"])
        # memory write is async on same thread after recommend; give a beat
        deadline = time.time() + 2
        while not self.mem_writes and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.mem_writes)


if __name__ == "__main__":
    unittest.main()
