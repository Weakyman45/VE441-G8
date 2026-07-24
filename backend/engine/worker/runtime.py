from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..bus import EventBus
from ..events import Event, EventType
from ..logging_store import LoggingStore
from ..retrieval import merge_candidates, retrieve_text, retrieve_visual
from ..session import SessionStore
from . import planner
from ..recommend_agent import rank_products


class WorkerRuntime:
    """Background planner–worker loop listening on the shared EventBus."""

    def __init__(
        self,
        bus: EventBus,
        sessions: SessionStore,
        logs: LoggingStore,
        catalog_fn: Callable[[], list[dict]],
    ) -> None:
        self.bus = bus
        self.sessions = sessions
        self.logs = logs
        self.catalog_fn = catalog_fn
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        # Intent update already follows each utterance from TalkerBridge.
        bus.subscribe(EventType.USER_INTENT_UPDATED, self._on_user_signal)
        bus.subscribe(EventType.SESSION_INTERRUPTED, self._on_interrupted)

    def start(self) -> None:
        print("[worker] runtime listening for session events")

    def _on_interrupted(self, event: Event) -> None:
        cancel = self._cancel_flag(event.session_id)
        cancel.set()
        self.sessions.set_worker_status(event.session_id, "cancelled", "Interrupted by user")
        self.logs.log_trace(event.session_id, "runtime", "interrupted", event.payload)
        self.bus.emit(
            Event(
                type=EventType.WORKER_STATUS,
                session_id=event.session_id,
                payload={"status": "cancelled", "message": "Re-planning after interrupt"},
            )
        )

    def _on_user_signal(self, event: Event) -> None:
        # Run each plan on a dedicated thread so Talker stays non-blocking.
        threading.Thread(
            target=self._run_pipeline,
            args=(event.session_id, event),
            name=f"worker-{event.session_id[:8]}",
            daemon=True,
        ).start()

    def _cancel_flag(self, session_id: str) -> threading.Event:
        with self._lock:
            flag = self._cancel.get(session_id)
            if flag is None:
                flag = threading.Event()
                self._cancel[session_id] = flag
            return flag

    def _run_pipeline(self, session_id: str, trigger: Event) -> None:
        cancel = self._cancel_flag(session_id)
        cancel.set()  # cancel any previous
        cancel = threading.Event()
        with self._lock:
            self._cancel[session_id] = cancel

        state = self.sessions.require(session_id)
        t0 = time.time()
        self.sessions.set_worker_status(session_id, "planning", "Planning tasks")
        plan = planner.plan(state)
        plan_id = plan["plan_id"]
        self.sessions.set_worker_status(session_id, "planning", "Plan ready", plan_id=plan_id)
        self.logs.log_trace(session_id, "planner", "plan_created", plan, latency_ms=(time.time() - t0) * 1000, plan_id=plan_id)
        self.bus.emit(
            Event(
                type=EventType.WORKER_PLAN_CREATED,
                session_id=session_id,
                payload=plan,
            )
        )
        self.bus.emit(
            Event(
                type=EventType.WORKER_STATUS,
                session_id=session_id,
                payload={"status": "searching", "message": "Retrieving catalog matches", "plan_id": plan_id},
            )
        )
        if cancel.is_set():
            return

        t1 = time.time()
        self.sessions.set_worker_status(session_id, "searching", "Retrieving text and visual matches", plan_id=plan_id)
        try:
            catalog = self.catalog_fn()
            # Text and visual retrieval are independent worker tasks. Run them
            # together, then merge their candidates before recommendation.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="catalog-retrieval") as pool:
                text_future = pool.submit(
                    retrieve_text,
                    state.preference,
                    catalog,
                    query_hint=plan.get("query_hint"),
                    limit=80,
                )
                visual_future = pool.submit(
                    retrieve_visual,
                    state.preference,
                    catalog,
                    state.image_refs,
                    limit=80,
                )
                text_candidates = text_future.result()
                visual_candidates = visual_future.result()
            candidates = merge_candidates(
                text_candidates,
                visual_candidates,
                limit=80,
            )
        except Exception as exc:
            self.logs.log_trace(session_id, "retrieval", "error", {"error": str(exc)}, plan_id=plan_id)
            self.sessions.set_worker_status(session_id, "idle", f"Retrieval failed: {exc}", plan_id=plan_id)
            return
        self.logs.log_trace(
            session_id,
            "retrieval",
            "candidates_ready",
            {
                "count": len(candidates),
                "text_count": len(text_candidates),
                "visual_count": len(visual_candidates),
            },
            latency_ms=(time.time() - t1) * 1000,
            plan_id=plan_id,
        )
        self.bus.emit(
            Event(
                type=EventType.WORKER_CANDIDATES_READY,
                session_id=session_id,
                payload={
                    "plan_id": plan_id,
                    "count": len(candidates),
                    "text_count": len(text_candidates),
                    "visual_count": len(visual_candidates),
                },
            )
        )
        if cancel.is_set():
            return

        t2 = time.time()
        self.sessions.set_worker_status(session_id, "recommending", "Ranking matches", plan_id=plan_id)
        bundle = rank_products(plan_id, state.preference, candidates)
        if cancel.is_set():
            return
        self.sessions.set_bundle(session_id, bundle)
        self.logs.log_trace(
            session_id,
            "recommend",
            "recommendation_ready",
            {"top": [r.id for r in bundle.ranked], "summary": bundle.summary},
            latency_ms=(time.time() - t2) * 1000,
            plan_id=plan_id,
        )
        self.bus.emit(
            Event(
                type=EventType.WORKER_RECOMMENDATION_READY,
                session_id=session_id,
                payload=bundle.to_dict(),
            )
        )
        self.bus.emit(
            Event(
                type=EventType.WORKER_STATUS,
                session_id=session_id,
                payload={"status": "ready", "message": bundle.summary, "plan_id": plan_id},
            )
        )
