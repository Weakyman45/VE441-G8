from __future__ import annotations

import threading
import time
from typing import Callable

from ..bus import EventBus
from ..events import Event, EventType
from ..logging_store import LoggingStore
from ..session import SessionStore
from ..exp_config import load_config
from ..modality_router import route
from ..verifier import verify_candidates
from . import planner
from .recall_worker import RecallAgent
from .recommend_worker import rank_products


class WorkerRuntime:
    """Background planner–worker loop listening on the shared EventBus."""

    def __init__(
        self,
        bus: EventBus,
        sessions: SessionStore,
        logs: LoggingStore,
        search_fn: Callable[[dict], list[dict]],
    ) -> None:
        self.bus = bus
        self.sessions = sessions
        self.logs = logs
        self.search_fn = search_fn
        self.recall_agent = RecallAgent(search_fn)
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

    @staticmethod
    def _load_query_image(state) -> tuple[bytes, str] | None:
        """读取该 session 最近一张上传图的字节 + mime(供视觉召回)。无则返回 None。"""
        import os
        for ref in reversed(getattr(state, "image_refs", None) or []):
            path = ref.get("path") if isinstance(ref, dict) else None
            if path and os.path.exists(path):
                try:
                    with open(path, "rb") as fh:
                        data = fh.read()
                    if data:
                        return data, (ref.get("mime_type") or "image/jpeg")
                except OSError:
                    continue
        return None

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
                payload={"status": "searching", "message": "Searching catalog", "plan_id": plan_id},
            )
        )
        if cancel.is_set():
            return

        # 创新点一:按输入模态规划本次检索路径(文本/图文/纯图片)
        cfg = load_config()
        route_plan = route(state, cfg)
        self.logs.log_trace(session_id, "router", "route_planned", route_plan.to_dict(), plan_id=plan_id)

        t1 = time.time()
        self.sessions.set_worker_status(session_id, "searching", "Searching products", plan_id=plan_id)
        query_image = self._load_query_image(state)

        # 召回 Agent:按模态计划编排 文本(关键词,含 enriched_text) + 图片(视觉) 两路召回并融合
        try:
            recall_result = self.recall_agent.recall(
                state=state,
                route_plan=route_plan,
                cfg=cfg,
                query_image=query_image,
            )
        except Exception as exc:
            self.logs.log_trace(session_id, "recall", "error", {"error": str(exc)}, plan_id=plan_id)
            self.sessions.set_worker_status(session_id, "idle", f"Recall failed: {exc}", plan_id=plan_id)
            return

        candidates = recall_result.candidates
        recall_stats = recall_result.to_dict()
        recall_stats["modality"] = route_plan.modality

        self.logs.log_trace(
            session_id, "recall", "candidates_ready", recall_stats,
            latency_ms=(time.time() - t1) * 1000, plan_id=plan_id,
        )
        self.bus.emit(
            Event(
                type=EventType.WORKER_CANDIDATES_READY,
                session_id=session_id,
                payload={"plan_id": plan_id, **recall_result.to_dict()},
            )
        )
        if cancel.is_set():
            return

        # 创新点三:约束感知校验(品类 + must-have),拒绝项带理由并入 excluded
        rejected: list[dict] = []
        if route_plan.do_verify:
            self.sessions.set_worker_status(session_id, "verifying", "Verifying constraints", plan_id=plan_id)
            vres = verify_candidates(state.preference, candidates, cfg)
            candidates = vres.kept
            rejected = vres.rejected
            self.logs.log_trace(session_id, "verifier", "verified", vres.to_dict(), plan_id=plan_id)
            if cancel.is_set():
                return

        t2 = time.time()
        self.sessions.set_worker_status(session_id, "recommending", "Ranking matches", plan_id=plan_id)
        bundle = rank_products(
            plan_id, state.preference, candidates,
            modality=route_plan.modality, extra_excluded=rejected,
        )
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
