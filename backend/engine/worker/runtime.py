from __future__ import annotations

import threading
import time
from typing import Any, Callable

from ..bus import EventBus
from ..events import Event, EventType
from ..exp_config import load_config
from ..logging_store import LoggingStore
from ..memory_store import MEMORY_STORE, MemoryStore
from ..session import SessionStore
from . import planner
from .executor import Executor
from .recall_worker import RecallAgent


class WorkerRuntime:
    """Planner (control plane) + Executor (data plane) on the shared EventBus."""

    def __init__(
        self,
        bus: EventBus,
        sessions: SessionStore,
        logs: LoggingStore,
        search_fn: Callable[[dict], list[dict]],
        *,
        memory: MemoryStore | None = None,
    ) -> None:
        self.bus = bus
        self.sessions = sessions
        self.logs = logs
        self.search_fn = search_fn
        self.memory = memory or MEMORY_STORE
        self.executor = Executor(search_fn, recall_agent=RecallAgent(search_fn))
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        bus.subscribe(EventType.USER_INTENT_UPDATED, self._on_user_signal)
        bus.subscribe(EventType.SESSION_INTERRUPTED, self._on_interrupted)
        bus.subscribe(EventType.MEMORY_WRITE_REQUESTED, self._on_memory_write)

    def start(self) -> None:
        print("[worker] runtime listening for session events (planner+executor)")

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
        threading.Thread(
            target=self._run_pipeline,
            args=(event.session_id, event),
            name=f"worker-{event.session_id[:8]}",
            daemon=True,
        ).start()

    def _on_memory_write(self, event: Event) -> None:
        cfg = load_config()
        if not cfg.memory:
            return
        state = self.sessions.require(event.session_id)
        payload = event.payload or {}
        writes = payload.get("memory_write") or []
        meta = self.memory.apply_writes(
            session=state,
            writes=writes if isinstance(writes, list) else [],
            user_id=str(payload.get("user_id") or "") or None,
            opted_in_memory=bool(payload.get("opted_in_memory")),
            config=cfg,
        )
        self.logs.log_trace(event.session_id, "memory", "write", meta)
        self.bus.emit(
            Event(
                type=EventType.MEMORY_UPDATED,
                session_id=event.session_id,
                payload={
                    "memory_summary": state.memory_summary,
                    **meta,
                },
            )
        )

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

        cfg = load_config()
        state = self.sessions.require(session_id)
        payload = trigger.payload or {}
        utterance = str(payload.get("raw_query") or payload.get("utterance") or "")
        user_id = str(payload.get("user_id") or "") or None
        opted_in = bool(payload.get("opted_in_memory"))

        attempt = 0
        replan_ctx: dict[str, Any] | None = None
        max_attempts = 1 + (cfg.max_replans if cfg.planner_replan else 0)

        while attempt < max_attempts:
            if cancel.is_set():
                return

            t0 = time.time()
            self.sessions.set_worker_status(session_id, "planning", "Planning tasks")
            plan = planner.plan(
                state,
                utterance=utterance or None,
                user_id=user_id,
                opted_in_memory=opted_in,
                replan_ctx=replan_ctx,
                memory=self.memory,
                config=cfg,
            )
            plan_id = plan["plan_id"]
            self.sessions.set_worker_status(
                session_id, "planning", "Plan ready", plan_id=plan_id
            )
            self.logs.log_trace(
                session_id,
                "planner",
                "plan_created",
                plan,
                latency_ms=(time.time() - t0) * 1000,
                plan_id=plan_id,
            )
            self.bus.emit(
                Event(
                    type=EventType.WORKER_PLAN_CREATED,
                    session_id=session_id,
                    payload=plan,
                )
            )

            if plan.get("refused"):
                self.sessions.set_worker_status(
                    session_id, "idle", plan.get("message") or "refused", plan_id=plan_id
                )
                self.bus.emit(
                    Event(
                        type=EventType.WORKER_STATUS,
                        session_id=session_id,
                        payload={
                            "status": "refused",
                            "message": plan.get("message") or "refused",
                            "plan_id": plan_id,
                        },
                    )
                )
                return

            if cancel.is_set():
                return

            self.sessions.set_worker_status(
                session_id, "searching", "Executing plan", plan_id=plan_id
            )
            self.bus.emit(
                Event(
                    type=EventType.WORKER_STATUS,
                    session_id=session_id,
                    payload={
                        "status": "searching",
                        "message": "Executing recall/verify/recommend",
                        "plan_id": plan_id,
                    },
                )
            )

            t1 = time.time()

            def _memory_builder(sess, bundle):
                return self.memory.build_summary_writes(
                    sess, bundle, use_llm=bool(cfg.planner_llm and cfg.memory)
                )

            exec_result = self.executor.run(
                plan,
                state,
                cancel=cancel,
                config=cfg,
                memory_write_builder=_memory_builder,
            )
            self.logs.log_trace(
                session_id,
                "executor",
                "finished",
                exec_result.to_dict(),
                latency_ms=(time.time() - t1) * 1000,
                plan_id=plan_id,
            )

            if cancel.is_set():
                return

            if exec_result.need_replan:
                self.logs.log_trace(
                    session_id,
                    "replan",
                    "verify_reject",
                    {
                        "attempt": attempt,
                        "reason": exec_result.reject_reason,
                    },
                    plan_id=plan_id,
                )
                attempt += 1
                replan_ctx = {
                    "attempt": attempt,
                    "reject_reason": exec_result.reject_reason,
                    "relax_ops": (plan.get("hints") or {}).get("relax_ops")
                    or ["drop_hard", "raise_budget_20", "broaden_keywords"],
                }
                continue

            if not exec_result.ok or not exec_result.bundle:
                self.sessions.set_worker_status(
                    session_id,
                    "idle",
                    exec_result.error or "Execution failed",
                    plan_id=plan_id,
                )
                return

            self.bus.emit(
                Event(
                    type=EventType.WORKER_CANDIDATES_READY,
                    session_id=session_id,
                    payload={
                        "plan_id": plan_id,
                        "count": len(exec_result.candidates),
                        "rejected": len(exec_result.rejected),
                    },
                )
            )

            bundle = exec_result.bundle
            self.sessions.set_bundle(session_id, bundle)
            self.memory.remember_bundle(session_id, bundle)
            self.memory.sync_preference_slots(session_id, state.preference)
            self.logs.log_trace(
                session_id,
                "recommend",
                "recommendation_ready",
                {"top": [r.id for r in bundle.ranked], "summary": bundle.summary},
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
                    payload={
                        "status": "ready",
                        "message": bundle.summary,
                        "plan_id": plan_id,
                    },
                )
            )

            if exec_result.memory_writes and cfg.memory:
                self.bus.emit(
                    Event(
                        type=EventType.MEMORY_WRITE_REQUESTED,
                        session_id=session_id,
                        payload={
                            "memory_write": exec_result.memory_writes,
                            "user_id": user_id or "",
                            "opted_in_memory": opted_in,
                            "plan_id": plan_id,
                        },
                    )
                )
            return

        # Exhausted replans
        self.sessions.set_worker_status(
            session_id, "idle", "No matches after replan attempts"
        )
