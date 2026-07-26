"""Online smoke-test: Planner → Executor on real catalog.db with LLM enabled."""
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

# Online defaults (can still override from shell / .env)
os.environ["VS_PLANNER_LLM"] = os.environ.get("VS_PLANNER_LLM") or "1"
os.environ["VS_VERIFIER"] = os.environ.get("VS_VERIFIER") or "llm"
os.environ["VS_VISUAL_RECALL"] = os.environ.get("VS_VISUAL_RECALL") or "1"
os.environ["VS_MEMORY"] = os.environ.get("VS_MEMORY") or "1"
os.environ["VS_INTENT_SHORTCIRCUIT"] = os.environ.get("VS_INTENT_SHORTCIRCUIT") or "1"
os.environ["VS_PLANNER_REPLAN"] = os.environ.get("VS_PLANNER_REPLAN") or "1"
os.environ["VS_MODALITY_ROUTING"] = os.environ.get("VS_MODALITY_ROUTING") or "1"

import sqlite3

from engine.bus import EventBus
from engine.events import Event, EventType
from engine.exp_config import load_config
from engine.llm.qwen_client import qwen_configured
from engine.logging_store import LoggingStore
from engine.memory_store import MemoryStore
from engine.models import PreferenceProfile, SessionState
from engine.session import SessionStore
from engine.worker import planner
from engine.worker.executor import Executor
from engine.worker.runtime import WorkerRuntime

import server as srv

DB = os.path.join(HERE, "data", "catalog.db")


def check_db() -> int:
    if not os.path.exists(DB):
        raise SystemExit(f"MISSING {DB}")
    conn = sqlite3.connect(DB)
    try:
        n = conn.execute("SELECT COUNT(*) FROM laptops").fetchone()[0]
        sample = conn.execute(
            "SELECT id, name, price FROM laptops LIMIT 3"
        ).fetchall()
    finally:
        conn.close()
    print(f"[db] {DB} rows={n}")
    for row in sample:
        print(f"     sample: {row}")
    return int(n)


def run_direct_pipeline() -> None:
    print("\n=== A) Online Planner + Executor (catalog search) ===")
    cfg = load_config()
    print("[cfg]", cfg.summary())
    print("[qwen_configured]", qwen_configured())
    if not qwen_configured():
        raise SystemExit(
            "DASHSCOPE_API_KEY missing. Put it in backend/.env then re-run."
        )

    state = SessionState(session_id="smoke1")
    state.preference = PreferenceProfile(
        category="shoes",
        search_keywords=["running", "shoes", "white"],
        raw_query="我想要白色透气跑鞋，预算150左右",
        budget=150,
        soft=["breathable"],
    )
    state.conversation = [
        {"role": "user", "text": "我想要白色透气跑鞋，预算150左右"},
    ]

    mem = MemoryStore(db_path=os.path.join(HERE, "data", "_smoke_memory.db"))
    t0 = time.time()
    plan = planner.plan(
        state,
        utterance="我想要白色透气跑鞋，预算150左右",
        memory=mem,
        config=cfg,
    )
    t_plan = (time.time() - t0) * 1000
    print(
        f"[plan] id={plan['plan_id']} planner={plan['planner']} "
        f"intent={plan['decision'].get('intent')} "
        f"stages={[s['agent'] for s in plan['stages']]} "
        f"query_hint={plan.get('query_hint')!r} "
        f"focus={plan.get('focus')} "
        f"refused={plan.get('refused')} ({t_plan:.0f} ms)"
    )
    if plan.get("refused"):
        raise SystemExit("unexpected refuse")

    ex = Executor(srv.search)
    t1 = time.time()
    result = ex.run(
        plan, state, config=cfg, memory_write_builder=mem.build_summary_writes
    )
    t_ex = (time.time() - t1) * 1000
    print(
        f"[exec] ok={result.ok} candidates={len(result.candidates)} "
        f"rejected={len(result.rejected)} need_replan={result.need_replan} "
        f"error={result.error!r} ({t_ex:.0f} ms)"
    )
    print(f"[exec] stage_trace={result.stage_trace}")
    if result.need_replan:
        print("[replan] verify rejected all; asking planner again...")
        plan_r = planner.plan(
            state,
            utterance="我想要白色透气跑鞋，预算150左右",
            memory=mem,
            config=cfg,
            replan_ctx={
                "attempt": 1,
                "reject_reason": result.reject_reason,
                "relax_ops": (plan.get("hints") or {}).get("relax_ops")
                or ["drop_hard", "raise_budget_20", "broaden_keywords"],
            },
        )
        result = ex.run(plan_r, state, config=cfg)
        print(
            f"[exec-replan] ok={result.ok} candidates={len(result.candidates)} "
            f"rejected={len(result.rejected)}"
        )
    if not result.ok or not result.bundle:
        raise SystemExit("executor failed")
    ranked = result.bundle.ranked[:5]
    print(f"[bundle] summary={result.bundle.summary!r}")
    for i, r in enumerate(ranked, 1):
        print(f"  #{i} {r.id} | {r.name} | ${r.price} | score={r.score}")
    mem.remember_bundle(state.session_id, result.bundle)
    state.worker.last_bundle = result.bundle

    print("\n=== B) Online follow-up (shortcircuit + ref) ===")
    state.conversation.append({"role": "user", "text": "第二个便宜一点"})
    state.preference.raw_query = "第二个便宜一点"
    t2 = time.time()
    plan2 = planner.plan(
        state,
        utterance="第二个便宜一点",
        memory=mem,
        config=cfg,
    )
    t_plan2 = (time.time() - t2) * 1000
    print(
        f"[plan2] planner={plan2['planner']} intent={plan2['decision'].get('intent')} "
        f"shortcircuit={plan2['decision'].get('shortcircuit')} "
        f"stages={[s['agent'] for s in plan2['stages']]} "
        f"refs={plan2['decision'].get('memory', {}).get('resolved_refs')} "
        f"({t_plan2:.0f} ms)"
    )
    result2 = ex.run(plan2, state, config=cfg)
    print(
        f"[exec2] ok={result2.ok} top="
        f"{[r.name for r in (result2.bundle.ranked[:3] if result2.bundle else [])]}"
    )
    mem.close()


def run_eventbus_runtime() -> None:
    print("\n=== C) EventBus WorkerRuntime (online) ===")
    import threading

    bus = EventBus()
    sessions = SessionStore()
    logs = LoggingStore(os.path.join(HERE, "data", "_smoke_logs.db"))
    mem = MemoryStore(db_path=os.path.join(HERE, "data", "_smoke_memory2.db"))
    ready = threading.Event()
    payload_box: dict = {}

    def on_ready(ev: Event) -> None:
        payload_box["bundle"] = ev.payload
        ready.set()

    bus.subscribe(EventType.WORKER_RECOMMENDATION_READY, on_ready)
    rt = WorkerRuntime(bus, sessions, logs, srv.search, memory=mem)
    rt.start()

    state = sessions.create("smoke_rt")
    state.preference = PreferenceProfile(
        category="shoes",
        search_keywords=["running", "shoes"],
        raw_query="推荐几双适合跑步的鞋",
        budget=120,
    )
    state.conversation = [{"role": "user", "text": "推荐几双适合跑步的鞋"}]
    bus.emit(
        Event(
            type=EventType.USER_INTENT_UPDATED,
            session_id="smoke_rt",
            payload=state.preference.to_dict(),
        )
    )
    if not ready.wait(120):
        raise SystemExit("timeout waiting for WORKER_RECOMMENDATION_READY")
    ranked = (payload_box.get("bundle") or {}).get("ranked") or []
    summary = (payload_box.get("bundle") or {}).get("summary")
    print(f"[runtime] got {len(ranked)} ranked; summary={summary!r}")
    for i, r in enumerate(ranked[:5], 1):
        print(f"  #{i} {r.get('id')} | {r.get('name')} | ${r.get('price')}")
    mem.close()


def main() -> int:
    srv.DB_PATH = DB
    n = check_db()
    if n <= 0:
        raise SystemExit("empty catalog")
    try:
        run_direct_pipeline()
        run_eventbus_runtime()
    except Exception:
        traceback.print_exc()
        return 1
    print("\n=== SMOKE OK (ONLINE): multi-agent + LLM on catalog.db ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
