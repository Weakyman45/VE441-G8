#!/usr/bin/env python3
"""Evaluate Planner routing correctness + latency (Full vs Fixed).

Does NOT compute Hit@K. Optional --e2e also runs Executor for end-to-end latency only.

Usage (from repo root):
  py -3 scripts/build_planner_cases.py
  py -3 scripts/eval_planner_routing.py
  py -3 scripts/eval_planner_routing.py --e2e
  py -3 scripts/eval_planner_routing.py --variants full,fixed --no-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

DEFAULT_CASES = BACKEND / "data" / "eval" / "planner_cases.jsonl"
DEFAULT_OUT_DIR = BACKEND / "data" / "eval"

VARIANT_ENV: dict[str, dict[str, str]] = {
    "full": {
        "VS_MODALITY_ROUTING": "1",
        "VS_INTENT_SHORTCIRCUIT": "1",
        "VS_MEMORY": "1",
        "VS_PLANNER_REPLAN": "1",
        "VS_PLANNER_LLM": "1",
        "VS_VISUAL_RECALL": "1",
        "VS_VERIFIER": "rule",
    },
    "fixed": {
        "VS_MODALITY_ROUTING": "0",
        "VS_INTENT_SHORTCIRCUIT": "0",
        "VS_MEMORY": "0",
        "VS_PLANNER_REPLAN": "0",
        "VS_PLANNER_LLM": "0",
        "VS_VISUAL_RECALL": "0",
        "VS_VERIFIER": "rule",
    },
}


def _apply_env(variant: str, *, no_llm: bool) -> None:
    for k, v in VARIANT_ENV[variant].items():
        os.environ[k] = v
    if no_llm:
        os.environ["VS_PLANNER_LLM"] = "0"
    # Force re-read
    from engine import exp_config

    exp_config.CONFIG = exp_config.load_config()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _session_from_case(case: dict[str, Any]):
    from engine.models import (
        PreferenceProfile,
        RankedProduct,
        RecommendationBundle,
        SessionState,
        WorkerState,
    )

    pref_raw = case.get("preference") or {}
    pref = PreferenceProfile(
        category=str(pref_raw.get("category") or ""),
        budget=pref_raw.get("budget"),
        use_case=str(pref_raw.get("use_case") or ""),
        platform=str(pref_raw.get("platform") or "No preference"),
        touch=str(pref_raw.get("touch") or "Not required"),
        hard=list(pref_raw.get("hard") or []),
        soft=list(pref_raw.get("soft") or []),
        visual_context=str(pref_raw.get("visual_context") or ""),
        raw_query=str(pref_raw.get("raw_query") or case.get("utterance") or ""),
        search_keywords=list(pref_raw.get("search_keywords") or []),
    )
    state = SessionState(session_id=f"eval-{case['case_id']}", preference=pref)
    state.conversation = [{"role": "user", "text": case.get("utterance") or ""}]
    if case.get("has_image"):
        state.image_refs = [{"path": "eval_dummy.jpg", "mime_type": "image/jpeg"}]
    lb = case.get("last_bundle")
    if lb and lb.get("ranked"):
        ranked = [
            RankedProduct(
                id=str(r.get("id") or ""),
                name=str(r.get("name") or ""),
                price=int(r.get("price") or 0),
                score=int(r.get("score") or 0),
                reasons=list(r.get("reasons") or []),
                summary=str(r.get("summary") or ""),
                rating=float(r.get("rating") or 0),
                platform=str(r.get("platform") or "Windows"),
                display=str(r.get("display") or ""),
                performance=str(r.get("performance") or ""),
                weight_kg=float(r.get("weight_kg") or 0),
                image_url=str(r.get("image_url") or ""),
            )
            for r in lb["ranked"]
        ]
        state.worker = WorkerState(
            plan_id=str(lb.get("plan_id") or "seed"),
            status="ready",
            last_bundle=RecommendationBundle(
                plan_id=str(lb.get("plan_id") or "seed"),
                ranked=ranked,
                summary=str(lb.get("summary") or ""),
                status="ready",
            ),
            message=str(lb.get("summary") or ""),
        )
    return state


def _expected_stages(case: dict[str, Any], *, visual_ready: bool) -> list[str]:
    gold = case.get("gold") or {}
    stages = list(gold.get("stages") or [])
    if not visual_ready:
        stages = [s for s in stages if s != "visual_recall"]
        # image-only with no visual -> text_recall safety net in planner
        if not stages and gold.get("should_refuse"):
            return []
        if stages == ["verify", "recommend"]:
            stages = ["text_recall", "verify", "recommend"]
        if not stages and not gold.get("should_refuse"):
            stages = ["text_recall", "verify", "recommend"]
    return stages


def _pred_stages(plan: dict[str, Any]) -> list[str]:
    return [str(s.get("agent") or "") for s in (plan.get("stages") or [])]


def _run_one(
    case: dict[str, Any],
    *,
    variant: str,
    e2e: bool,
    search_fn,
    visual_ready: bool,
) -> dict[str, Any]:
    from engine.exp_config import load_config
    from engine.memory_store import MemoryStore
    from engine.worker import planner
    from engine.worker.executor import Executor

    cfg = load_config()
    state = _session_from_case(case)
    mem = MemoryStore(db_path=str(BACKEND / "data" / "eval" / f"_mem_{variant}.db"))
    # Seed last_ranked for ref resolution
    if state.worker.last_bundle:
        mem.remember_bundle(state.session_id, state.worker.last_bundle)

    t0 = time.perf_counter()
    plan = planner.plan(
        state,
        utterance=case.get("utterance") or "",
        memory=mem,
        config=cfg,
    )
    planner_ms = (time.perf_counter() - t0) * 1000.0

    gold = case.get("gold") or {}
    pred_intent = str((plan.get("decision") or {}).get("intent") or "")
    pred_sc = bool((plan.get("decision") or {}).get("shortcircuit"))
    pred_refuse = bool(plan.get("refused"))
    pred_refs = list((plan.get("decision") or {}).get("memory", {}).get("resolved_refs") or [])
    pred_stages = _pred_stages(plan)
    exp_stages = _expected_stages(case, visual_ready=visual_ready)

    # Intent: refuse cases gold_intent is "refuse"
    gold_intent = str(gold.get("intent") or "")
    if gold.get("should_refuse"):
        intent_ok = pred_refuse and pred_intent in ("refuse", gold_intent)
    else:
        intent_ok = (not pred_refuse) and pred_intent == gold_intent

    sc_ok = pred_sc == bool(gold.get("should_shortcircuit"))
    refuse_ok = pred_refuse == bool(gold.get("should_refuse"))

    gold_ref = gold.get("ref_product_id")
    if gold_ref:
        ref_ok = gold_ref in pred_refs
        ref_applicable = True
    else:
        ref_ok = None
        ref_applicable = False

    dag_ok = pred_stages == exp_stages

    e2e_ms = None
    e2e_ok = None
    if e2e and not pred_refuse:
        ex = Executor(search_fn)
        t1 = time.perf_counter()
        result = ex.run(plan, state, config=cfg)
        e2e_ms = (time.perf_counter() - t1) * 1000.0 + planner_ms
        e2e_ok = bool(result.ok)
    elif e2e and pred_refuse:
        e2e_ms = planner_ms
        e2e_ok = True

    mem.close()
    return {
        "case_id": case["case_id"],
        "bucket": case["bucket"],
        "variant": variant,
        "utterance": case.get("utterance"),
        "gold_intent": gold_intent,
        "pred_intent": pred_intent,
        "intent_ok": intent_ok,
        "gold_shortcircuit": bool(gold.get("should_shortcircuit")),
        "pred_shortcircuit": pred_sc,
        "shortcircuit_ok": sc_ok,
        "gold_refuse": bool(gold.get("should_refuse")),
        "pred_refuse": pred_refuse,
        "refuse_ok": refuse_ok,
        "gold_ref": gold_ref,
        "pred_refs": pred_refs,
        "ref_ok": ref_ok,
        "ref_applicable": ref_applicable,
        "gold_stages": exp_stages,
        "pred_stages": pred_stages,
        "dag_ok": dag_ok,
        "planner_ms": round(planner_ms, 2),
        "e2e_ms": None if e2e_ms is None else round(e2e_ms, 2),
        "e2e_ok": e2e_ok,
        "planner_mode": plan.get("planner"),
        "query_hint": plan.get("query_hint"),
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / n

    ref_rows = [r for r in rows if r.get("ref_applicable")]
    ref_acc = (
        sum(1 for r in ref_rows if r.get("ref_ok")) / len(ref_rows) if ref_rows else None
    )
    planner_lat = [float(r["planner_ms"]) for r in rows]
    e2e_lat = [float(r["e2e_ms"]) for r in rows if r.get("e2e_ms") is not None]

    by_bucket: dict[str, Any] = {}
    buckets = sorted({r["bucket"] for r in rows})
    for b in buckets:
        br = [r for r in rows if r["bucket"] == b]
        bn = len(br) or 1
        bref = [r for r in br if r.get("ref_applicable")]
        by_bucket[b] = {
            "n": len(br),
            "intent_acc": sum(1 for r in br if r["intent_ok"]) / bn,
            "shortcircuit_acc": sum(1 for r in br if r["shortcircuit_ok"]) / bn,
            "refuse_acc": sum(1 for r in br if r["refuse_ok"]) / bn,
            "dag_acc": sum(1 for r in br if r["dag_ok"]) / bn,
            "ref_acc": (
                sum(1 for r in bref if r["ref_ok"]) / len(bref) if bref else None
            ),
            "planner_ms_p50": statistics.median([r["planner_ms"] for r in br]),
        }

    return {
        "n": len(rows),
        "intent_acc": rate("intent_ok"),
        "shortcircuit_acc": rate("shortcircuit_ok"),
        "refuse_acc": rate("refuse_ok"),
        "dag_acc": rate("dag_ok"),
        "ref_acc": ref_acc,
        "ref_n": len(ref_rows),
        "planner_ms_p50": statistics.median(planner_lat) if planner_lat else None,
        "planner_ms_p95": (
            statistics.quantiles(planner_lat, n=20)[18]
            if len(planner_lat) >= 20
            else (max(planner_lat) if planner_lat else None)
        ),
        "planner_ms_mean": statistics.mean(planner_lat) if planner_lat else None,
        "e2e_ms_p50": statistics.median(e2e_lat) if e2e_lat else None,
        "e2e_ms_mean": statistics.mean(e2e_lat) if e2e_lat else None,
        "by_bucket": by_bucket,
    }


def _dummy_search(params: dict) -> list[dict]:
    q = ((params.get("q") or [""])[0] or "").lower()
    catalog = [
        {
            "id": "prod_a",
            "name": "Alpha Running Shoes White",
            "price": 45,
            "rating": 4.6,
            "platform": "Windows",
            "summary": "running",
            "display": "",
            "performance": "",
            "weight_kg": 0.8,
            "image_url": "",
        },
        {
            "id": "prod_b",
            "name": "Beta Breathable Trainers",
            "price": 38,
            "rating": 4.4,
            "platform": "Windows",
            "summary": "breathable",
            "display": "",
            "performance": "",
            "weight_kg": 0.7,
            "image_url": "",
        },
        {
            "id": "prod_c",
            "name": "Gamma Trail Shoes",
            "price": 60,
            "rating": 4.3,
            "platform": "Windows",
            "summary": "trail",
            "display": "",
            "performance": "",
            "weight_kg": 0.9,
            "image_url": "",
        },
    ]
    hits = [c for c in catalog if any(t in c["name"].lower() for t in q.split() if len(t) > 2)]
    return hits or catalog


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval Planner routing + latency")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variants", type=str, default="full,fixed")
    ap.add_argument("--e2e", action="store_true", help="Also run Executor; record E2E latency")
    ap.add_argument("--no-llm", action="store_true", help="Force VS_PLANNER_LLM=0 for all variants")
    ap.add_argument("--use-catalog", action="store_true", help="Use server.search on catalog.db for --e2e")
    ap.add_argument("--limit", type=int, default=0, help="Only first N cases (debug)")
    args = ap.parse_args()

    if not args.cases.is_file():
        print(f"[eval] cases not found: {args.cases}")
        print("[eval] run: py -3 scripts/build_planner_cases.py")
        return 1

    # Load dotenv early
    import engine.exp_config  # noqa: F401
    from engine import enrichment

    visual_ready = bool(enrichment.has_enrichment())
    search_fn = _dummy_search
    if args.e2e and args.use_catalog:
        import server as srv

        srv.DB_PATH = str(BACKEND / "data" / "catalog.db")
        search_fn = srv.search
        print(f"[eval] e2e search=catalog ({srv.DB_PATH})")
    elif args.e2e:
        print("[eval] e2e search=dummy catalog")

    cases = _load_cases(args.cases)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_ENV:
            print(f"[eval] unknown variant {v}; choose from {list(VARIANT_ENV)}")
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for variant in variants:
        print(f"\n[eval] === variant={variant} cases={len(cases)} ===")
        _apply_env(variant, no_llm=args.no_llm)
        rows = []
        for i, case in enumerate(cases, 1):
            row = _run_one(
                case,
                variant=variant,
                e2e=args.e2e,
                search_fn=search_fn,
                visual_ready=visual_ready and variant == "full",
            )
            rows.append(row)
            all_rows.append(row)
            mark = "OK" if row["dag_ok"] and row["intent_ok"] else ".."
            print(
                f"  [{i}/{len(cases)}] {case['case_id']} {mark} "
                f"intent={row['pred_intent']} sc={row['pred_shortcircuit']} "
                f"stages={row['pred_stages']} {row['planner_ms']:.0f}ms"
            )
        summaries[variant] = _agg(rows)

    results_path = args.out_dir / "planner_routing_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.out_dir / "planner_routing_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "visual_ready": visual_ready,
                "e2e": args.e2e,
                "no_llm": args.no_llm,
                "n_cases": len(cases),
                "variants": summaries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = args.out_dir / "planner_routing_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "variant",
                "n",
                "intent_acc",
                "shortcircuit_acc",
                "ref_acc",
                "refuse_acc",
                "dag_acc",
                "planner_ms_p50",
                "planner_ms_mean",
                "e2e_ms_p50",
                "e2e_ms_mean",
            ]
        )
        for variant, s in summaries.items():
            w.writerow(
                [
                    variant,
                    s["n"],
                    f"{s['intent_acc']:.4f}",
                    f"{s['shortcircuit_acc']:.4f}",
                    "" if s["ref_acc"] is None else f"{s['ref_acc']:.4f}",
                    f"{s['refuse_acc']:.4f}",
                    f"{s['dag_acc']:.4f}",
                    s["planner_ms_p50"],
                    round(s["planner_ms_mean"], 2) if s["planner_ms_mean"] is not None else "",
                    s["e2e_ms_p50"] if s["e2e_ms_p50"] is not None else "",
                    round(s["e2e_ms_mean"], 2) if s["e2e_ms_mean"] is not None else "",
                ]
            )

    print("\n======== SUMMARY ========")
    for variant, s in summaries.items():
        print(
            f"{variant}: intent={s['intent_acc']:.3f} sc={s['shortcircuit_acc']:.3f} "
            f"ref={s['ref_acc']} refuse={s['refuse_acc']:.3f} dag={s['dag_acc']:.3f} "
            f"planner_p50={s['planner_ms_p50']}ms "
            f"e2e_p50={s['e2e_ms_p50']}"
        )
    print(f"[eval] results -> {results_path}")
    print(f"[eval] summary -> {summary_path}")
    print(f"[eval] csv     -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
