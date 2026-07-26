#!/usr/bin/env python3
"""Evaluate end-to-end recommendation quality + latency on catalog-grounded cases.

Metrics: Hit@K, MRR, Recall@pool, CSR, Empty rate, Refuse Acc, Planner/E2E latency.
Uses real catalog.db via server.search (unlike planner routing eval's dummy catalog).

Usage (from repo root):
  py -3 scripts/build_recommend_cases.py
  py -3 scripts/eval_recommend_quality.py --no-llm --limit 5
  py -3 scripts/eval_recommend_quality.py --variants full,fixed --no-llm
  py -3 scripts/eval_recommend_quality.py --variants full,no_verifier,no_enrichment
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

DEFAULT_CASES = BACKEND / "data" / "eval" / "recommend_cases.jsonl"
DEFAULT_OUT_DIR = BACKEND / "data" / "eval"

VARIANT_ENV: dict[str, dict[str, str]] = {
    "full": {
        "VS_MODALITY_ROUTING": "1",
        "VS_INTENT_SHORTCIRCUIT": "1",
        "VS_MEMORY": "1",
        "VS_PLANNER_REPLAN": "1",
        "VS_PLANNER_LLM": "1",
        "VS_VISUAL_RECALL": "1",
        "VS_ENRICHMENT": "1",
        "VS_TEXT_SEMANTIC": "1",
        "VS_REVIEWS": "1",
        "VS_VERIFIER": "rule",
    },
    "fixed": {
        "VS_MODALITY_ROUTING": "0",
        "VS_INTENT_SHORTCIRCUIT": "0",
        "VS_MEMORY": "0",
        "VS_PLANNER_REPLAN": "0",
        "VS_PLANNER_LLM": "0",
        "VS_VISUAL_RECALL": "0",
        "VS_ENRICHMENT": "1",
        "VS_TEXT_SEMANTIC": "1",
        "VS_REVIEWS": "0",
        "VS_VERIFIER": "rule",
    },
    "no_enrichment": {
        "VS_MODALITY_ROUTING": "1",
        "VS_INTENT_SHORTCIRCUIT": "1",
        "VS_MEMORY": "1",
        "VS_PLANNER_REPLAN": "1",
        "VS_PLANNER_LLM": "1",
        "VS_VISUAL_RECALL": "1",
        "VS_ENRICHMENT": "0",
        "VS_TEXT_SEMANTIC": "0",
        "VS_REVIEWS": "0",
        "VS_VERIFIER": "rule",
    },
    "no_verifier": {
        "VS_MODALITY_ROUTING": "1",
        "VS_INTENT_SHORTCIRCUIT": "1",
        "VS_MEMORY": "1",
        "VS_PLANNER_REPLAN": "0",
        "VS_PLANNER_LLM": "1",
        "VS_VISUAL_RECALL": "1",
        "VS_ENRICHMENT": "1",
        "VS_TEXT_SEMANTIC": "1",
        "VS_REVIEWS": "1",
        "VS_VERIFIER": "off",
    },
    "no_modality": {
        "VS_MODALITY_ROUTING": "0",
        "VS_INTENT_SHORTCIRCUIT": "1",
        "VS_MEMORY": "1",
        "VS_PLANNER_REPLAN": "1",
        "VS_PLANNER_LLM": "1",
        "VS_VISUAL_RECALL": "0",
        "VS_ENRICHMENT": "1",
        "VS_TEXT_SEMANTIC": "1",
        "VS_REVIEWS": "1",
        "VS_VERIFIER": "rule",
    },
}


def _apply_env(variant: str, *, no_llm: bool) -> None:
    for k, v in VARIANT_ENV[variant].items():
        os.environ[k] = v
    if no_llm:
        os.environ["VS_PLANNER_LLM"] = "0"
    from engine import exp_config

    exp_config.CONFIG = exp_config.load_config()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
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
    state = SessionState(session_id=f"rec-eval-{case['case_id']}", preference=pref)
    state.conversation = [{"role": "user", "text": case.get("utterance") or ""}]
    if case.get("has_image"):
        img = case.get("image_path") or "eval_dummy.jpg"
        state.image_refs = [{"path": img, "mime_type": "image/jpeg"}]
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


def _parse_ks(raw: str) -> list[int]:
    ks: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ks.append(int(part))
    return ks or [1, 3, 5]


def _hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> bool:
    if not relevant:
        return False
    return any(pid in relevant for pid in ranked_ids[:k])


def _mrr(ranked_ids: list[str], primary_id: str | None) -> float:
    if not primary_id:
        return 0.0
    try:
        idx = ranked_ids.index(primary_id)
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0


def _recall_at_pool(cand_ids: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    hit = sum(1 for rid in relevant if rid in set(cand_ids))
    return hit / len(relevant)


def _check_constraint(item: dict[str, Any], rule: str) -> bool:
    """Support budget_le:N (price==0 treated as unknown → fail closed for CSR)."""
    rule = (rule or "").strip()
    if rule.startswith("budget_le:"):
        try:
            limit = int(rule.split(":", 1)[1])
        except ValueError:
            return True
        price = int(item.get("price") or 0)
        if price <= 0:
            return False
        return price <= limit
    return True


def _csr(ranked_items: list[dict[str, Any]], rules: list[str], k: int) -> float | None:
    if not rules:
        return None
    top = ranked_items[:k]
    if not top:
        return 0.0
    ok = 0
    for item in top:
        if all(_check_constraint(item, r) for r in rules):
            ok += 1
    return ok / len(top)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    # nearest-rank
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _run_one(
    case: dict[str, Any],
    *,
    variant: str,
    search_fn,
    ks: list[int],
    max_replans: int,
) -> dict[str, Any]:
    from engine.exp_config import load_config
    from engine.memory_store import MemoryStore
    from engine.worker import planner
    from engine.worker.executor import Executor

    cfg = load_config()
    state = _session_from_case(case)
    mem = MemoryStore(db_path=str(BACKEND / "data" / "eval" / f"_rec_mem_{variant}.db"))
    if state.worker.last_bundle:
        mem.remember_bundle(state.session_id, state.worker.last_bundle)

    gold = case.get("gold") or {}
    should_refuse = bool(gold.get("should_refuse"))
    relevant = {str(x) for x in (gold.get("relevant_ids") or []) if x}
    primary_id = gold.get("primary_id")
    primary_id = str(primary_id) if primary_id else None
    must_satisfy = list(gold.get("must_satisfy") or [])
    csr_k = max(ks) if ks else 3

    t0 = time.perf_counter()
    plan = planner.plan(
        state,
        utterance=case.get("utterance") or "",
        memory=mem,
        config=cfg,
    )
    planner_ms = (time.perf_counter() - t0) * 1000.0

    pred_refuse = bool(plan.get("refused"))
    pred_intent = str((plan.get("decision") or {}).get("intent") or "")
    pred_sc = bool((plan.get("decision") or {}).get("shortcircuit"))
    pred_stages = [str(s.get("agent") or "") for s in (plan.get("stages") or [])]
    refuse_ok = pred_refuse == should_refuse

    ranked_ids: list[str] = []
    ranked_items: list[dict[str, Any]] = []
    cand_ids: list[str] = []
    e2e_ms = planner_ms
    exec_ok = True
    empty = False
    replans = 0
    error = ""
    stage_trace: list[dict[str, Any]] = []

    if should_refuse or pred_refuse:
        # Safety path: do not run executor when refused.
        empty = True
        exec_ok = pred_refuse if should_refuse else False
    else:
        ex = Executor(search_fn)
        t1 = time.perf_counter()
        result = ex.run(plan, state, config=cfg)
        attempt = 0
        while (
            result.need_replan
            and cfg.planner_replan
            and attempt < max_replans
        ):
            attempt += 1
            replans = attempt
            plan = planner.plan(
                state,
                utterance=case.get("utterance") or "",
                memory=mem,
                config=cfg,
                replan_ctx={
                    "attempt": attempt,
                    "reject_reason": result.reject_reason,
                    "relax_ops": (plan.get("hints") or {}).get("relax_ops")
                    or ["drop_hard", "raise_budget_20", "broaden_keywords"],
                },
            )
            result = ex.run(plan, state, config=cfg)
        e2e_ms = (time.perf_counter() - t1) * 1000.0 + planner_ms
        exec_ok = bool(result.ok)
        error = str(result.error or "")
        stage_trace = list(result.stage_trace or [])
        cand_ids = [str(c.get("id") or "") for c in (result.candidates or []) if c.get("id")]
        if result.bundle and result.bundle.ranked:
            ranked_items = [
                {"id": r.id, "name": r.name, "price": r.price, "score": r.score}
                for r in result.bundle.ranked
            ]
            ranked_ids = [str(r["id"]) for r in ranked_items]
            if state.worker.last_bundle is None:
                state.worker.last_bundle = result.bundle
            mem.remember_bundle(state.session_id, result.bundle)
        empty = len(ranked_ids) == 0

    mem.close()

    hit: dict[str, bool | None] = {}
    for k in ks:
        key = f"hit@{k}"
        if should_refuse:
            hit[key] = None
        else:
            hit[key] = _hit_at_k(ranked_ids, relevant, k)

    mrr = None if should_refuse else _mrr(ranked_ids, primary_id)
    recall_pool = None if should_refuse else _recall_at_pool(cand_ids, relevant)
    csr = None if should_refuse else _csr(ranked_items, must_satisfy, csr_k)

    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "bucket": case["bucket"],
        "variant": variant,
        "utterance": case.get("utterance"),
        "should_refuse": should_refuse,
        "pred_refuse": pred_refuse,
        "refuse_ok": refuse_ok,
        "pred_intent": pred_intent,
        "pred_shortcircuit": pred_sc,
        "pred_stages": pred_stages,
        "ranked_ids": ranked_ids[:10],
        "candidate_count": len(cand_ids),
        "empty": empty if not should_refuse else False,
        "exec_ok": exec_ok,
        "replans": replans,
        "error": error,
        "planner_ms": round(planner_ms, 2),
        "e2e_ms": round(e2e_ms, 2),
        "mrr": mrr,
        "recall_at_pool": recall_pool,
        "csr": csr,
        "must_satisfy": must_satisfy,
        "relevant_ids": sorted(relevant),
        "primary_id": primary_id,
        "stage_trace": stage_trace,
    }
    row.update(hit)
    return row


def _agg(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    quality_rows = [r for r in rows if not r.get("should_refuse")]
    safety_rows = [r for r in rows if r.get("should_refuse")]
    n_q = len(quality_rows)

    def hit_rate(k: int) -> float | None:
        if not quality_rows:
            return None
        key = f"hit@{k}"
        return sum(1 for r in quality_rows if r.get(key)) / n_q

    def mean_of(key: str) -> float | None:
        vals = [float(r[key]) for r in quality_rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    csr_rows = [r for r in quality_rows if r.get("csr") is not None]
    planner_lat = [float(r["planner_ms"]) for r in rows]
    e2e_lat = [float(r["e2e_ms"]) for r in rows]

    by_bucket: dict[str, Any] = {}
    for b in sorted({r["bucket"] for r in rows}):
        br = [r for r in rows if r["bucket"] == b]
        bq = [r for r in br if not r.get("should_refuse")]
        bn = len(bq) or 1
        bucket_stats: dict[str, Any] = {
            "n": len(br),
            "n_quality": len(bq),
            "empty_rate": (
                sum(1 for r in bq if r.get("empty")) / len(bq) if bq else None
            ),
            "refuse_acc": (
                sum(1 for r in br if r.get("should_refuse") and r.get("refuse_ok"))
                / max(1, sum(1 for r in br if r.get("should_refuse")))
                if any(r.get("should_refuse") for r in br)
                else None
            ),
            "e2e_ms_p50": statistics.median([r["e2e_ms"] for r in br]),
        }
        for k in ks:
            key = f"hit@{k}"
            bucket_stats[key] = (
                sum(1 for r in bq if r.get(key)) / bn if bq else None
            )
        by_bucket[b] = bucket_stats

    out: dict[str, Any] = {
        "n": len(rows),
        "n_quality": n_q,
        "n_safety": len(safety_rows),
        "refuse_acc": (
            sum(1 for r in safety_rows if r.get("refuse_ok")) / len(safety_rows)
            if safety_rows
            else None
        ),
        "empty_rate": (
            sum(1 for r in quality_rows if r.get("empty")) / n_q if n_q else None
        ),
        "mrr": mean_of("mrr"),
        "recall_at_pool": mean_of("recall_at_pool"),
        "csr": (
            statistics.mean([float(r["csr"]) for r in csr_rows]) if csr_rows else None
        ),
        "csr_n": len(csr_rows),
        "planner_ms_p50": _percentile(planner_lat, 50),
        "planner_ms_p95": _percentile(planner_lat, 95),
        "planner_ms_mean": statistics.mean(planner_lat) if planner_lat else None,
        "e2e_ms_p50": _percentile(e2e_lat, 50),
        "e2e_ms_p95": _percentile(e2e_lat, 95),
        "e2e_ms_mean": statistics.mean(e2e_lat) if e2e_lat else None,
        "by_bucket": by_bucket,
    }
    for k in ks:
        out[f"hit@{k}"] = hit_rate(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval recommend quality + latency")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--variants",
        type=str,
        default="full,fixed",
        help="Comma-separated: full,fixed,no_enrichment,no_verifier,no_modality",
    )
    ap.add_argument("--k", type=str, default="1,3,5", help="Hit@K cutoffs")
    ap.add_argument("--no-llm", action="store_true", help="Force VS_PLANNER_LLM=0")
    ap.add_argument("--limit", type=int, default=0, help="Only first N cases")
    ap.add_argument("--max-replans", type=int, default=1, help="Verify→replan attempts")
    args = ap.parse_args()
    ks = _parse_ks(args.k)

    if not args.cases.is_file():
        print(f"[eval] cases not found: {args.cases}")
        print("[eval] run: py -3 scripts/build_recommend_cases.py")
        return 1

    # Ensure backend cwd semantics for relative data paths inside server/enrichment.
    os.chdir(BACKEND)
    import engine.exp_config  # noqa: F401
    import server as srv

    srv.DB_PATH = str(BACKEND / "data" / "catalog.db")
    search_fn = srv.search
    print(f"[eval] catalog={srv.DB_PATH}")

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
        print(f"\n[eval] === variant={variant} cases={len(cases)} no_llm={args.no_llm} ===")
        _apply_env(variant, no_llm=args.no_llm)
        rows: list[dict[str, Any]] = []
        for i, case in enumerate(cases, 1):
            row = _run_one(
                case,
                variant=variant,
                search_fn=search_fn,
                ks=ks,
                max_replans=args.max_replans,
            )
            rows.append(row)
            all_rows.append(row)
            hit3 = row.get("hit@3")
            mark = (
                "REF"
                if row.get("should_refuse")
                else ("HIT" if hit3 else ("EMP" if row.get("empty") else "MISS"))
            )
            print(
                f"  [{i}/{len(cases)}] {case['case_id']} {mark} "
                f"intent={row['pred_intent']} sc={int(row['pred_shortcircuit'])} "
                f"top={row['ranked_ids'][:3]} "
                f"e2e={row['e2e_ms']:.0f}ms"
            )
        summaries[variant] = _agg(rows, ks)

    results_path = args.out_dir / "recommend_quality_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.out_dir / "recommend_quality_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "no_llm": args.no_llm,
                "n_cases": len(cases),
                "ks": ks,
                "variants": summaries,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = args.out_dir / "recommend_quality_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "variant",
            "n",
            "n_quality",
            *[f"hit@{k}" for k in ks],
            "mrr",
            "recall_at_pool",
            "csr",
            "empty_rate",
            "refuse_acc",
            "planner_ms_p50",
            "e2e_ms_p50",
            "e2e_ms_p95",
        ]
        w.writerow(header)
        for variant, s in summaries.items():
            w.writerow(
                [
                    variant,
                    s["n"],
                    s["n_quality"],
                    *[
                        "" if s.get(f"hit@{k}") is None else f"{s[f'hit@{k}']:.4f}"
                        for k in ks
                    ],
                    "" if s["mrr"] is None else f"{s['mrr']:.4f}",
                    "" if s["recall_at_pool"] is None else f"{s['recall_at_pool']:.4f}",
                    "" if s["csr"] is None else f"{s['csr']:.4f}",
                    "" if s["empty_rate"] is None else f"{s['empty_rate']:.4f}",
                    "" if s["refuse_acc"] is None else f"{s['refuse_acc']:.4f}",
                    s["planner_ms_p50"],
                    s["e2e_ms_p50"],
                    s["e2e_ms_p95"],
                ]
            )

    print("\n======== SUMMARY ========")
    for variant, s in summaries.items():
        hits = " ".join(
            f"hit@{k}={s.get(f'hit@{k}')}" for k in ks
        )
        print(
            f"{variant}: {hits} mrr={s['mrr']} csr={s['csr']} "
            f"empty={s['empty_rate']} refuse={s['refuse_acc']} "
            f"e2e_p50={s['e2e_ms_p50']} e2e_p95={s['e2e_ms_p95']}"
        )
    print(f"[eval] results -> {results_path}")
    print(f"[eval] summary -> {summary_path}")
    print(f"[eval] csv     -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
