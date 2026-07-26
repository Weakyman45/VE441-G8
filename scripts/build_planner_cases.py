#!/usr/bin/env python3
"""Construct a Planner routing evaluation set (decision gold, not Hit@K).

Usage (from repo root):
  py -3 scripts/build_planner_cases.py
  py -3 scripts/build_planner_cases.py --out backend/data/eval/planner_cases.jsonl --seed 42

Optional:
  py -3 scripts/build_planner_cases.py --with-llm   # paraphrase templates via Qwen
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

DEFAULT_OUT = BACKEND / "data" / "eval" / "planner_cases.jsonl"

# Synthetic prior recommendations for multi-turn buckets.
_FAKE_RANKED = [
    {"id": "prod_a", "name": "Alpha Running Shoes White", "price": 45, "score": 95},
    {"id": "prod_b", "name": "Beta Breathable Trainers", "price": 38, "score": 92},
    {"id": "prod_c", "name": "Gamma Trail Shoes", "price": 60, "score": 88},
]


def _bundle(plan_id: str = "seed_bundle") -> dict[str, Any]:
    ranked = []
    for i, item in enumerate(_FAKE_RANKED):
        ranked.append(
            {
                **item,
                "reasons": ["seed"],
                "summary": "",
                "rating": 4.5,
                "platform": "Windows",
                "display": "",
                "performance": "",
                "weight_kg": 0.8,
                "image_url": "",
            }
        )
    return {
        "plan_id": plan_id,
        "ranked": ranked,
        "excluded": [],
        "summary": "Seed recommendations for planner eval.",
        "status": "ready",
    }


def _pref(**kwargs: Any) -> dict[str, Any]:
    base = {
        "category": "",
        "budget": None,
        "use_case": "",
        "platform": "No preference",
        "touch": "Not required",
        "hard": [],
        "soft": [],
        "visual_context": "",
        "raw_query": "",
        "search_keywords": [],
    }
    base.update(kwargs)
    return base


def _gold_stages(
    *,
    refuse: bool,
    shortcircuit: bool,
    modality: str,
    verify: bool = True,
) -> list[str]:
    if refuse:
        return []
    if shortcircuit:
        return ["rerank_existing"]
    stages: list[str] = []
    if modality in ("text", "text_image"):
        stages.append("text_recall")
    if modality in ("text_image", "image"):
        stages.append("visual_recall")
    if not stages:
        stages.append("text_recall")
    if verify:
        stages.append("verify")
    stages.append("recommend")
    return stages


def _case(
    case_id: str,
    bucket: str,
    utterance: str,
    *,
    preference: dict[str, Any],
    gold_intent: str,
    should_shortcircuit: bool,
    should_refuse: bool,
    modality: str = "text",
    last_bundle: dict[str, Any] | None = None,
    has_image: bool = False,
    text_empty: bool = False,
    ref_index: int | None = None,
) -> dict[str, Any]:
    ref_id = None
    if ref_index is not None and last_bundle:
        ranked = last_bundle.get("ranked") or []
        if 0 <= ref_index < len(ranked):
            ref_id = ranked[ref_index]["id"]
    pref = dict(preference)
    pref["raw_query"] = utterance if not text_empty else pref.get("raw_query") or ""
    return {
        "case_id": case_id,
        "bucket": bucket,
        "utterance": utterance,
        "preference": pref,
        "last_bundle": last_bundle,
        "has_image": has_image,
        "text_empty": text_empty,
        "gold": {
            "intent": gold_intent,
            "should_shortcircuit": should_shortcircuit,
            "ref_product_id": ref_id,
            "should_refuse": should_refuse,
            "modality": modality,
            "stages": _gold_stages(
                refuse=should_refuse,
                shortcircuit=should_shortcircuit,
                modality=modality,
                verify=True,
            ),
        },
    }


def build_cases(rng: random.Random) -> list[dict[str, Any]]:
  
    return cases


def maybe_paraphrase(cases: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Optional LLM paraphrase for non-safety utterances."""
    try:
        from engine.llm.qwen_client import chat_json, qwen_configured
    except Exception:
        print("[build] qwen_client unavailable; skip --with-llm")
        return cases
    if not qwen_configured():
        print("[build] DASHSCOPE_API_KEY missing; skip --with-llm")
        return cases

    out: list[dict[str, Any]] = []
    for case in cases:
        out.append(case)
        if case["bucket"] in ("safety", "modality"):
            continue
        if rng.random() > 0.35:
            continue
        try:
            data = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the shopper utterance naturally in the SAME language. "
                            "Keep the shopping intent. Return JSON "
                            '{"utterance":"..."} only.'
                        ),
                    },
                    {"role": "user", "content": case["utterance"]},
                ],
                temperature=0.7,
            )
            new_u = str((data or {}).get("utterance") or "").strip()
            if len(new_u) < 2 or new_u == case["utterance"]:
                continue
            clone = json.loads(json.dumps(case))
            clone["case_id"] = case["case_id"] + "p"
            clone["utterance"] = new_u
            clone["preference"]["raw_query"] = new_u
            out.append(clone)
        except Exception as exc:
            print(f"[build] paraphrase skip {case['case_id']}: {exc}")
    out.sort(key=lambda c: c["case_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Planner routing eval cases")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases = build_cases(rng)
    if args.with_llm:
        # Load backend .env via exp_config side effect
        import engine.exp_config  # noqa: F401
        cases = maybe_paraphrase(cases, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    buckets: dict[str, int] = {}
    for c in cases:
        buckets[c["bucket"]] = buckets.get(c["bucket"], 0) + 1
    print(f"[build] wrote {len(cases)} cases -> {args.out}")
    print(f"[build] buckets: {buckets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
