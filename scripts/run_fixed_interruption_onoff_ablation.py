#!/usr/bin/env python3
"""Deprecated text-only ON/OFF ablation for VoiceShop interruption handling.

This script does NOT use the real VoiceShop backend API or Realtime Talker.
Use run_realtime_talker_interruption_onoff_ablation.py for the real backend API
fixed experiment.

Each input JSONL row describes one fixed interrupted turn. The runner replays
the same case twice:

- off: no interruption prompt and no [INTERRUPTED] marker.
- on: interruption prompt plus [INTERRUPTED] marker after heard text.

Example:
    python scripts/run_fixed_interruption_onoff_ablation.py \
        --cases data/experiments/interruption_cases.jsonl \
        --out-dir data/experiments/fixed_interruption_onoff

To create a starter manifest:
    python scripts/run_fixed_interruption_onoff_ablation.py \
        --write-sample-cases data/experiments/interruption_cases.sample.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from interruption_eval_common import (  # noqa: E402
    add_llm_args,
    append_jsonl,
    configured_chat_model,
    generate_reply,
    judge_reply,
    load_dotenv,
    read_jsonl,
    reset_outputs,
    write_json,
    write_metrics,
    write_sample_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed VoiceShop interruption ON/OFF ablation cases.",
    )
    parser.add_argument("--cases", type=Path, help="Input JSONL fixed cases.")
    parser.add_argument("--out-dir", type=Path, help="Output directory.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--write-sample-cases",
        type=Path,
        default=None,
        help="Write sample cases JSONL and exit.",
    )
    add_llm_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.write_sample_cases:
        write_sample_cases(args.write_sample_cases)
        print(f"Wrote sample cases to {args.write_sample_cases}")
        return
    if not args.cases or not args.out_dir:
        parser.error("--cases and --out-dir are required unless --write-sample-cases is used")

    load_dotenv(args.env_file, override=False)
    cases = read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reset_outputs(args.out_dir, ["generations.jsonl", "judgments.jsonl", "metrics.csv", "summary.md"])
    write_json(
        args.out_dir / "run_config.json",
        vars(args)
        | {
            "cases_count": len(cases),
            "effective_model": configured_chat_model(args.model),
            "effective_judge_model": configured_chat_model(args.judge_model),
        },
    )

    generations: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    case_by_id = {case["case_id"]: case for case in cases}

    for case in cases:
        heard_text = str(case.get("assistant_heard_text") or "").strip()
        if not heard_text:
            raise ValueError(f"case_id={case['case_id']} missing assistant_heard_text")
        for arm in ("off", "on"):
            generation = generate_reply(
                case,
                arm=arm,
                heard_text=heard_text,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            generations.append(generation)
            append_jsonl(args.out_dir / "generations.jsonl", [generation])

            judgment = judge_reply(
                generation,
                case_by_id[generation["case_id"]],
                judge_model=args.judge_model,
                judge_temperature=args.judge_temperature,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            judgments.append(judgment)
            append_jsonl(args.out_dir / "judgments.jsonl", [judgment])
            print(
                f"{case['case_id']} arm={arm} "
                f"hallucinated={judgment.get('hallucinated_already_said')} "
                f"referent={judgment.get('referent_correct')}"
            )

    write_metrics(args.out_dir, judgments)
    print(f"Done. Wrote {len(generations)} generations to {args.out_dir}")


if __name__ == "__main__":
    main()
