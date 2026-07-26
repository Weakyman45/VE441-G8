#!/usr/bin/env python3
"""Deprecated text-only probability ON/OFF interruption stress test.

This script does NOT use the real VoiceShop backend API or Realtime Talker.
Use run_realtime_probability_interruption_onoff_ablation.py for the real
backend API probability experiment.

This runner samples whether each case is interrupted and where the cutoff falls.
The sampled interruption plan is paired: the OFF and ON arms use the exact same
case, interrupt decision, cutoff, and user interrupt text.

Rows that miss the interrupt probability are logged in sampled_cases.jsonl but
are not sent to the model by default, because the target metrics are about
post-interruption recovery. Use --include-misses to also generate ordinary
follow-up replies for missed cases that define next_user_text.

Example:
    python scripts/run_probability_interruption_onoff_ablation.py \
        --cases data/experiments/interruption_cases.jsonl \
        --out-dir data/experiments/prob_interruption_onoff \
        --probabilities 0.25 0.5 0.75 \
        --repeats 3
"""

from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from interruption_eval_common import (  # noqa: E402
    add_llm_args,
    append_jsonl,
    configured_chat_model,
    cutoff_by_fraction,
    generate_reply,
    judge_reply,
    load_dotenv,
    read_jsonl,
    reset_outputs,
    write_json,
    write_metrics,
    write_sample_cases,
)


DEFAULT_INTERRUPT_TEXTS = ("wait", "hold on", "what about that one?", "mm-hmm")


def probability(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run probability-based VoiceShop interruption ON/OFF ablation.",
    )
    parser.add_argument("--cases", type=Path, help="Input JSONL cases.")
    parser.add_argument("--out-dir", type=Path, help="Output directory.")
    parser.add_argument("--probabilities", nargs="+", type=probability, default=[0.25, 0.5, 0.75])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-fraction", type=float, default=0.2)
    parser.add_argument("--max-fraction", type=float, default=0.8)
    parser.add_argument(
        "--interrupt-texts",
        nargs="+",
        default=list(DEFAULT_INTERRUPT_TEXTS),
        help="Fallback user interruption texts when a case does not provide one.",
    )
    parser.add_argument(
        "--include-misses",
        action="store_true",
        help="Also generate non-interrupted follow-up rows for probability misses.",
    )
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
    if args.min_fraction > args.max_fraction:
        parser.error("--min-fraction must be <= --max-fraction")

    load_dotenv(args.env_file, override=False)
    base_cases = read_jsonl(args.cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reset_outputs(
        args.out_dir,
        [
            "sampled_cases.jsonl",
            "generations.jsonl",
            "judgments.jsonl",
            "metrics.csv",
            "metrics_by_probability.csv",
            "summary.md",
        ],
    )
    write_json(
        args.out_dir / "run_config.json",
        vars(args)
        | {
            "base_cases_count": len(base_cases),
            "effective_model": configured_chat_model(args.model),
            "effective_judge_model": configured_chat_model(args.judge_model),
        },
    )

    rng = random.Random(args.seed)
    sampled_cases = sample_cases(base_cases, args=args, rng=rng)
    append_jsonl(args.out_dir / "sampled_cases.jsonl", sampled_cases)

    generations: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []

    for case in sampled_cases:
        if not case["interrupted"] and not args.include_misses:
            continue
        if not case["interrupted"] and not case.get("next_user_text"):
            continue

        heard_text = str(case.get("assistant_heard_text") or "").strip()
        if not heard_text:
            raise ValueError(f"sample_id={case['case_id']} missing assistant_heard_text")

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
            generation["probability"] = case["interrupt_probability"]
            generation["repeat_index"] = case["repeat_index"]
            generation["interrupted"] = case["interrupted"]
            generations.append(generation)
            append_jsonl(args.out_dir / "generations.jsonl", [generation])

            judgment = judge_reply(
                generation,
                case,
                judge_model=args.judge_model,
                judge_temperature=args.judge_temperature,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            judgment["probability"] = case["interrupt_probability"]
            judgment["repeat_index"] = case["repeat_index"]
            judgment["interrupted"] = case["interrupted"]
            judgments.append(judgment)
            append_jsonl(args.out_dir / "judgments.jsonl", [judgment])
            print(
                f"p={case['interrupt_probability']} repeat={case['repeat_index']} "
                f"{case['case_id']} arm={arm} interrupted={case['interrupted']}"
            )

    write_metrics(args.out_dir, judgments)
    write_probability_metrics(args.out_dir, judgments)
    print(f"Done. Sampled {len(sampled_cases)} case plans; generated {len(generations)} replies.")


def sample_cases(
    base_cases: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for probability_value in args.probabilities:
        for repeat_index in range(args.repeats):
            for base_case in base_cases:
                interrupted = rng.random() < probability_value
                case = deepcopy(base_case)
                base_id = str(base_case["case_id"])
                case["base_case_id"] = base_id
                case["case_id"] = (
                    f"{base_id}__p{_probability_label(probability_value)}"
                    f"__r{repeat_index:02d}"
                )
                case["interrupt_probability"] = probability_value
                case["repeat_index"] = repeat_index
                case["interrupted"] = interrupted
                if interrupted:
                    fraction = rng.uniform(args.min_fraction, args.max_fraction)
                    heard_text, char_cutoff = cutoff_by_fraction(
                        str(base_case.get("assistant_full_text") or ""),
                        fraction,
                    )
                    case["assistant_heard_text"] = heard_text
                    case["cutoff_fraction"] = round(fraction, 4)
                    case["char_cutoff"] = char_cutoff
                    if not case.get("user_interrupt_text"):
                        case["user_interrupt_text"] = rng.choice(args.interrupt_texts)
                else:
                    case["assistant_heard_text"] = str(base_case.get("assistant_full_text") or "")
                    case["user_interrupt_text"] = str(base_case.get("next_user_text") or "")
                    case["cutoff_fraction"] = None
                    case["char_cutoff"] = None
                sampled.append(case)
    return sampled


def write_probability_metrics(out_dir: Path, judgments: list[dict[str, Any]]) -> None:
    import csv

    metric_names = [
        "hallucinated_already_said",
        "referent_correct",
        "key_info_retained",
        "redundant_repeat",
        "backchannel_misresponse",
    ]
    rows: list[dict[str, Any]] = []
    probabilities = sorted({row.get("probability") for row in judgments if row.get("probability") is not None})
    for probability_value in probabilities:
        for arm in ("off", "on"):
            arm_rows = [
                row for row in judgments
                if row.get("arm") == arm and row.get("probability") == probability_value
            ]
            for metric in metric_names:
                values = [row.get(metric) for row in arm_rows if isinstance(row.get(metric), bool)]
                rows.append(
                    {
                        "probability": probability_value,
                        "arm": arm,
                        "metric": metric,
                        "n": len(values),
                        "rate": round(sum(1 for value in values if value) / len(values), 4)
                        if values
                        else "",
                    }
                )
    path = out_dir / "metrics_by_probability.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["probability", "arm", "metric", "n", "rate"])
        writer.writeheader()
        writer.writerows(rows)


def _probability_label(value: float) -> str:
    percent = round(value * 100)
    if abs(value - percent / 100) < 1e-9:
        return str(percent)
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
