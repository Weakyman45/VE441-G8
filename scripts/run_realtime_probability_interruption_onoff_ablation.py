#!/usr/bin/env python3
"""Probability-based true VoiceShop Realtime Talker ON/OFF ablation.

This is the Realtime/backend-API version of the probability stress test. It
starts two real VoiceShop backend processes, connects to /api/v1/realtime/ws,
and only differs by INTERRUPTION_HANDLING_ENABLED.

Unlike the text-only probability script, this runner always exercises the real
backend session API, TalkerBridge, Realtime WebSocket proxy, response.cancel,
and conversation.item.truncate path.
"""

from __future__ import annotations

import argparse
import os
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
    judge_reply,
    load_dotenv,
    read_jsonl,
    reset_outputs,
    write_json,
    write_metrics,
)
from run_realtime_talker_interruption_onoff_ablation import (  # noqa: E402
    run_realtime_case,
    start_backend,
    write_sample_realtime_cases,
)
from voiceshop_user_simulator import (  # noqa: E402
    configured_user_api,
    configured_user_model,
    generate_dynamic_user_opening,
    persona_manifest_json,
)


def probability(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run probability-based true VoiceShop Realtime Talker ON/OFF ablation.",
    )
    parser.add_argument("--cases", type=Path, help="Input JSONL Realtime cases.")
    parser.add_argument("--out-dir", type=Path, help="Output directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=18965)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--response-timeout", type=float, default=80.0)
    parser.add_argument("--post-interrupt-timeout", type=float, default=80.0)
    parser.add_argument("--interrupt-audio-end-ms", type=int, default=1200)
    parser.add_argument("--probabilities", nargs="+", type=probability, default=[0.25, 0.5, 0.75])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-interrupt-after-chars", type=int, default=60)
    parser.add_argument("--max-interrupt-after-chars", type=int, default=180)
    parser.add_argument(
        "--dynamic-user",
        action="store_true",
        help="Generate user opening/interruption with an OpenAI-compatible chat model.",
    )
    parser.add_argument(
        "--user-model",
        default=None,
        help="Dynamic user model. Defaults to OPENAI_CHAT_MODEL from --env-file.",
    )
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument(
        "--user-persona",
        default="case",
        choices=["case", "impatient", "hesitant", "balanced"],
        help="Dynamic user persona override; 'case' uses each case's user_persona.",
    )
    parser.add_argument(
        "--include-misses",
        action="store_true",
        help="Also run probability misses as ordinary no-interrupt Realtime cases.",
    )
    parser.add_argument("--keep-backends", action="store_true")
    parser.add_argument(
        "--write-sample-cases",
        type=Path,
        default=None,
        help="Write sample Realtime cases JSONL and exit.",
    )
    parser.add_argument(
        "--write-user-personas",
        type=Path,
        default=None,
        help="Write VoiceShop simulated-user persona manifest and exit.",
    )
    add_llm_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.write_user_personas:
        args.write_user_personas.parent.mkdir(parents=True, exist_ok=True)
        args.write_user_personas.write_text(persona_manifest_json() + "\n", encoding="utf-8")
        print(f"Wrote user personas to {args.write_user_personas}")
        return
    if args.write_sample_cases:
        write_sample_realtime_cases(args.write_sample_cases)
        print(f"Wrote sample cases to {args.write_sample_cases}")
        return
    if not args.cases or not args.out_dir:
        parser.error("--cases and --out-dir are required unless --write-sample-cases is used")
    if args.min_interrupt_after_chars > args.max_interrupt_after_chars:
        parser.error("--min-interrupt-after-chars must be <= --max-interrupt-after-chars")

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
            "user_api": configured_user_api(),
            "effective_user_model": configured_user_model(args.user_model),
            "effective_judge_model": configured_chat_model(args.judge_model),
            "openai_realtime_model": os.environ.get("OPENAI_REALTIME_MODEL"),
        },
    )

    rng = random.Random(args.seed)
    sampled_cases = sample_realtime_cases(base_cases, args=args, rng=rng)
    append_jsonl(args.out_dir / "sampled_cases.jsonl", sampled_cases)

    backends = {}
    try:
        if args.dry_run:
            base_urls = {
                "off": f"http://{args.host}:{args.base_port}",
                "on": f"http://{args.host}:{args.base_port + 1}",
            }
        else:
            backends["off"] = start_backend(
                arm="off",
                host=args.host,
                port=args.base_port,
                out_dir=args.out_dir,
                startup_timeout=args.startup_timeout,
            )
            backends["on"] = start_backend(
                arm="on",
                host=args.host,
                port=args.base_port + 1,
                out_dir=args.out_dir,
                startup_timeout=args.startup_timeout,
            )
            base_urls = {arm: backend.base_url for arm, backend in backends.items()}

        generations: list[dict[str, Any]] = []
        judgments: list[dict[str, Any]] = []
        for case in sampled_cases:
            if not case["sampled_interrupt"] and not args.include_misses:
                continue
            if args.dynamic_user:
                if case.get("user_text") or case.get("initial_user_text"):
                    case["static_user_text"] = case.get("user_text") or case.get("initial_user_text")
                case["user_text"] = generate_dynamic_user_opening(
                    case,
                    model=args.user_model,
                    temperature=args.user_temperature,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    persona_key=args.user_persona,
                )
            for arm in ("off", "on"):
                generation = run_realtime_case(
                    case,
                    arm=arm,
                    base_url=base_urls[arm],
                    response_timeout=args.response_timeout,
                    post_interrupt_timeout=args.post_interrupt_timeout,
                    default_interrupt_after_chars=int(case["interrupt_after_chars"]),
                    default_audio_end_ms=args.interrupt_audio_end_ms,
                    dry_run=args.dry_run,
                    dynamic_user=args.dynamic_user,
                    user_model=args.user_model,
                    user_temperature=args.user_temperature,
                    user_timeout=args.timeout,
                    user_persona=args.user_persona,
                )
                generation["interrupt_probability"] = case["interrupt_probability"]
                generation["repeat_index"] = case["repeat_index"]
                generation["sampled_interrupt"] = case["sampled_interrupt"]
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
                judgment["interrupt_probability"] = case["interrupt_probability"]
                judgment["repeat_index"] = case["repeat_index"]
                judgment["sampled_interrupt"] = case["sampled_interrupt"]
                judgments.append(judgment)
                append_jsonl(args.out_dir / "judgments.jsonl", [judgment])
                print(
                    f"p={case['interrupt_probability']} repeat={case['repeat_index']} "
                    f"{case['case_id']} arm={arm} sampled_interrupt={case['sampled_interrupt']}"
                )

        write_metrics(args.out_dir, judgments)
        write_probability_metrics(args.out_dir, judgments)
        print(
            f"Done. Sampled {len(sampled_cases)} Realtime plans; "
            f"generated {len(generations)} replies."
        )
    finally:
        if not args.keep_backends:
            for backend in backends.values():
                backend.stop()


def sample_realtime_cases(
    base_cases: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for probability_value in args.probabilities:
        for repeat_index in range(args.repeats):
            for base_case in base_cases:
                case = deepcopy(base_case)
                base_id = str(base_case["case_id"])
                sampled_interrupt = rng.random() < probability_value
                case["base_case_id"] = base_id
                case["case_id"] = (
                    f"{base_id}__p{_probability_label(probability_value)}"
                    f"__r{repeat_index:02d}"
                )
                case["interrupt_probability"] = probability_value
                case["repeat_index"] = repeat_index
                case["sampled_interrupt"] = sampled_interrupt
                if sampled_interrupt:
                    case["interrupt_after_chars"] = rng.randint(
                        args.min_interrupt_after_chars,
                        args.max_interrupt_after_chars,
                    )
                else:
                    # Very large cutoff means no interrupt is triggered before the
                    # first response naturally finishes.
                    case["interrupt_after_chars"] = 10_000_000
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
    probabilities = sorted(
        {
            row.get("interrupt_probability")
            for row in judgments
            if row.get("interrupt_probability") is not None
        }
    )
    for probability_value in probabilities:
        for arm in ("off", "on"):
            arm_rows = [
                row
                for row in judgments
                if row.get("arm") == arm
                and row.get("interrupt_probability") == probability_value
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
