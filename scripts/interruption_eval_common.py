#!/usr/bin/env python3
"""Shared helpers for VoiceShop interruption ON/OFF ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / "backend" / ".env"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.5"
INTERRUPTED_MARKER = "[INTERRUPTED]"

BASE_TALKER_PROMPT = (
    "You are VoiceShop++, a concise spoken retail shopping assistant. "
    "Always reply in English only. Help the shopper compare products, answer "
    "follow-up questions, and clarify constraints. Keep replies short and "
    "spoken-style."
)

INTERRUPTION_PROMPT = f"""
If a prior assistant message contains {INTERRUPTED_MARKER}, that marker is the cut-off point.
The user did not hear anything after that marker.

## Handling interruptions (barge-in)

The user can and will interrupt you mid-sentence. When that happens:

1. You did NOT finish speaking. The user only heard the words up to the
   point where they cut in - never assume the rest was heard. Never claim
   or imply you "already mentioned" or "just said" something that came
   after the cut-off point.

2. Whatever the user says when interrupting is their most current and
   highest-priority intent. Adopt it immediately and let it override your
   previous plan or the answer you were in the middle of giving.

3. Be brief. Respond directly to what they now want. Do NOT re-read,
   recap, or repeat the recommendations or details they already heard
   before interrupting.

4. Only restate something from the unspoken part if it is essential to the
   user's new request AND they were cut off before hearing it - and then
   keep it to a single short clause, not a re-listing.

5. If the interruption is just a backchannel ("uh-huh", "okay", "right",
   "mm-hmm", "got it") or clearly not addressed to you (background speech,
   someone else talking), do NOT treat it as a new instruction. Continue
   naturally from where you were.

Keep every reply short and spoken-style - one or two sentences.
""".strip()

JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a spoken shopping assistant interruption experiment.
Judge only the assistant reply under test, using the case metadata as ground
truth. Return one JSON object and no extra text.
""".strip()

JUDGE_USER_TEMPLATE = """
Case metadata:
{case_json}

Experiment arm:
{arm}

Assistant reply under test:
{assistant_reply}

Score these fields:
- hallucinated_already_said: true if the assistant claims or implies it already
  said something the user did not hear after the cutoff.
- referent_correct: true if pronouns or references such as "second one" point
  to the expected entity. If no expected_referent is provided, use null.
- key_info_retained: true if the assistant includes or asks for the important
  unheard information needed for the user's latest request. If no
  unheard_key_info is provided, use null.
- redundant_repeat: true if it repeats details the user already heard before
  the cutoff more than needed.
- backchannel_misresponse: true if interrupt_type is backchannel, bystander, or
  noise and the assistant wrongly treats it as a new substantive request.

Return JSON with exactly these keys:
{{
  "hallucinated_already_said": true/false,
  "referent_correct": true/false/null,
  "key_info_retained": true/false/null,
  "redundant_repeat": true/false,
  "backchannel_misresponse": true/false,
  "rationale": "short reason"
}}
""".strip()


def load_dotenv(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="Generation model. Defaults to OPENAI_CHAT_MODEL from --env-file.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model. Defaults to OPENAI_CHAT_MODEL from --env-file.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Env file with OPENAI_API_KEY, OPENAI_REALTIME_MODEL, and OPENAI_CHAT_MODEL.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM APIs.")


def configured_chat_model(model: str | None = None) -> str:
    return (model or os.environ.get("OPENAI_CHAT_MODEL") or DEFAULT_OPENAI_CHAT_MODEL).strip()


def resolve_model(model: str | None) -> tuple[str, str, str]:
    configured = configured_chat_model(model)
    if "/" in configured:
        provider, name = configured.split("/", 1)
    elif configured.startswith("qwen"):
        provider, name = "qwen", configured
    else:
        provider, name = "openai", configured
    provider = provider.lower()
    if provider in ("dashscope", "aliyun"):
        provider = "qwen"
    if provider not in ("openai", "qwen"):
        raise ValueError(f"Unsupported model provider in {configured!r}")

    if provider == "openai":
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (os.environ.get("OPENAI_CHAT_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    else:
        api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        base_url = (
            os.environ.get("QWEN_CHAT_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
    if not api_key:
        raise RuntimeError(f"Missing API key for provider={provider}")
    return name, api_key, base_url


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float,
    timeout: float,
) -> str:
    name, api_key, base_url = resolve_model(model)
    payload = {
        "model": name,
        "messages": messages,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body[:800]}") from exc
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError(f"Empty LLM response: {json.dumps(data)[:500]}")
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row.setdefault("case_id", f"{path.stem}_{line_no}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def reset_outputs(out_dir: Path, names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()


def write_sample_cases(path: Path) -> None:
    rows = [
        {
            "case_id": "sample_referent_second",
            "interrupt_type": "true_interrupt",
            "history_before": [
                {
                    "role": "user",
                    "content": "I need a lightweight laptop for school under 900 dollars.",
                }
            ],
            "assistant_full_text": (
                "I would compare three options. The first is a budget ThinkPad for durability. "
                "The second is the Swift Go because it is lighter and has about ten hours of "
                "battery life. The third is a gaming model, but it is heavier."
            ),
            "assistant_heard_text": (
                "I would compare three options. The first is a budget ThinkPad for durability. "
                "The second is the Swift Go"
            ),
            "user_interrupt_text": "Wait, what about the second one?",
            "expected_referent": "the Swift Go",
            "heard_info": ["first option is a budget ThinkPad", "second option is Swift Go"],
            "unheard_key_info": ["second option is lighter", "second option has about ten hours of battery life"],
        },
        {
            "case_id": "sample_backchannel",
            "interrupt_type": "backchannel",
            "history_before": [
                {"role": "user", "content": "Find me noise cancelling earbuds for commuting."}
            ],
            "assistant_full_text": (
                "Sure. For commuting, I would prioritize strong ANC, comfort, and battery life. "
                "The Sony option is best for noise cancelling, while the Anker option is cheaper."
            ),
            "assistant_heard_text": (
                "Sure. For commuting, I would prioritize strong ANC, comfort, and battery life."
            ),
            "user_interrupt_text": "mm-hmm",
            "expected_referent": "",
            "heard_info": ["prioritize ANC, comfort, and battery life"],
            "unheard_key_info": ["Sony is best for noise cancelling", "Anker is cheaper"],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_system_prompt(arm: str) -> str:
    if arm == "on":
        return f"{BASE_TALKER_PROMPT}\n\n{INTERRUPTION_PROMPT}"
    if arm == "off":
        return BASE_TALKER_PROMPT
    raise ValueError(f"Unknown arm: {arm}")


def build_messages_for_case(case: dict[str, Any], *, arm: str, heard_text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(arm)}]
    messages.extend(_normalize_history(case.get("history_before") or []))
    assistant_text = heard_text.strip()
    if arm == "on" and assistant_text:
        assistant_text = f"{assistant_text} {INTERRUPTED_MARKER}"
    messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": str(case.get("user_interrupt_text") or "").strip()})
    return messages


def generate_reply(
    case: dict[str, Any],
    *,
    arm: str,
    heard_text: str,
    model: str | None,
    temperature: float,
    timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    messages = build_messages_for_case(case, arm=arm, heard_text=heard_text)
    started = time.time()
    if dry_run:
        reply = f"[dry-run {arm}] {case.get('user_interrupt_text', '')}"
    else:
        reply = chat_completion(
            messages,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    return {
        "case_id": case["case_id"],
        "arm": arm,
        "assistant_heard_text": heard_text,
        "user_interrupt_text": case.get("user_interrupt_text", ""),
        "assistant_reply": reply,
        "messages": messages,
        "latency_seconds": round(time.time() - started, 3),
    }


def judge_reply(
    generation: dict[str, Any],
    case: dict[str, Any],
    *,
    judge_model: str | None,
    judge_temperature: float,
    timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    prompt = JUDGE_USER_TEMPLATE.format(
        case_json=json.dumps(_judge_case_view(case), ensure_ascii=False, indent=2),
        arm=generation["arm"],
        assistant_reply=generation["assistant_reply"],
    )
    if dry_run:
        parsed = {
            "hallucinated_already_said": False,
            "referent_correct": None,
            "key_info_retained": None,
            "redundant_repeat": False,
            "backchannel_misresponse": False,
            "rationale": "dry run",
        }
        raw = json.dumps(parsed)
    else:
        raw = chat_completion(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=judge_model,
            temperature=judge_temperature,
            timeout=timeout,
        )
        parsed = parse_json_object(raw)
    return {
        "case_id": case["case_id"],
        "arm": generation["arm"],
        "raw_judge": raw,
        **parsed,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def write_metrics(out_dir: Path, judgments: list[dict[str, Any]]) -> None:
    metric_names = [
        "hallucinated_already_said",
        "referent_correct",
        "key_info_retained",
        "redundant_repeat",
        "backchannel_misresponse",
    ]
    rows: list[dict[str, Any]] = []
    for arm in ("off", "on"):
        arm_rows = [row for row in judgments if row.get("arm") == arm]
        for metric in metric_names:
            values = [row.get(metric) for row in arm_rows if isinstance(row.get(metric), bool)]
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n": len(values),
                    "rate": round(sum(1 for value in values if value) / len(values), 4)
                    if values
                    else "",
                }
            )
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["arm", "metric", "n", "rate"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.md").write_text(build_summary(rows), encoding="utf-8")


def build_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# Interruption ON/OFF Ablation Summary", ""]
    for metric in sorted({row["metric"] for row in rows}):
        off = next((row for row in rows if row["arm"] == "off" and row["metric"] == metric), {})
        on = next((row for row in rows if row["arm"] == "on" and row["metric"] == metric), {})
        lines.append(
            f"- `{metric}`: off={off.get('rate', '')} (n={off.get('n', 0)}), "
            f"on={on.get('rate', '')} (n={on.get('n', 0)})"
        )
    return "\n".join(lines) + "\n"


def cutoff_by_fraction(text: str, fraction: float) -> tuple[str, int]:
    if not text:
        return "", 0
    cutoff = max(1, min(len(text), round(len(text) * fraction)))
    while cutoff < len(text) and text[cutoff - 1].isalnum() and text[cutoff].isalnum():
        cutoff += 1
    return text[:cutoff].rstrip(), cutoff


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _normalize_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "user", "assistant") and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _judge_case_view(case: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "interrupt_type",
        "user_persona",
        "user_simulator",
        "assistant_full_text",
        "assistant_heard_text",
        "user_interrupt_text",
        "expected_referent",
        "heard_info",
        "unheard_key_info",
        "cutoff_fraction",
        "char_cutoff",
    ]
    view = {key: case.get(key) for key in keys if key in case and key != "user_simulator"}
    user_sim = case.get("user_simulator")
    if isinstance(user_sim, dict):
        view["user_simulator"] = {
            "task_specific_persona": user_sim.get("task_specific_persona"),
            "runtime_persona_config": user_sim.get("runtime_persona_config"),
            "shopping_scenario": user_sim.get("shopping_scenario"),
        }
    return view
