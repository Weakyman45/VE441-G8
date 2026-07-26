#!/usr/bin/env python3
"""True VoiceShop Realtime Talker ON/OFF interruption ablation.

This script starts two local VoiceShop backend processes with identical config
except INTERRUPTION_HANDLING_ENABLED:

- off: INTERRUPTION_HANDLING_ENABLED=0
- on:  INTERRUPTION_HANDLING_ENABLED=1

For each case and arm it:
1. creates a real backend session,
2. opens the real VoiceShop Realtime WebSocket proxy,
3. sends the case user_text,
4. waits until the assistant is streaming,
5. sends response.cancel + conversation.item.truncate at the configured cutoff,
6. sends the user interrupt_text,
7. collects the Realtime Talker recovery reply,
8. judges ON vs OFF with the shared LLM judge.

Example:
    python scripts/run_realtime_talker_interruption_onoff_ablation.py \
        --cases data/experiments/realtime_interruption_cases.jsonl \
        --out-dir data/experiments/realtime_talker_onoff \
        --dynamic-user

Starter cases:
    python scripts/run_realtime_talker_interruption_onoff_ablation.py \
        --write-sample-cases data/experiments/realtime_interruption_cases.sample.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
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
from voiceshop_user_simulator import (  # noqa: E402
    ShoppingScenario,
    build_case_user_simulator_metadata,
    configured_user_api,
    configured_user_model,
    generate_dynamic_interrupt,
    generate_dynamic_user_opening,
    persona_manifest_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run true VoiceShop Realtime Talker interruption ON/OFF ablation.",
    )
    parser.add_argument("--cases", type=Path, help="Input JSONL Realtime cases.")
    parser.add_argument("--out-dir", type=Path, help="Output directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=18865)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--response-timeout", type=float, default=80.0)
    parser.add_argument("--post-interrupt-timeout", type=float, default=80.0)
    parser.add_argument("--interrupt-after-chars", type=int, default=80)
    parser.add_argument("--interrupt-audio-end-ms", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-backends", action="store_true")
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
            "user_api": configured_user_api(),
            "effective_user_model": configured_user_model(args.user_model),
            "effective_judge_model": configured_chat_model(args.judge_model),
            "openai_realtime_model": os.environ.get("OPENAI_REALTIME_MODEL"),
        },
    )

    backends: dict[str, BackendProcess] = {}
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
        for case in cases:
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
                    default_interrupt_after_chars=args.interrupt_after_chars,
                    default_audio_end_ms=args.interrupt_audio_end_ms,
                    dry_run=args.dry_run,
                    dynamic_user=args.dynamic_user,
                    user_model=args.user_model,
                    user_temperature=args.user_temperature,
                    user_timeout=args.timeout,
                    user_persona=args.user_persona,
                )
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
                judgments.append(judgment)
                append_jsonl(args.out_dir / "judgments.jsonl", [judgment])
                print(
                    f"{case['case_id']} arm={arm} "
                    f"heard_chars={len(generation.get('assistant_heard_text') or '')} "
                    f"reply_chars={len(generation.get('assistant_reply') or '')}"
                )

        write_metrics(args.out_dir, judgments)
        print(f"Done. Wrote {len(generations)} Realtime generations to {args.out_dir}")
    finally:
        if not args.keep_backends:
            for backend in backends.values():
                backend.stop()


class BackendProcess:
    def __init__(self, *, arm: str, proc: subprocess.Popen, base_url: str, log_path: Path) -> None:
        self.arm = arm
        self.proc = proc
        self.base_url = base_url
        self.log_path = log_path

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


def start_backend(
    *,
    arm: str,
    host: str,
    port: int,
    out_dir: Path,
    startup_timeout: float,
) -> BackendProcess:
    env = os.environ.copy()
    env["INTERRUPTION_HANDLING_ENABLED"] = "1" if arm == "on" else "0"
    env.setdefault("LOCAL_VAD_ENABLED", "0")
    log_path = out_dir / f"backend_{arm}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--host", host, "--port", str(port)],
        cwd=BACKEND_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://{host}:{port}"
    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Backend {arm} exited early; see {log_path}")
        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/realtime/check", timeout=1.0) as resp:
                json.loads(resp.read().decode("utf-8"))
                return BackendProcess(arm=arm, proc=proc, base_url=base_url, log_path=log_path)
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    raise RuntimeError(f"Backend {arm} did not start within {startup_timeout}s; see {log_path}")


def run_realtime_case(
    case: dict[str, Any],
    *,
    arm: str,
    base_url: str,
    response_timeout: float,
    post_interrupt_timeout: float,
    default_interrupt_after_chars: int,
    default_audio_end_ms: int,
    dry_run: bool,
    dynamic_user: bool = False,
    user_model: str | None = None,
    user_temperature: float = 0.7,
    user_timeout: float = 60.0,
    user_persona: str = "case",
) -> dict[str, Any]:
    raw_initial_user_text = str(case.get("user_text") or case.get("initial_user_text") or "").strip()
    initial_user_text = raw_initial_user_text
    if dynamic_user and not initial_user_text:
        initial_user_text = generate_dynamic_user_opening(
            case,
            model=user_model,
            temperature=user_temperature,
            timeout=user_timeout,
            dry_run=dry_run,
            persona_key=user_persona,
        )
        case["user_text"] = initial_user_text
    if not initial_user_text:
        initial_user_text = "Help me compare options."

    if dry_run:
        interrupt_text = str(case.get("interrupt_text") or case.get("user_interrupt_text") or "").strip()
        if dynamic_user:
            interrupt_text = generate_dynamic_interrupt(
                case,
                assistant_heard_text="[dry-run heard text]",
                model=user_model,
                temperature=user_temperature,
                timeout=user_timeout,
                dry_run=True,
                persona_key=user_persona,
            )
        return {
            "case_id": case["case_id"],
            "arm": arm,
            "dynamic_user": dynamic_user,
            "user_persona": user_persona if user_persona != "case" else case.get("user_persona"),
            "initial_user_text": initial_user_text,
            "assistant_heard_text": "[dry-run heard text]",
            "user_interrupt_text": interrupt_text,
            "assistant_reply": f"[dry-run realtime {arm}]",
            "events": [],
        }

    session = create_session(base_url)
    ws = RealtimeWebSocket.open(_ws_url(base_url, session["ws_url"]))
    events: list[dict[str, Any]] = []
    first_reply = ""
    recovery_reply = ""
    item_id = ""
    interrupted = False
    collecting_recovery = False
    interrupt_after_chars = int(case.get("interrupt_after_chars") or default_interrupt_after_chars)
    audio_end_ms = int(case.get("interrupt_audio_end_ms") or default_audio_end_ms)
    interrupt_text = str(case.get("interrupt_text") or case.get("user_interrupt_text") or "wait").strip()

    try:
        send_user_text(ws, initial_user_text)
        deadline = time.time() + response_timeout
        while time.time() < deadline:
            event = ws.recv_json(timeout=1.0)
            if event is None:
                continue
            events.append(compact_event(event))
            event_type = event.get("type")
            if event_type in ("response.output_item.added", "response.output_item.created"):
                item = event.get("item") or {}
                item_id = item.get("id") or event.get("item_id") or item_id
            elif event_type in (
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
                "response.output_text.delta",
                "response.text.delta",
            ):
                delta = event.get("delta") or ""
                if collecting_recovery:
                    recovery_reply += delta
                else:
                    first_reply += delta
                    if not interrupted and len(first_reply) >= interrupt_after_chars:
                        heard_text = first_reply[:interrupt_after_chars].strip()
                        if dynamic_user:
                            interrupt_text = generate_dynamic_interrupt(
                                case,
                                assistant_heard_text=heard_text,
                                model=user_model,
                                temperature=user_temperature,
                                timeout=user_timeout,
                                dry_run=False,
                                persona_key=user_persona,
                            )
                        interrupt_realtime(ws, item_id=item_id, audio_end_ms=audio_end_ms)
                        send_user_text(ws, interrupt_text)
                        interrupted = True
                        collecting_recovery = True
                        deadline = time.time() + post_interrupt_timeout
            elif event_type in ("response.done", "response.cancelled"):
                if collecting_recovery and recovery_reply.strip():
                    break
            elif event_type == "error":
                message = ((event.get("error") or {}).get("message") or event.get("message") or "")
                if "Cancellation failed" not in message and "no active response" not in message:
                    raise RuntimeError(f"Realtime error: {message}")

        return {
            "case_id": case["case_id"],
            "arm": arm,
            "dynamic_user": dynamic_user,
            "user_persona": user_persona if user_persona != "case" else case.get("user_persona"),
            "initial_user_text": initial_user_text,
            "session": session,
            "assistant_heard_text": first_reply[:interrupt_after_chars].strip(),
            "first_reply_prefix": first_reply.strip(),
            "user_interrupt_text": interrupt_text,
            "assistant_reply": recovery_reply.strip(),
            "interrupted": interrupted,
            "item_id": item_id,
            "events": events,
        }
    finally:
        ws.close()


def create_session(base_url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base_url}/api/v1/session",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_user_text(ws: "RealtimeWebSocket", text: str) -> None:
    ws.send_json(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )
    ws.send_json({"type": "response.create"})


def interrupt_realtime(ws: "RealtimeWebSocket", *, item_id: str, audio_end_ms: int) -> None:
    ws.send_json({"type": "response.cancel"})
    if item_id:
        ws.send_json(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms,
            }
        )


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("type")
    compact: dict[str, Any] = {"type": event_type}
    for key in ("item_id", "response_id", "event_id", "delta", "transcript", "text"):
        value = event.get(key)
        if isinstance(value, str) and value:
            compact[key] = value[:500]
    if isinstance(event.get("item"), dict):
        compact["item_id"] = event["item"].get("id")
    if event_type == "error":
        compact["error"] = event.get("error") or event.get("message")
    return compact


class RealtimeWebSocket:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    @classmethod
    def open(cls, url: str) -> "RealtimeWebSocket":
        parsed = urlparse(url)
        if parsed.scheme != "ws":
            raise ValueError(f"Only ws:// local URLs are supported, got {url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("utf-8"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket closed during handshake")
            response += chunk
        status = response.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
        if " 101 " not in status:
            raise ConnectionError(f"WebSocket handshake failed: {status}")
        sock.settimeout(None)
        return cls(sock)

    def send_json(self, obj: dict[str, Any]) -> None:
        self._write_frame(0x1, json.dumps(obj, ensure_ascii=False).encode("utf-8"), mask=True)

    def recv_json(self, *, timeout: float) -> dict[str, Any] | None:
        old_timeout = self.sock.gettimeout()
        self.sock.settimeout(timeout)
        try:
            while True:
                opcode, payload = self._read_frame()
                if opcode == 0x8:
                    raise ConnectionError("WebSocket closed")
                if opcode == 0x9:
                    self._write_frame(0xA, payload, mask=True)
                    continue
                if opcode == 0x1:
                    return json.loads(payload.decode("utf-8"))
                return None
        except socket.timeout:
            return None
        finally:
            self.sock.settimeout(old_timeout)

    def close(self) -> None:
        try:
            self._write_frame(0x8, b"", mask=True)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def _read_frame(self) -> tuple[int, bytes]:
        header = _recv_exact(self.sock, 2)
        opcode = header[0] & 0x0F
        masked = (header[1] & 0x80) != 0
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exact(self.sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exact(self.sock, 8))[0]
        mask_key = _recv_exact(self.sock, 4) if masked else b""
        payload = _recv_exact(self.sock, length) if length else b""
        if masked:
            payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _write_frame(self, opcode: int, payload: bytes, *, mask: bool) -> None:
        header = bytearray([0x80 | (opcode & 0x0F)])
        length = len(payload)
        if length < 126:
            header.append((0x80 if mask else 0x00) | length)
        elif length < (1 << 16):
            header.append((0x80 if mask else 0x00) | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append((0x80 if mask else 0x00) | 127)
            header.extend(struct.pack("!Q", length))
        if mask:
            mask_key = os.urandom(4)
            header.extend(mask_key)
            payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("socket closed while reading")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _ws_url(base_url: str, ws_url: str) -> str:
    parsed = urlparse(base_url)
    path = ws_url if ws_url.startswith("/") else f"/{ws_url}"
    return f"ws://{parsed.hostname}:{parsed.port}{path}"


def write_sample_realtime_cases(path: Path) -> None:
    laptop_scenario = ShoppingScenario(
        product_category="lightweight laptops",
        reason_for_shopping="The shopper needs a laptop for school and everyday productivity.",
        known_info="Budget is under 900 dollars. Portability and battery life matter.",
        unknown_info="The shopper has not decided between Windows and macOS.",
        task_instructions=(
            "Get a concise comparison of three options. If one option sounds promising, "
            "ask follow-up questions about it."
        ),
    )
    earbuds_scenario = ShoppingScenario(
        product_category="noise cancelling earbuds",
        reason_for_shopping="The shopper commutes and wants earbuds that reduce train and street noise.",
        known_info="Noise cancellation, comfort, and value matter.",
        unknown_info="Exact budget and brand preference are not fixed.",
        task_instructions="Compare two choices and clarify the most practical trade-off.",
    )
    rows = [
        {
            "case_id": "rt_second_option",
            "user_persona": "balanced",
            "user_simulator": build_case_user_simulator_metadata(
                scenario=laptop_scenario,
                persona_key="balanced",
            ),
            "interrupt_type": "true_interrupt",
            "user_text": (
                "I need a lightweight laptop under 900 dollars. Compare three options, "
                "and give each option a distinct number."
            ),
            "interrupt_text": "Wait, what about the second one?",
            "interrupt_after_chars": 120,
            "expected_referent": "the second option the assistant just introduced",
            "heard_info": ["the beginning of the assistant comparison"],
            "unheard_key_info": ["important details after the cutoff about the second option"],
        },
        {
            "case_id": "rt_backchannel",
            "user_persona": "hesitant",
            "user_simulator": build_case_user_simulator_metadata(
                scenario=earbuds_scenario,
                persona_key="hesitant",
            ),
            "interrupt_type": "backchannel",
            "user_text": (
                "I need noise cancelling earbuds for commuting. Give me a short comparison "
                "of two choices."
            ),
            "interrupt_text": "mm-hmm",
            "interrupt_after_chars": 100,
            "expected_referent": "",
            "heard_info": ["the beginning of the assistant comparison"],
            "unheard_key_info": ["remaining recommendation details after the cutoff"],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
