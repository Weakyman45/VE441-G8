"""Terminal Talker–Worker test (no Android App, no Realtime audio).

Usage (from repo root or backend/):
    py -3 backend/cli_chat.py
    py -3 backend/cli_chat.py --no-llm          # rule summary only
    py -3 backend/cli_chat.py --once "1500 左右做设计的笔记本"

Commands inside the REPL:
    /quit          exit
    /interrupt     emit session.interrupted (cancel in-flight Worker)
    /pref          print current preference
    /bundle        print last recommendation JSON
    /session       print session snapshot

Requires backend/.env DASHSCOPE_API_KEY for Qwen LLM summaries (optional with --no-llm).
Clash / system HTTP proxy is used automatically by urllib if configured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import server  # noqa: E402  — loads .env, exposes search()
from engine.bus import EventBus
from engine.events import Event, EventType
from engine.intent import extract_preference
from engine.logging_store import LoggingStore
from engine.realtime_map import RT_CLIENT_ITEM_CREATE, realtime_source
from engine.session import SessionStore
from engine.talker.text_llm import summarize_recommendations
from engine.worker.runtime import WorkerRuntime


def _print(title: str, obj: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj)


class CliHarness:
    def __init__(self, *, use_llm: bool) -> None:
        self.use_llm = use_llm
        self.bus = EventBus()
        self.sessions = SessionStore()
        self.logs = LoggingStore(os.path.join(HERE, "data", "cli_engine_logs.db"))
        self.runtime = WorkerRuntime(self.bus, self.sessions, self.logs, server.search)
        self.runtime.start()
        self.session_id = uuid.uuid4().hex[:16]
        self.sessions.create(self.session_id)
        self._last_bundle: dict | None = None
        self._ready = threading.Event()
        self.bus.subscribe(EventType.WORKER_RECOMMENDATION_READY, self._on_ready)
        self.bus.subscribe(EventType.WORKER_STATUS, self._on_status)
        print(f"[cli] session_id={self.session_id}")
        print("[cli] type a need (e.g. 3000左右拍视频的笔记本). /quit to exit.")

    def _on_status(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        msg = (event.payload or {}).get("message") or (event.payload or {}).get("status")
        print(f"[worker.status] {msg}")

    def _on_ready(self, event: Event) -> None:
        if event.session_id != self.session_id:
            return
        self._last_bundle = event.payload or {}
        self._ready.set()

    def interrupt(self) -> None:
        self.bus.emit(
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=self.session_id,
                payload={
                    "reason": "cli_interrupt",
                    "source": realtime_source(
                        {"type": "input_audio_buffer.speech_started", "event_id": "cli"}
                    ),
                },
            )
        )
        print("[cli] interrupted — in-flight plan cancelled")

    def handle_user_text(self, text: str) -> None:
        text = text.strip()
        if len(text) < 2:
            return
        # Same domain path as TalkerBridge, with a synthetic Realtime source
        # (as if the app sent conversation.item.create with input_text).
        source = realtime_source(
            {
                "type": RT_CLIENT_ITEM_CREATE,
                "event_id": f"cli-{uuid.uuid4().hex[:8]}",
            },
            extra={"channel": "cli_text", "content_type": "input_text"},
        )
        state = self.sessions.require(self.session_id)
        self.sessions.append_turn(self.session_id, "user", text)
        self.logs.log_conversation(self.session_id, "user", text)
        preference = extract_preference(text, state.preference)
        self.sessions.update_preference(self.session_id, preference)
        self.logs.log_trace(self.session_id, "cli", "intent_updated", preference.to_dict())

        _print("preference", preference.to_dict())

        self._ready.clear()
        self._last_bundle = None
        self.bus.emit(
            Event(
                type=EventType.USER_UTTERANCE,
                session_id=self.session_id,
                payload={"text": text, "source": source},
            )
        )
        self.bus.emit(
            Event(
                type=EventType.USER_INTENT_UPDATED,
                session_id=self.session_id,
                payload={**preference.to_dict(), "source": source},
            )
        )

        print("[cli] waiting for Worker…")
        if not self._ready.wait(timeout=30):
            print("[cli] timeout — no recommendation_ready")
            return

        bundle = self._last_bundle or {}
        ranked = bundle.get("ranked") or []
        _print(
            "worker.recommendation_ready (top)",
            [
                {
                    "name": r.get("name"),
                    "price": r.get("price"),
                    "score": r.get("score"),
                    "reasons": r.get("reasons"),
                }
                for r in ranked[:3]
            ],
        )

        if self.use_llm:
            print("[cli] asking text LLM (Talker stand-in)…")
            spoken = summarize_recommendations(
                user_text=text,
                preference=preference.to_dict(),
                bundle=bundle,
            )
        else:
            spoken = bundle.get("summary") or "(no summary)"
        self.sessions.append_turn(self.session_id, "assistant", spoken)
        self.logs.log_conversation(self.session_id, "assistant", spoken)
        _print("Talker (text LLM)", spoken)

    def snapshot(self) -> dict | None:
        return self.sessions.snapshot(self.session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI test for Talker–Worker engine")
    parser.add_argument("--no-llm", action="store_true", help="skip Chat Completions summary")
    parser.add_argument("--once", type=str, default="", help="single utterance then exit")
    args = parser.parse_args()

    harness = CliHarness(use_llm=not args.no_llm)

    if args.once.strip():
        harness.handle_user_text(args.once)
        return

    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[cli] bye")
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "quit", "exit"):
            print("[cli] bye")
            break
        if line == "/interrupt":
            harness.interrupt()
            continue
        if line == "/pref":
            snap = harness.snapshot() or {}
            _print("preference", snap.get("preference"))
            continue
        if line == "/bundle":
            snap = harness.snapshot() or {}
            worker = snap.get("worker") or {}
            _print("last_bundle", worker.get("last_bundle"))
            continue
        if line == "/session":
            _print("session", harness.snapshot())
            continue
        harness.handle_user_text(line)
        # tiny pause so status lines flush before next prompt
        time.sleep(0.05)


if __name__ == "__main__":
    main()
