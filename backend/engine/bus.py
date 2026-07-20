from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable

from .events import Event

Listener = Callable[[Event], None]


class EventBus:
    """In-process pub/sub for Talker ↔ Worker coordination."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[Listener]] = defaultdict(list)
        self._any: list[Listener] = []

    def subscribe(self, event_type: str, listener: Listener) -> None:
        with self._lock:
            self._subs[event_type].append(listener)

    def subscribe_all(self, listener: Listener) -> None:
        with self._lock:
            self._any.append(listener)

    def emit(self, event: Event) -> None:
        with self._lock:
            listeners = list(self._subs.get(event.type, [])) + list(self._any)
        for fn in listeners:
            try:
                fn(event)
            except Exception as exc:  # keep bus alive
                print(f"[bus] listener error on {event.type}: {exc}")
