from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any


class LoggingStore:
    """SQLite conversation + agent_trace logs for Phase 1 observability."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        ts REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS agent_trace (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        plan_id TEXT,
                        agent TEXT NOT NULL,
                        action TEXT NOT NULL,
                        detail TEXT,
                        latency_ms REAL,
                        ts REAL NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def log_conversation(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO conversation_log(session_id, role, content, ts) VALUES (?,?,?,?)",
                    (session_id, role, content, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    def log_trace(
        self,
        session_id: str,
        agent: str,
        action: str,
        detail: Any = None,
        latency_ms: float | None = None,
        plan_id: str = "",
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO agent_trace(session_id, plan_id, agent, action, detail, latency_ms, ts)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (
                        session_id,
                        plan_id,
                        agent,
                        action,
                        json.dumps(detail, ensure_ascii=False) if detail is not None else None,
                        latency_ms,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
