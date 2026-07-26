"""Lightweight shopping memory: session working memory + optional user facts.

Planner reads (prefill / resolve refs). Executor emits MEMORY_WRITE_REQUESTED;
Runtime applies writes via apply_writes. No vector RAG.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from .exp_config import CONFIG, ExpConfig
from .llm.qwen_client import chat_json, qwen_configured
from .models import PreferenceProfile, RankedProduct, RecommendationBundle, SessionState

DEFAULT_MEMORY_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "memory.db",
)

_ORDINAL_ZH = {
    "第一": 0,
    "第1": 0,
    "第二个": 1,
    "第二": 1,
    "第2": 1,
    "第三个": 2,
    "第三": 2,
    "第3": 2,
    "第四个": 3,
    "第四": 3,
    "第4": 3,
    "第五个": 4,
    "第五": 4,
    "第5": 4,
}
_ORDINAL_EN = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}


@dataclass
class SessionMemory:
    preference_slots: dict[str, Any] = field(default_factory=dict)
    last_ranked: list[dict[str, str]] = field(default_factory=list)
    memory_summary: str = ""
    mentioned_product_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    def __init__(self, db_path: str = DEFAULT_MEMORY_DB) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionMemory] = {}
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.isolation_level = None  # autocommit; avoid lingering locks on Windows
        return conn

    def close(self) -> None:
        """Best-effort hook for tests; connections are per-call and not pooled."""
        with self._lock:
            self._sessions.clear()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT PRIMARY KEY,
                    facts_json TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    updated_ts REAL NOT NULL DEFAULT 0
                )
                """
            )
        finally:
            conn.close()

    def get_session(self, session_id: str) -> SessionMemory:
        with self._lock:
            mem = self._sessions.get(session_id)
            if mem is None:
                mem = SessionMemory()
                self._sessions[session_id] = mem
            return mem

    def load_user_facts(self, user_id: str) -> dict[str, Any]:
        if not user_id:
            return {}
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT facts_json FROM user_memory WHERE user_id=?",
                    (user_id,),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return {}
        try:
            data = json.loads(row[0] or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def prefill(
        self,
        session: SessionState,
        *,
        user_id: str | None = None,
        opted_in_memory: bool = False,
        config: ExpConfig | None = None,
    ) -> dict[str, Any]:
        """Fill empty preference fields from session/user memory. Returns meta."""
        cfg = config or CONFIG
        if not cfg.memory:
            return {"enabled": False, "prefilled_keys": []}

        pref = session.preference
        mem = self.get_session(session.session_id)
        slots = dict(mem.preference_slots)
        if opted_in_memory and user_id:
            slots = {**self.load_user_facts(user_id), **slots}

        filled: list[str] = []
        for key in (
            "category",
            "budget",
            "use_case",
            "platform",
            "touch",
            "hard",
            "soft",
            "search_keywords",
            "visual_context",
        ):
            current = getattr(pref, key, None)
            if not _is_empty(current) or key not in slots:
                continue
            value = slots[key]
            if key == "budget":
                try:
                    setattr(pref, key, int(value) if value is not None else None)
                except (TypeError, ValueError):
                    continue
            elif key in ("hard", "soft", "search_keywords"):
                if isinstance(value, list):
                    setattr(pref, key, [str(x) for x in value])
                else:
                    continue
            else:
                setattr(pref, key, value)
            filled.append(key)

        if mem.memory_summary and not session.memory_summary:
            session.memory_summary = mem.memory_summary
        elif session.memory_summary and not mem.memory_summary:
            mem.memory_summary = session.memory_summary

        return {
            "enabled": True,
            "prefilled_keys": filled,
            "opted_in": bool(opted_in_memory and user_id),
        }

    def resolve_refs(
        self,
        utterance: str,
        session: SessionState,
        *,
        config: ExpConfig | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        cfg = config or CONFIG
        if not cfg.memory:
            return {"enabled": False, "product_ids": [], "method": "off"}

        ranked = self._ranked_view(session)
        if not ranked:
            return {"enabled": True, "product_ids": [], "method": "none"}

        text = (utterance or "").strip()
        rule_ids = self._resolve_refs_rules(text, ranked, session.session_id)
        if rule_ids:
            return {
                "enabled": True,
                "product_ids": rule_ids,
                "method": "rules",
                "ranked_count": len(ranked),
            }

        if use_llm and cfg.planner_llm and qwen_configured() and text:
            llm_ids = self._resolve_refs_llm(text, ranked)
            if llm_ids:
                return {
                    "enabled": True,
                    "product_ids": llm_ids,
                    "method": "llm",
                    "ranked_count": len(ranked),
                }

        return {
            "enabled": True,
            "product_ids": [],
            "method": "miss",
            "ranked_count": len(ranked),
        }

    def remember_bundle(self, session_id: str, bundle: RecommendationBundle) -> None:
        mem = self.get_session(session_id)
        mem.last_ranked = [
            {"id": r.id, "name": r.name, "index": str(i)}
            for i, r in enumerate(bundle.ranked[:20])
        ]

    def sync_preference_slots(self, session_id: str, pref: PreferenceProfile) -> None:
        mem = self.get_session(session_id)
        slots: dict[str, Any] = {}
        for key in (
            "category",
            "budget",
            "use_case",
            "platform",
            "touch",
            "hard",
            "soft",
            "search_keywords",
            "visual_context",
        ):
            val = getattr(pref, key, None)
            if not _is_empty(val):
                slots[key] = val
        mem.preference_slots.update(slots)

    def apply_writes(
        self,
        *,
        session: SessionState,
        writes: list[dict[str, Any]],
        user_id: str | None = None,
        opted_in_memory: bool = False,
        config: ExpConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or CONFIG
        if not cfg.memory:
            return {"applied": 0, "enabled": False}

        applied = 0
        mem = self.get_session(session.session_id)
        user_updates: dict[str, Any] = {}

        for item in writes or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            scope = str(item.get("scope") or "session").strip().lower()
            value = item.get("value")
            if not key:
                continue
            if key == "last_ranked" and isinstance(value, list):
                mem.last_ranked = [
                    {"id": str(x.get("id") or ""), "name": str(x.get("name") or ""), "index": str(x.get("index", i))}
                    for i, x in enumerate(value)
                    if isinstance(x, dict) and x.get("id")
                ][:20]
                applied += 1
            elif key == "memory_summary":
                summary = str(value or "").strip()
                mem.memory_summary = summary
                session.memory_summary = summary
                applied += 1
            elif key == "preference_slots" and isinstance(value, dict):
                mem.preference_slots.update(value)
                applied += 1
            elif key == "mentioned_product_id":
                mem.mentioned_product_id = str(value or "")
                applied += 1
            elif scope == "user":
                if opted_in_memory and user_id:
                    user_updates[key] = value
                    applied += 1
            else:
                mem.preference_slots[key] = value
                applied += 1

        if user_updates and opted_in_memory and user_id:
            self._merge_user_facts(user_id, user_updates, mem.memory_summary)

        return {
            "applied": applied,
            "enabled": True,
            "user_writes": len(user_updates),
        }

    def build_summary_writes(
        self,
        session: SessionState,
        bundle: RecommendationBundle,
        *,
        use_llm: bool = True,
    ) -> list[dict[str, Any]]:
        """Produce writeback intents after a successful recommendation."""
        ranked_payload = [
            {"id": r.id, "name": r.name, "index": i}
            for i, r in enumerate(bundle.ranked[:10])
        ]
        writes: list[dict[str, Any]] = [
            {"key": "last_ranked", "value": ranked_payload, "scope": "session"},
            {
                "key": "preference_slots",
                "value": _slots_from_pref(session.preference),
                "scope": "session",
            },
        ]
        summary = ""
        if use_llm and qwen_configured() and CONFIG.planner_llm:
            summary = self._llm_summary(session, bundle)
        if not summary:
            summary = _rule_summary(session, bundle)
        writes.append({"key": "memory_summary", "value": summary, "scope": "session"})
        # Stable facts for opted-in user scope (Runtime decides whether to honor).
        slots = _slots_from_pref(session.preference)
        for key in ("category", "budget", "platform", "hard"):
            if key in slots:
                writes.append({"key": key, "value": slots[key], "scope": "user"})
        return writes

    def _ranked_view(self, session: SessionState) -> list[dict[str, str]]:
        mem = self.get_session(session.session_id)
        if mem.last_ranked:
            return list(mem.last_ranked)
        bundle = session.worker.last_bundle
        if not bundle:
            return []
        return [
            {"id": r.id, "name": r.name, "index": str(i)}
            for i, r in enumerate(bundle.ranked[:20])
        ]

    def _resolve_refs_rules(
        self,
        text: str,
        ranked: list[dict[str, str]],
        session_id: str,
    ) -> list[str]:
        if not text or not ranked:
            return []
        lower = text.lower()

        for phrase, idx in _ORDINAL_ZH.items():
            if phrase in text and idx < len(ranked):
                return [ranked[idx]["id"]]
        for phrase, idx in _ORDINAL_EN.items():
            if re.search(rf"\b{re.escape(phrase)}\b", lower) and idx < len(ranked):
                return [ranked[idx]["id"]]

        m = re.search(r"第\s*(\d+)\s*个", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(ranked):
                return [ranked[idx]["id"]]
        m = re.search(r"\b(?:option|number|#)\s*(\d+)\b", lower)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(ranked):
                return [ranked[idx]["id"]]

        if any(p in text for p in ("这个", "刚才那个", "刚刚那个", "上一个")) or any(
            p in lower for p in ("this one", "that one", "the last one")
        ):
            mem = self.get_session(session_id)
            if mem.mentioned_product_id:
                return [mem.mentioned_product_id]
            return [ranked[0]["id"]]

        # Direct name substring match
        hits = []
        for item in ranked:
            name = (item.get("name") or "").lower()
            if name and name in lower:
                hits.append(item["id"])
        return hits[:3]

    def _resolve_refs_llm(self, text: str, ranked: list[dict[str, str]]) -> list[str]:
        try:
            data = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You resolve product references in a shopping dialogue. "
                            "Return ONLY JSON: {\"product_ids\": [\"...\"]}. "
                            "Pick from the provided ranked list ids only. "
                            "If unclear, return an empty list."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"utterance={text}\nranked={ranked[:10]}",
                    },
                ],
                temperature=0.0,
            )
        except Exception:
            return []
        ids = data.get("product_ids") if isinstance(data, dict) else None
        if not isinstance(ids, list):
            return []
        allowed = {r["id"] for r in ranked}
        return [str(x) for x in ids if str(x) in allowed][:5]

    def _llm_summary(self, session: SessionState, bundle: RecommendationBundle) -> str:
        top = [
            {"name": r.name, "price": r.price, "id": r.id}
            for r in bundle.ranked[:3]
        ]
        try:
            data = chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize durable shopping memory in 1-2 short sentences. "
                            "Return ONLY JSON: {\"memory_summary\": \"...\"}. "
                            "Capture stable preferences, not ephemeral chatter."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"preference={session.preference.to_dict()}\n"
                            f"prior_summary={session.memory_summary}\n"
                            f"top={top}"
                        ),
                    },
                ],
                temperature=0.2,
            )
            if isinstance(data, dict):
                return str(data.get("memory_summary") or "").strip()[:400]
        except Exception:
            return ""
        return ""

    def _merge_user_facts(self, user_id: str, updates: dict[str, Any], summary: str) -> None:
        import time

        with self._lock:
            facts = self.load_user_facts(user_id)
            facts.update(updates)
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO user_memory(user_id, facts_json, summary, updated_ts)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        facts_json=excluded.facts_json,
                        summary=excluded.summary,
                        updated_ts=excluded.updated_ts
                    """,
                    (
                        user_id,
                        json.dumps(facts, ensure_ascii=False),
                        summary or "",
                        time.time(),
                    ),
                )
            finally:
                conn.close()


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() in ("No preference", "Not required")
    if isinstance(value, list):
        return len(value) == 0
    return False


def _slots_from_pref(pref: PreferenceProfile) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "category",
        "budget",
        "use_case",
        "platform",
        "touch",
        "hard",
        "soft",
        "search_keywords",
        "visual_context",
    ):
        val = getattr(pref, key, None)
        if not _is_empty(val):
            out[key] = val
    return out


def _rule_summary(session: SessionState, bundle: RecommendationBundle) -> str:
    pref = session.preference
    bits = []
    if pref.category:
        bits.append(f"looking for {pref.category}")
    if pref.budget:
        bits.append(f"budget around {pref.budget}")
    if bundle.ranked:
        bits.append(f"last top pick {bundle.ranked[0].name}")
    return "; ".join(bits) if bits else (session.memory_summary or "active shopping session")


# Process-wide default store (tests may construct their own).
MEMORY_STORE = MemoryStore()
