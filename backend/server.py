"""VoiceShop++ backend — catalog API + Talker–Worker dual runtime.

Default Talker stack is GPT Realtime when OPENAI_API_KEY is available, with
Qwen Omni Realtime as the fallback (+ Qwen chat for Planner/Workers).

Run (from the project root; DB path is resolved relative to this file):
    python backend/server.py --port 8000

Endpoints:
    GET  /health
    GET  /api/v1/search
    GET  /api/v1/products/{id}
    POST /api/v1/session
    POST /api/v1/image
    GET  /api/v1/session/{id}
    GET  /api/v1/session/{id}/recommendations
    GET  /api/v1/realtime/ws?session_id=...        (default Talker = auto)
    GET  /api/v1/qwen/realtime/ws?session_id=...   (Qwen Omni Talker)
    GET  /api/v1/openai/realtime/ws?session_id=... (GPT Realtime)
    GET  /api/v1/realtime/token                    (legacy OpenAI ephemeral token)
    GET  /voice-test                               (PC browser voice call test page)
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from qwen_proxy import connect_qwen_omni_realtime, default_omni_session_update
from ws_proxy import UpstreamSender, connect_openai_realtime, relay_sockets, ws_accept_key

from engine.bus import EventBus
from engine.events import Event, EventType
from engine.intent import extract_preference
from engine.logging_store import LoggingStore
from engine.llm.analyze import analyze_need, analyze_need_stream
from engine.llm.vision import describe_shopping_image, visual_context_text
from engine.session import SessionStore
from engine.talker.bridge import TalkerBridge
from engine.talker.duplex_controls import (
    ExperimentEventLogger,
    HalfDuplexMicGate,
    half_duplex_mic_gate_requested,
)
from engine.talker.denoise import build_talker_denoiser, talker_denoise_requested
from engine.talker.local_realtime_vad import LocalRealtimeVadGate
from engine.talker.shopping_safety import SHOPPING_SAFETY_BOUNDARIES, shopping_safety_enabled
from engine.worker.runtime import WorkerRuntime

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "data", "catalog.db")
ENV_FILE = os.path.join(HERE, ".env")
FALLBACK_OPENAI_ENV_FILE = "/Users/YukiWang/Downloads/tau2-bench-dev-tau3/.env"
ENGINE_LOG_DB = os.path.join(HERE, "data", "engine_logs.db")
VOICE_TEST_HTML = os.path.join(HERE, "static", "voice_test.html")
UPLOAD_DIR = os.path.join(HERE, "data", "uploads")
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def load_dotenv(path: str = ENV_FILE, *, override: bool = True) -> None:
    """Load backend/.env into os.environ.

    override=True so local .env wins over stale shell / system env vars
    (e.g. an old QWEN_OMNI_VOICE=Cherry).
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        print(f"WARNING: could not read {path}: {exc}")


load_dotenv()
load_dotenv(FALLBACK_OPENAI_ENV_FILE, override=False)

# Re-read after dotenv so later code sees the file values.
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
QWEN_OMNI_MODEL = (
    os.environ.get("QWEN_OMNI_MODEL") or "qwen3.5-omni-flash-realtime"
).strip()
QWEN_OMNI_VOICE = (os.environ.get("QWEN_OMNI_VOICE") or "Tina").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
REALTIME_MODEL = (
    os.environ.get("OPENAI_REALTIME_MODEL_OVERRIDE")
    or os.environ.get("OPENAI_REALTIME_MODEL")
    or "gpt-realtime"
).strip()
OPENAI_REALTIME_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "marin").strip() or "marin"
OPENAI_TRANSCRIBE_MODEL = (
    os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip()
    or "gpt-4o-mini-transcribe"
)
TALKER_PROVIDER_REQUESTED = (os.environ.get("TALKER_PROVIDER") or "auto").strip().lower()
DUPLEX_EXPERIMENT_CONDITION = os.environ.get("DUPLEX_EXPERIMENT_CONDITION", "").strip()
DUPLEX_EXPERIMENT_LOG_DIR = os.environ.get("DUPLEX_EXPERIMENT_LOG_DIR", "").strip()

COLUMNS = [
    "id", "name", "price", "rating", "rating_number", "display", "performance",
    "battery", "weight_kg", "summary", "review_sentiment", "weakness",
    "reasons", "trade_offs", "store", "image_url", "platform",
]

DB_PATH = DEFAULT_DB


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


INTERRUPTED_MARKER = "[INTERRUPTED]"
UNSPOKEN_MARKER = "[UNSPOKEN]"

INTERRUPTION_INSTRUCTIONS = f"""
If a prior assistant message contains {INTERRUPTED_MARKER}, that marker is the cut-off point.
The user did not hear anything after that marker.
If the message also contains {UNSPOKEN_MARKER}, the text after it is unspoken
draft content from the interrupted reply. Treat it only as private context, not
as something the user heard.

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


def interruption_handling_enabled() -> bool:
    return env_bool("INTERRUPTION_HANDLING_ENABLED", True)


def build_shopping_instructions() -> str:
    base = (
        "You are VoiceShop++, a helpful retail shopping assistant (Talker). "
        "Always reply in English only — never switch to Chinese or other languages. "
        "Help shoppers clarify what they want to buy: product category, budget, must-haves, "
        "nice-to-haves, brand preferences, and constraints. "
        "Ask concise clarifying questions when critical fields are missing. "
        "A background Worker may search a catalog; when Worker notes arrive, summarize them "
        "naturally in English without inventing products. "
        "Keep spoken replies short so the user can interrupt."
    )
    if shopping_safety_enabled():
        base = f"{base}\n\n{SHOPPING_SAFETY_BOUNDARIES}"
    if not interruption_handling_enabled():
        return base
    return f"{base}\n\n{INTERRUPTION_INSTRUCTIONS}"


SHOPPING_INSTRUCTIONS = build_shopping_instructions()


def resolve_talker_provider(requested: str | None = None) -> str:
    """Resolve Talker provider with GPT Realtime preferred and Qwen as fallback."""
    wanted = (requested or TALKER_PROVIDER_REQUESTED or "auto").strip().lower()
    if wanted in ("gpt", "gpt-realtime", "gpt_realtime"):
        wanted = "openai"
    if wanted in ("auto", "default", ""):
        return "openai" if OPENAI_API_KEY else "qwen"
    if wanted == "openai" and not OPENAI_API_KEY and DASHSCOPE_API_KEY:
        return "qwen"
    if wanted == "qwen" and not DASHSCOPE_API_KEY and OPENAI_API_KEY:
        return "openai"
    return wanted


def talker_input_sample_rate(provider: str | None = None) -> int:
    return 24_000 if resolve_talker_provider(provider) == "openai" else 16_000


def local_vad_requested() -> bool:
    return env_bool("LOCAL_VAD_ENABLED", True)


def duplex_experiment_config(provider: str | None = None) -> dict:
    talker = resolve_talker_provider(provider)
    return {
        "condition": DUPLEX_EXPERIMENT_CONDITION or None,
        "talker": talker,
        "interruption_handling_enabled": interruption_handling_enabled(),
        "local_vad_enabled": local_vad_requested() and talker in ("openai", "qwen"),
        "half_duplex_mic_gate_enabled": half_duplex_mic_gate_requested(),
        "experiment_log_dir": DUPLEX_EXPERIMENT_LOG_DIR or None,
    }


def default_talker_ws_path() -> str:
    if resolve_talker_provider() == "openai":
        return "/api/v1/openai/realtime/ws"
    return "/api/v1/qwen/realtime/ws"

# Dual-runtime globals (initialized in main)
SESSION_STORE = SessionStore()
EVENT_BUS = EventBus()
LOG_STORE = LoggingStore(ENGINE_LOG_DB)
WORKER_RUNTIME: WorkerRuntime | None = None


def mint_realtime_token() -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before starting the server.")

    payload = {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": SHOPPING_INSTRUCTIONS,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                    "transcription": {"model": OPENAI_TRANSCRIBE_MODEL},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": OPENAI_REALTIME_VOICE,
                },
            },
        }
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OPENAI_API_BASE_URL}/realtime/client_secrets",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI client_secrets failed ({exc.code}): {detail}") from exc

    value = data.get("value")
    if not value:
        raise RuntimeError(f"OpenAI response missing ephemeral key: {data}")

    return {
        "value": value,
        "expires_at": data.get("expires_at"),
        "model": REALTIME_MODEL,
        "session": data.get("session"),
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    item = {key: row[key] for key in COLUMNS}
    for field in ("reasons", "trade_offs"):
        raw = item.get(field) or ""
        item[field] = [p.strip() for p in raw.split("||") if p.strip()]
    return item


def _session_image_data_urls(state, max_images: int = 3) -> list[str]:
    """Load the session's most recent uploaded image(s) from disk and return
    them as base64 data URLs, so the need-analysis LLM can consume the picture
    as multimodal input alongside the text (not just pre-extracted keywords)."""
    urls: list[str] = []
    for ref in (getattr(state, "image_refs", None) or [])[-max_images:]:
        path = ref.get("path")
        mime = ref.get("mime_type") or "image/jpeg"
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            continue
        b64 = base64.b64encode(raw).decode("ascii")
        urls.append(f"data:{mime};base64,{b64}")
    return urls


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def search(params: dict) -> list[dict]:
    q = (params.get("q", [""])[0] or "").strip().lower()
    max_price = _to_int(params.get("max_price", [None])[0])
    min_rating = _to_float(params.get("min_rating", [None])[0])
    sort = (params.get("sort", ["popular"])[0] or "popular").lower()
    limit = _to_int(params.get("limit", [60])[0]) or 60
    limit = max(1, min(limit, 200))

    where = []
    args: list = []
    if max_price:
        where.append("price > 0 AND price <= ?")
        args.append(max_price)
    if min_rating:
        where.append("rating >= ?")
        args.append(min_rating)

    tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) >= 3]
    # Keep more tokens (10) so specific product words survive alongside generic
    # ones; relevance ordering below decides what actually ranks.
    seen: set[str] = set()
    tokens = [t for t in tokens if t not in _STOPWORDS and not (t in seen or seen.add(t))][:10]

    order = {
        "price": "price > 0 DESC, price ASC",
        "rating": "rating DESC, rating_number DESC",
        "popular": "rating_number DESC",
    }.get(sort, "rating_number DESC")

    # Relevance = how many query tokens appear in the product name. Order by that
    # FIRST so an item matching several query words (e.g. a real "running shoe")
    # beats a merely-popular item that happens to share one generic word like
    # "breathable". Args must be laid out in SQL order: WHERE ... ORDER BY ... LIMIT.
    rel_args: list = []
    order_by = order
    if tokens:
        clause = " OR ".join(["LOWER(name) LIKE ?"] * len(tokens))
        where.append(f"({clause})")
        args.extend([f"%{t}%" for t in tokens])
        rel_expr = " + ".join(["(LOWER(name) LIKE ?)"] * len(tokens))
        rel_args = [f"%{t}%" for t in tokens]
        order_by = f"({rel_expr}) DESC, {order}"

    sql = "SELECT * FROM laptops"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by} LIMIT ?"
    args.extend(rel_args)
    args.append(limit)

    conn = _connect()
    try:
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_product(pid: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM laptops WHERE id = ?", (pid,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def count() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM laptops").fetchone()[0]
    finally:
        conn.close()


_STOPWORDS = {
    "the", "and", "for", "with", "under", "need", "want",
    "good", "budget", "care", "about", "that", "have", "give", "your", "some",
    "rmb", "usd", "dollar", "dollars",
    # Generic colors / adjectives / demographics: these match almost anything
    # in an English catalog (a "Black/White dress" matched a shoe query), so we
    # drop them from matching and let the real product nouns decide relevance.
    "white", "black", "red", "blue", "green", "yellow", "gray", "grey", "pink",
    "purple", "orange", "brown", "silver", "gold", "beige", "navy",
    "color", "colors", "colour", "colours", "multicolor",
    "breathable", "lightweight", "comfortable", "comfy", "casual", "waterproof",
    "durable", "premium", "quality", "soft", "warm", "cool", "versatile",
    "simple", "classic", "stylish", "fashion", "fashionable", "new", "best",
    "size", "sizes", "women", "womens", "woman", "men", "mens", "man",
    "unisex", "adult", "adults", "kids", "boys", "girls",
}


def _to_int(value) -> int | None:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "reference.jpg").strip("._")
    return clean[:80] or "reference.jpg"


def _image_ext(mime_type: str, filename: str) -> str:
    lower = filename.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if "png" in mime_type:
        return ".png"
    if "webp" in mime_type:
        return ".webp"
    return ".jpg"


def _dedup_append(values: list, item: str) -> None:
    item = (item or "").strip()
    if not item:
        return
    low = item.lower()
    if all(low != v.lower() for v in values):
        values.append(item)


def _merge_llm_analysis(profile, analysis: dict) -> None:
    """Fold an LLM shopping brief into the rule-based PreferenceProfile so the
    Worker (search + recommend) actually uses the model's understanding."""
    if analysis.get("budget"):
        profile.budget = analysis["budget"]
    if analysis.get("category"):
        profile.category = analysis["category"]
    if analysis.get("use_case"):
        profile.use_case = analysis["use_case"]
    if analysis.get("platform") in ("Windows", "macOS"):
        profile.platform = analysis["platform"]
    for m in analysis.get("must_haves", []):
        _dedup_append(profile.hard, m)
    for s in analysis.get("nice_to_haves", []):
        _dedup_append(profile.soft, s)
    # English catalog keywords drive the Worker search, so keep the freshest set.
    if analysis.get("search_keywords"):
        profile.search_keywords = list(analysis["search_keywords"])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_voice_test(self) -> None:
        try:
            with open(VOICE_TEST_HTML, "rb") as fh:
                body = fh.read()
        except OSError as exc:
            self._send(500, {"error": "voice_test_missing", "detail": str(exc)})
            return
        self._send_bytes(200, body, "text/html; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def _stream_line(self, obj: dict) -> None:
        """Write one newline-delimited JSON object to the open response stream."""
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _stream_analyze(self, sid: str, text: str) -> None:
        """Stream the LLM's need-analysis reasoning live, then a final brief.

        Wire format: one JSON object per line (the server runs HTTP/1.0, so the
        response body is delimited by connection close):
            {"type":"delta","text":"...partial reasoning..."}
            ...
            {"type":"done","analysis":{...},"preference":{...}}

        On success the final brief is folded into the session preference exactly
        like the blocking /text endpoint, so the Worker recommendations stay in
        sync whether the client used streaming or not.
        """
        state = SESSION_STORE.create_or_get(sid)
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        prior = state.preference.to_dict()
        session_images = _session_image_data_urls(state)
        analysis: dict | None = None
        try:
            for kind, payload in analyze_need_stream(text, prior=prior, images=session_images):
                if kind == "delta":
                    self._stream_line({"type": "delta", "text": payload})
                elif kind == "done":
                    analysis = payload
        except (BrokenPipeError, ConnectionResetError):
            return  # client paused / disconnected mid-stream
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            try:
                self._stream_line({"type": "error", "detail": str(exc)[:200]})
            except OSError:
                pass
            return

        try:
            combined_text = text
            if state.preference.visual_context:
                combined_text = (
                    f"{text}\n\nImage context already attached: "
                    f"{state.preference.visual_context}"
                )
            updated = extract_preference(combined_text, state.preference)
            if state.preference.visual_context and not updated.visual_context:
                updated.visual_context = state.preference.visual_context
            if analysis and analysis.get("provider") == "qwen":
                _merge_llm_analysis(updated, analysis)

            SESSION_STORE.update_preference(sid, updated)
            SESSION_STORE.append_turn(sid, "user", text)
            LOG_STORE.log_conversation(sid, "user", text)
            LOG_STORE.log_trace(sid, "text", "intent_updated", {
                **updated.to_dict(), "llm_provider": (analysis or {}).get("provider"),
            })
            EVENT_BUS.emit(Event(
                type=EventType.USER_INTENT_UPDATED,
                session_id=sid,
                payload={**updated.to_dict(), "source": {"type": "text_input"}},
            ))
            self._stream_line({
                "type": "done",
                "analysis": analysis or {},
                "preference": updated.to_dict(),
            })
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001 - stream already started; never fall through to _send(500)
            try:
                self._stream_line({"type": "error", "detail": str(exc)[:200]})
            except OSError:
                pass
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/v1/session":
                body = self._read_json()
                sid = body.get("session_id") or uuid.uuid4().hex[:16]
                state = SESSION_STORE.create_or_get(sid)
                talker = resolve_talker_provider()
                ws_base = default_talker_ws_path()
                self._send(200, {
                    "session_id": state.session_id,
                    "talker": talker,
                    "talker_requested": TALKER_PROVIDER_REQUESTED,
                    "input_audio_sample_rate": talker_input_sample_rate(talker),
                    "interruption_handling_enabled": interruption_handling_enabled(),
                    "duplex_experiment": duplex_experiment_config(talker),
                    "ws_url": f"{ws_base}?session_id={state.session_id}",
                })
            elif path == "/api/v1/image":
                body = self._read_json()
                sid = (body.get("session_id") or uuid.uuid4().hex[:16]).strip()
                filename = _safe_filename(body.get("filename") or "reference.jpg")
                mime_type = (body.get("mime_type") or "image/jpeg").strip()
                raw_b64 = (body.get("image_base64") or "").strip()
                if "," in raw_b64 and raw_b64.startswith("data:"):
                    raw_b64 = raw_b64.split(",", 1)[1]
                try:
                    image_bytes = base64.b64decode(raw_b64, validate=True)
                except (binascii.Error, ValueError):
                    self._send(400, {"error": "invalid_image_base64"})
                    return
                if not image_bytes:
                    self._send(400, {"error": "empty_image"})
                    return
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    self._send(413, {
                        "error": "image_too_large",
                        "max_bytes": MAX_IMAGE_BYTES,
                    })
                    return

                state = SESSION_STORE.create_or_get(sid)
                image_id = "img_" + uuid.uuid4().hex[:12]
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                ext = _image_ext(mime_type, filename)
                stored_path = os.path.join(UPLOAD_DIR, f"{sid}_{image_id}{ext}")
                with open(stored_path, "wb") as fh:
                    fh.write(image_bytes)

                # --- TEST MODE: image AI vision analysis DISABLED ---
                # The Qwen-VL keyword extraction is commented out on purpose so
                # we can verify the /text multimodal path: with no visual_context
                # / search_keywords written here, any image-derived signal in the
                # final brief can ONLY come from the image being fed directly into
                # the analysis LLM at /text. We still SAVE the file + register the
                # image_ref (path/mime_type) so _session_image_data_urls picks it
                # up for the multimodal call.
                # analysis = describe_shopping_image(
                #     image_bytes,
                #     mime_type=mime_type,
                #     filename=filename,
                #     user_text=body.get("user_text") or state.preference.raw_query,
                # )
                # visual_text = visual_context_text(analysis)
                # prior = state.preference
                # updated = extract_preference(visual_text, prior)
                # updated.visual_context = visual_text
                # img_keywords = analysis.get("search_keywords")
                # if isinstance(img_keywords, list) and img_keywords:
                #     updated.search_keywords = [str(k).strip() for k in img_keywords if str(k).strip()][:6]
                # if analysis.get("product_category") and not updated.category:
                #     cat = str(analysis["product_category"]).strip()
                #     if cat.isascii():
                #         updated.category = cat
                # if visual_text and all(visual_text.lower() not in s.lower() for s in updated.soft):
                #     updated.soft.append(f"Image reference: {visual_text[:160]}")
                # if prior.raw_query and prior.raw_query not in updated.raw_query:
                #     updated.raw_query = f"{prior.raw_query} {visual_text}".strip()
                # SESSION_STORE.update_preference(sid, updated)
                SESSION_STORE.add_image_ref(sid, {
                    "image_id": image_id,
                    "filename": filename,
                    "mime_type": mime_type,
                    "path": stored_path,
                    "summary": "",
                })
                SESSION_STORE.append_turn(sid, "user", "[image uploaded]")
                LOG_STORE.log_conversation(sid, "user", "[image uploaded]")
                LOG_STORE.log_trace(sid, "vision", "image_stored_no_analysis", {
                    "image_id": image_id,
                    "filename": filename,
                    "note": "vision analysis disabled for multimodal test",
                })
                self._send(200, {
                    "session_id": sid,
                    "image_id": image_id,
                    "visual_context": "",
                    "analysis": {"provider": "disabled", "note": "vision analysis commented out for test"},
                    "preference": state.preference.to_dict(),
                })
            elif path.startswith("/api/v1/session/") and path.endswith("/analyze_stream"):
                sid = path[len("/api/v1/session/"):-len("/analyze_stream")].strip("/")
                body = self._read_json()
                text = (body.get("text") or "").strip()
                if not sid:
                    self._send(400, {"error": "missing_session_id"})
                    return
                if len(text) < 2:
                    self._send(400, {"error": "empty_text"})
                    return
                self._stream_analyze(sid, text)
            elif path.startswith("/api/v1/session/") and path.endswith("/text"):
                sid = path[len("/api/v1/session/"):-len("/text")].strip("/")
                body = self._read_json()
                text = (body.get("text") or "").strip()
                if not sid:
                    self._send(400, {"error": "missing_session_id"})
                    return
                if len(text) < 2:
                    self._send(400, {"error": "empty_text"})
                    return

                state = SESSION_STORE.create_or_get(sid)
                combined_text = text
                if state.preference.visual_context:
                    combined_text = (
                        f"{text}\n\nImage context already attached: "
                        f"{state.preference.visual_context}"
                    )
                updated = extract_preference(combined_text, state.preference)
                if state.preference.visual_context and not updated.visual_context:
                    updated.visual_context = state.preference.visual_context

                # Every text (incl. transcribed voice) is analyzed by the LLM.
                # If the session has uploaded image(s), send them together with
                # the text as one multimodal turn (image is real input, not just
                # pre-extracted keywords).
                session_images = _session_image_data_urls(state)
                analysis = analyze_need(
                    text, prior=state.preference.to_dict(), images=session_images
                )
                if analysis.get("provider") == "qwen":
                    _merge_llm_analysis(updated, analysis)

                SESSION_STORE.update_preference(sid, updated)
                SESSION_STORE.append_turn(sid, "user", text)
                LOG_STORE.log_conversation(sid, "user", text)
                LOG_STORE.log_trace(sid, "text", "intent_updated", {
                    **updated.to_dict(), "llm_provider": analysis.get("provider"),
                })
                EVENT_BUS.emit(Event(
                    type=EventType.USER_INTENT_UPDATED,
                    session_id=sid,
                    payload={**updated.to_dict(), "source": {"type": "text_input"}},
                ))
                self._send(200, {
                    "session_id": sid,
                    "preference": updated.to_dict(),
                    "visual_context": updated.visual_context,
                    "analysis": analysis,
                })
            else:
                self._send(404, {"error": "unknown_endpoint", "path": path})
        except Exception as exc:
            self._send(500, {"error": "server_error", "detail": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path in (
            "/api/v1/realtime/ws",
            "/api/v1/qwen/realtime/ws",
            "/api/v1/openai/realtime/ws",
        ):
            if self.headers.get("Upgrade", "").lower() == "websocket":
                sid = (params.get("session_id") or [""])[0] or uuid.uuid4().hex[:16]
                if path.endswith("/openai/realtime/ws"):
                    provider = "openai"
                elif path.endswith("/qwen/realtime/ws"):
                    provider = "qwen"
                else:
                    provider = TALKER_PROVIDER_REQUESTED
                self._handle_realtime_ws(sid, provider=provider)
                return
            self._send(400, {"error": "websocket_upgrade_required"})
            return

        try:
            if path in ("/voice-test", "/voice_test", "/voice-test.html"):
                self._send_voice_test()
            elif path == "/health" or path == "":
                self._send(200, {
                    "status": "ok",
                    "count": count(),
                    "talker": resolve_talker_provider(),
                    "talker_requested": TALKER_PROVIDER_REQUESTED,
                    "qwen_key_configured": bool(DASHSCOPE_API_KEY),
                    "openai_key_configured": bool(OPENAI_API_KEY),
                    "openai_model": REALTIME_MODEL,
                    "realtime_ws": default_talker_ws_path(),
                    "qwen_ws": "/api/v1/qwen/realtime/ws",
                    "openai_ws": "/api/v1/openai/realtime/ws",
                    "duplex_experiment": duplex_experiment_config(),
                    "engine": "talker-worker",
                    "llm": "qwen",
                    "voice_test": "/voice-test",
                })
            elif path == "/api/v1/search":
                self._send(200, {"results": search(params)})
            elif path == "/api/v1/realtime/token":
                self._send(200, mint_realtime_token())
            elif path == "/api/v1/realtime/check":
                talker = resolve_talker_provider()
                self._send(200, {
                    "status": "ok",
                    "talker": talker,
                    "talker_requested": TALKER_PROVIDER_REQUESTED,
                    "qwen_key_configured": bool(DASHSCOPE_API_KEY),
                    "qwen_model": QWEN_OMNI_MODEL,
                    "openai_key_configured": bool(OPENAI_API_KEY),
                    "openai_model": REALTIME_MODEL,
                    "input_audio_sample_rate": talker_input_sample_rate(talker),
                    "interruption_handling_enabled": interruption_handling_enabled(),
                    "duplex_experiment": duplex_experiment_config(talker),
                    "qwen_realtime_url": os.environ.get(
                        "QWEN_REALTIME_URL",
                        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                    ),
                    "ws_url": default_talker_ws_path(),
                    "note": "This checks local backend configuration only; WebSocket still needs outbound network access to the selected realtime provider.",
                })
            elif path.startswith("/api/v1/session/") and path.endswith("/recommendations"):
                sid = path[len("/api/v1/session/"):-len("/recommendations")].strip("/")
                snap = SESSION_STORE.snapshot(sid)
                if not snap:
                    self._send(404, {"error": "session_not_found", "session_id": sid})
                else:
                    worker = snap.get("worker") or {}
                    self._send(200, {
                        "session_id": sid,
                        "status": worker.get("status"),
                        "message": worker.get("message"),
                        "bundle": worker.get("last_bundle"),
                    })
            elif path.startswith("/api/v1/session/"):
                sid = path.rsplit("/", 1)[-1]
                snap = SESSION_STORE.snapshot(sid)
                if not snap:
                    self._send(404, {"error": "session_not_found", "session_id": sid})
                else:
                    self._send(200, snap)
            elif path.startswith("/api/v1/products/"):
                pid = path.rsplit("/", 1)[-1]
                product = get_product(pid)
                if product:
                    self._send(200, product)
                else:
                    self._send(404, {"error": "not_found", "id": pid})
            else:
                self._send(404, {"error": "unknown_endpoint", "path": path})
        except Exception as exc:
            self._send(500, {"error": "server_error", "detail": str(exc)})

    def _handle_realtime_ws(self, session_id: str, *, provider: str) -> None:
        sec_key = self.headers.get("Sec-WebSocket-Key")
        if not sec_key:
            self._send(400, {"error": "missing_sec_websocket_key"})
            return

        requested_provider = (provider or "auto").strip().lower()
        provider = resolve_talker_provider(requested_provider)
        experiment_logger = ExperimentEventLogger(DUPLEX_EXPERIMENT_LOG_DIR)
        denoiser = None
        local_vad_gate: LocalRealtimeVadGate | None = None
        half_duplex_gate: HalfDuplexMicGate | None = None
        if provider == "openai":
            if not OPENAI_API_KEY:
                self._send(500, {"error": "OPENAI_API_KEY not configured"})
                return
            protocol = "openai.realtime"
            connect_fn = lambda: connect_openai_realtime(OPENAI_API_KEY, REALTIME_MODEL)
            upstream_error = "openai_upstream_failed"
        else:
            if not DASHSCOPE_API_KEY:
                self._send(500, {"error": "DASHSCOPE_API_KEY not configured"})
                return
            protocol = "qwen.omni.realtime"
            connect_fn = lambda: connect_qwen_omni_realtime(
                DASHSCOPE_API_KEY, model=QWEN_OMNI_MODEL
            )
            upstream_error = "qwen_upstream_failed"

        if local_vad_requested():
            try:
                local_vad_gate = LocalRealtimeVadGate(
                    input_sample_rate=talker_input_sample_rate(provider),
                )
                print(f"[local-vad] enabled for {provider} Realtime input")
            except Exception as exc:  # noqa: BLE001
                local_vad_gate = None
                print(f"[local-vad] unavailable, falling back to {provider} server_vad: {exc}")

        SESSION_STORE.require(session_id)
        if talker_denoise_requested():
            try:
                denoiser = build_talker_denoiser(talker_input_sample_rate(provider))
                print(f"[denoise] enabled provider={provider}")
            except Exception as exc:  # noqa: BLE001
                denoiser = None
                print(f"[denoise] unavailable, forwarding raw audio: {exc}")
        if half_duplex_mic_gate_requested():
            half_duplex_gate = HalfDuplexMicGate(
                session_id=session_id,
                input_sample_rate=talker_input_sample_rate(provider),
                logger=experiment_logger,
            )
            print("[duplex-exp] half-duplex mic gate enabled")
        bridge = TalkerBridge(
            session_id, EVENT_BUS, SESSION_STORE, LOG_STORE, protocol=protocol
        )

        print(f"[ws] upgrade provider={provider} session={session_id} from {self.address_string()}")
        experiment_logger.log(
            session_id=session_id,
            action="ws_open",
            detail={
                **duplex_experiment_config(provider),
                "provider": provider,
                "protocol": protocol,
                "input_sample_rate": talker_input_sample_rate(provider),
                "output_sample_rate": 24_000 if provider in ("openai", "qwen") else None,
            },
        )
        try:
            upstream = connect_fn()
        except Exception as exc:
            print(f"[ws] upstream connect failed ({provider}): {exc}")
            self._send(502, {"error": upstream_error, "detail": str(exc)})
            return

        accept = ws_accept_key(sec_key)
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True

        client = self.connection

        def on_ready(sender: UpstreamSender) -> None:
            bridge.bind_sender(sender.send_json)
            if provider == "qwen":
                update = default_omni_session_update(SHOPPING_INSTRUCTIONS)
                if local_vad_gate is not None:
                    update.setdefault("session", {})["turn_detection"] = None
                # Force voice from current env / .env (never leave stale Cherry).
                voice = (os.environ.get("QWEN_OMNI_VOICE") or "Tina").strip() or "Tina"
                if voice.lower() == "cherry" and "3.5" in QWEN_OMNI_MODEL:
                    voice = "Tina"
                update.setdefault("session", {})["voice"] = voice
                print(f"[ws] session.update voice={voice} model={QWEN_OMNI_MODEL}")
                sender.send_json(update)
            else:
                turn_detection = None
                if local_vad_gate is None:
                    turn_detection = {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                    }
                sender.send_json({
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["audio"],
                        "voice": OPENAI_REALTIME_VOICE,
                        "instructions": SHOPPING_INSTRUCTIONS,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": OPENAI_TRANSCRIBE_MODEL,
                            "language": "en",
                        },
                        "turn_detection": turn_detection,
                    },
                })
            experiment_logger.log(
                session_id=session_id,
                action="session_update_sent",
                detail=duplex_experiment_config(provider),
            )

        def handle_client_text(text: str) -> None | str | list[str]:
            experiment_logger.log_frame(session_id=session_id, direction="client", text=text)
            frames = [text]
            if denoiser is not None:
                denoised_frames: list[str] = []
                for frame_text in frames:
                    transformed = denoiser.process_client_text(frame_text)
                    if transformed is None:
                        denoised_frames.append(frame_text)
                    else:
                        denoised_frames.extend(transformed)
                frames = denoised_frames
                if not frames:
                    return []
            if half_duplex_gate is not None:
                gated_frames: list[str] = []
                for frame_text in frames:
                    transformed = half_duplex_gate.process_client_text(frame_text)
                    if transformed is None:
                        gated_frames.append(frame_text)
                    elif isinstance(transformed, str):
                        gated_frames.append(transformed)
                    else:
                        gated_frames.extend(transformed)
                frames = gated_frames
                if not frames:
                    return []
            if local_vad_gate is not None:
                vad_frames: list[str] = []
                local_vad_changed = False
                for frame_text in frames:
                    bridge.on_client_text(frame_text)
                    transformed = local_vad_gate.process_client_text(frame_text)
                    if transformed is None:
                        vad_frames.append(frame_text)
                    else:
                        if local_vad_gate.consume_barge_in_cancel_pending():
                            bridge.record_local_barge_in_cancel(
                                source={
                                    "type": "local_vad_barge_in",
                                    "protocol": protocol,
                                }
                            )
                        local_vad_changed = True
                        experiment_logger.log(
                            session_id=session_id,
                            action="local_vad_transform",
                            detail=_summarize_transformed_frames(frame_text, transformed),
                        )
                        vad_frames.extend(transformed)
                if local_vad_changed:
                    return vad_frames
                return None if vad_frames == [text] else vad_frames
            for frame_text in frames:
                bridge.on_client_text(frame_text)
            return None if frames == [text] else frames

        def handle_upstream_text(text: str) -> None:
            experiment_logger.log_frame(session_id=session_id, direction="server", text=text)
            if half_duplex_gate is not None:
                half_duplex_gate.process_upstream_text(text)
            if local_vad_gate is not None:
                local_vad_gate.process_upstream_text(text)
            bridge.on_upstream_text(text)

        try:
            relay_sockets(
                client,
                upstream,
                on_log=lambda m: print(f"[ws:{provider}:{session_id[:8]}] {m}"),
                on_client_text=handle_client_text,
                on_upstream_text=handle_upstream_text,
                on_ready=on_ready,
            )
        finally:
            experiment_logger.log(
                session_id=session_id,
                action="ws_close",
                detail={"provider": provider},
            )
            print(f"[ws] session closed provider={provider} id={session_id}")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[req] {self.address_string()} {fmt % args}")


def _summarize_transformed_frames(input_text: str, frames: list[str]) -> dict:
    try:
        input_frame = json.loads(input_text)
    except json.JSONDecodeError:
        input_type = "invalid_json"
    else:
        input_type = input_frame.get("type")

    output_types: dict[str, int] = {}
    output_audio_bytes = 0
    for raw in frames:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            output_types["invalid_json"] = output_types.get("invalid_json", 0) + 1
            continue
        frame_type = str(frame.get("type") or "")
        output_types[frame_type] = output_types.get(frame_type, 0) + 1
        if frame_type == "input_audio_buffer.append":
            audio = frame.get("audio")
            if isinstance(audio, str) and audio:
                try:
                    output_audio_bytes += len(base64.b64decode(audio, validate=False))
                except Exception:
                    pass
    return {
        "input_type": input_type,
        "output_frame_count": len(frames),
        "output_types": output_types,
        "output_audio_bytes": output_audio_bytes,
    }


def main() -> None:
    global DB_PATH, WORKER_RUNTIME
    parser = argparse.ArgumentParser(description="VoiceShop++ Talker–Worker API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()

    DB_PATH = args.db
    if not os.path.exists(DB_PATH):
        raise SystemExit(
            f"catalog db not found: {DB_PATH}\nBuild it first with tools/build_laptops.py"
        )

    WORKER_RUNTIME = WorkerRuntime(EVENT_BUS, SESSION_STORE, LOG_STORE, search)
    WORKER_RUNTIME.start()

    total = count()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"VoiceShop++ backend serving {total} laptops")
    print(f"Listening on http://{args.host}:{args.port}  (db={DB_PATH})")
    print(
        f"Talker provider: {resolve_talker_provider()} "
        f"(requested={TALKER_PROVIDER_REQUESTED}; LLM agents: Qwen)"
    )
    print("Engine: Talker + Worker(Planner→Search→Recommend)")
    print("REST: /api/v1/session  /api/v1/session/{id}/recommendations")
    print(f"WS default: {default_talker_ws_path()}?session_id=...")
    print("WS Qwen:    /api/v1/qwen/realtime/ws")
    print("WS OpenAI:  /api/v1/openai/realtime/ws")
    print("PC voice:   http://127.0.0.1:%s/voice-test" % args.port)
    if os.path.isfile(ENV_FILE):
        print(f"Loaded local secrets from {ENV_FILE}")
    if os.path.isfile(FALLBACK_OPENAI_ENV_FILE):
        print(f"Loaded fallback OpenAI secrets from {FALLBACK_OPENAI_ENV_FILE}")
    if DASHSCOPE_API_KEY:
        print(f"Qwen Omni enabled (model={QWEN_OMNI_MODEL}, voice={QWEN_OMNI_VOICE})")
    else:
        print(f"WARNING: DASHSCOPE_API_KEY not set — create {ENV_FILE}")
    if OPENAI_API_KEY:
        print(f"OpenAI Realtime available (model={REALTIME_MODEL})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
