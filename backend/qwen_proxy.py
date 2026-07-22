"""Qwen-Omni-Realtime WebSocket upstream (DashScope / Model Studio).

Separate from OpenAI Realtime (`ws_proxy.connect_openai_realtime`).
Reuses low-level frame helpers from ws_proxy.
"""

from __future__ import annotations

import base64
import os
import ssl
from urllib.parse import urlparse

from ws_proxy import _tcp_via_proxy_or_direct  # noqa: PLC2701


def connect_qwen_omni_realtime(
    api_key: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 25.0,
) -> ssl.SSLSocket:
    """
    Connect to Qwen Omni Realtime.

    Env defaults:
      QWEN_REALTIME_URL = wss://dashscope.aliyuncs.com/api-ws/v1/realtime
      QWEN_OMNI_MODEL   = qwen3.5-omni-flash-realtime

    For Model Studio workspace endpoints, set e.g.:
      QWEN_REALTIME_URL=wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime
    """
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is empty")

    model_name = (
        model
        or os.environ.get("QWEN_OMNI_MODEL")
        or "qwen3.5-omni-flash-realtime"
    ).strip()
    raw_url = (
        base_url
        or os.environ.get("QWEN_REALTIME_URL")
        or "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    ).strip()
    if "://" not in raw_url:
        raw_url = "wss://" + raw_url
    parsed = urlparse(raw_url)
    host = parsed.hostname
    if not host:
        raise RuntimeError(f"invalid QWEN_REALTIME_URL: {raw_url}")
    port = parsed.port or 443
    path = parsed.path or "/api-ws/v1/realtime"
    query = parsed.query
    if "model=" not in query:
        q = f"model={model_name}"
        query = f"{query}&{q}" if query else q
    full_path = f"{path}?{query}" if query else path

    raw = _tcp_via_proxy_or_direct(host, port, timeout)
    ctx = ssl.create_default_context()
    ssock = ctx.wrap_socket(raw, server_hostname=host)
    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET {full_path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"Authorization: Bearer {key}\r\n"
        f"\r\n"
    )
    ssock.sendall(req.encode("utf-8"))
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = ssock.recv(4096)
        if not chunk:
            raise ConnectionError("Qwen closed during WebSocket handshake")
        buffer += chunk
    status_line = buffer.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
    if " 101 " not in status_line:
        raise ConnectionError(
            f"Qwen WebSocket handshake failed: {status_line} / {buffer[:400]!r}"
        )
    ssock.settimeout(None)
    return ssock


def default_omni_session_update(instructions: str) -> dict:
    """Official-ish Qwen Omni session.update payload (input 16 kHz PCM)."""
    # Qwen3.5-Omni-Realtime presets default to Tina; Cherry is for older Omni Flash.
    voice = (os.environ.get("QWEN_OMNI_VOICE") or "Tina").strip() or "Tina"
    model_name = (
        os.environ.get("QWEN_OMNI_MODEL") or "qwen3.5-omni-flash-realtime"
    ).strip()
    if voice.lower() == "cherry" and "3.5" in model_name:
        voice = "Tina"
    asr = (
        os.environ.get("QWEN_ASR_MODEL") or "qwen3-asr-flash-realtime"
    ).strip()
    vad = (os.environ.get("QWEN_TURN_DETECTION") or "server_vad").strip()
    turn: dict | None
    if vad in ("none", "manual", "off"):
        turn = None
    elif vad == "semantic_vad":
        turn = {
            "type": "semantic_vad",
            "threshold": 0.1,
            "prefix_padding_ms": 500,
            "silence_duration_ms": 900,
        }
    else:
        turn = {
            "type": "server_vad",
            "threshold": 0.4,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 800,
        }
    session: dict = {
        "modalities": ["text", "audio"],
        "voice": voice,
        "instructions": instructions,
        "input_audio_format": "pcm",
        "output_audio_format": "pcm",
        "input_audio_transcription": {"model": asr, "language": "en"},
    }
    if turn is not None:
        session["turn_detection"] = turn
    else:
        session["turn_detection"] = None
    return {"type": "session.update", "session": session}
