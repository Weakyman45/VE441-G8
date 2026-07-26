"""Minimal WebSocket helpers + OpenAI Realtime proxy (stdlib only)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
from typing import Callable, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlparse

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

TextHook = Callable[[str], Optional[Union[str, list[str]]]]


def ws_accept_key(sec_key: str) -> str:
    digest = hashlib.sha1((sec_key + WS_GUID).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        piece = sock.recv(n - got)
        if not piece:
            raise ConnectionError("socket closed while reading")
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exact(sock, 2)
    b0, b1 = header[0], header[1]
    opcode = b0 & 0x0F
    masked = (b1 & 0x80) != 0
    length = b1 & 0x7F
    if length == 126:
        length = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask_key = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def write_frame(sock: socket.socket, opcode: int, payload: bytes, *, mask: bool) -> None:
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))
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
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    sock.sendall(header + payload)


def _proxy_host_port() -> tuple[str, int] | None:
    """Use the same system HTTP(S) proxy that urllib sees (e.g. Clash 127.0.0.1:7890)."""
    import urllib.request
    from urllib.parse import urlparse as _urlparse

    proxies = urllib.request.getproxies()
    proxy_url = proxies.get("https") or proxies.get("http")
    if not proxy_url:
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url and "://" not in proxy_url:
            proxy_url = "http://" + proxy_url
    if not proxy_url:
        return None
    parsed = _urlparse(proxy_url)
    if not parsed.hostname or not parsed.port:
        return None
    return parsed.hostname, parsed.port


def _tcp_via_proxy_or_direct(host: str, port: int, timeout: float) -> socket.socket:
    proxy = _proxy_host_port()
    if proxy:
        phost, pport = proxy
        raw = socket.create_connection((phost, pport), timeout=timeout)
        raw.settimeout(timeout)
        connect_req = (
            f"CONNECT {host}:{port} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Proxy-Connection: Keep-Alive\r\n"
            f"\r\n"
        )
        raw.sendall(connect_req.encode("utf-8"))
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = raw.recv(4096)
            if not chunk:
                raise ConnectionError("Proxy closed during CONNECT")
            buffer += chunk
        status = buffer.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
        if " 200 " not in status:
            raise ConnectionError(f"Proxy CONNECT failed: {status}")
        return raw

    addrinfo = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    if not addrinfo:
        raise ConnectionError(f"No IPv4 address for {host}")
    family, socktype, proto, _, sockaddr = addrinfo[0]
    raw = socket.socket(family, socktype, proto)
    raw.settimeout(timeout)
    raw.connect(sockaddr)
    return raw


def connect_openai_realtime(api_key: str, model: str, timeout: float = 25.0) -> ssl.SSLSocket:
    base_url = os.environ.get("OPENAI_REALTIME_WS_BASE_URL", "wss://api.openai.com/v1/realtime").strip()
    parsed = urlparse(base_url)
    if parsed.scheme != "wss" or not parsed.hostname:
        raise ValueError(f"OPENAI_REALTIME_WS_BASE_URL must be a wss URL, got {base_url!r}")
    host = parsed.hostname
    port = parsed.port or 443
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["model"] = model
    path = parsed.path or "/v1/realtime"
    path = f"{path}?{urlencode(query)}"
    raw = _tcp_via_proxy_or_direct(host, port, timeout)
    ctx = ssl.create_default_context()
    ssock = ctx.wrap_socket(raw, server_hostname=host)
    sec_key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"Authorization: Bearer {api_key}\r\n"
        f"\r\n"
    )
    ssock.sendall(req.encode("utf-8"))
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = ssock.recv(4096)
        if not chunk:
            raise ConnectionError("OpenAI closed during WebSocket handshake")
        buffer += chunk
    status_line = buffer.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
    if " 101 " not in status_line:
        raise ConnectionError(f"OpenAI WebSocket handshake failed: {status_line} / {buffer[:300]!r}")
    ssock.settimeout(None)
    return ssock


class UpstreamSender:
    """Thread-safe helper so TalkerBridge can inject JSON events into OpenAI."""

    def __init__(self, upstream: socket.socket) -> None:
        self._upstream = upstream
        self._lock = threading.Lock()

    def send_json(self, obj: dict) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        with self._lock:
            write_frame(self._upstream, 0x1, payload, mask=True)


def relay_sockets(
    client: socket.socket,
    upstream: socket.socket,
    *,
    on_log: Callable[[str], None] | None = None,
    on_client_text: TextHook | None = None,
    on_upstream_text: TextHook | None = None,
    on_ready: Callable[[UpstreamSender], None] | None = None,
) -> None:
    """
    Bidirectional relay. Client=OkHttp (masked), upstream=OpenAI (server, unmasked).
    `on_ready` receives an UpstreamSender before the blocking join.
    """
    log = on_log or (lambda _m: None)
    stop = threading.Event()
    write_lock = threading.Lock()
    sender = UpstreamSender(upstream)
    sender._lock = write_lock  # noqa: SLF001 — share lock with pump writes

    def pump(src: socket.socket, dst: socket.socket, *, to_upstream: bool, name: str) -> None:
        try:
            while not stop.is_set():
                opcode, payload = read_frame(src)
                if opcode == 0x8:  # close
                    close_code = None
                    close_reason = ""
                    if len(payload) >= 2:
                        close_code = struct.unpack("!H", payload[:2])[0]
                        close_reason = payload[2:].decode("utf-8", errors="replace")
                    log(f"relay {name} close frame code={close_code} reason={close_reason}")
                    try:
                        if to_upstream:
                            with write_lock:
                                write_frame(dst, 0x8, payload, mask=True)
                        else:
                            write_frame(dst, 0x8, payload, mask=False)
                    except OSError:
                        pass
                    break
                if opcode == 0x9:  # ping -> pong
                    if to_upstream:
                        write_frame(src, 0xA, payload, mask=False)
                    else:
                        with write_lock:
                            write_frame(src, 0xA, payload, mask=True)
                    continue
                if opcode == 0xA:
                    continue
                if opcode in (0x1, 0x2, 0x0):
                    transformed: None | str | list[str] = None
                    if opcode == 0x1:
                        try:
                            text = payload.decode("utf-8")
                            if to_upstream and on_client_text:
                                transformed = on_client_text(text)
                            if not to_upstream and on_upstream_text:
                                on_upstream_text(text)
                        except Exception as hook_exc:
                            log(f"hook error ({name}): {hook_exc}")
                            transformed = None
                    if to_upstream:
                        outbound_payloads: list[bytes]
                        if opcode == 0x1 and transformed is not None:
                            if isinstance(transformed, str):
                                outbound_payloads = [transformed.encode("utf-8")] if transformed else []
                            else:
                                outbound_payloads = [item.encode("utf-8") for item in transformed if item]
                        else:
                            outbound_payloads = [payload]
                        with write_lock:
                            for outbound in outbound_payloads:
                                write_frame(dst, opcode, outbound, mask=True)
                    else:
                        write_frame(dst, opcode, payload, mask=False)
        except Exception as exc:
            log(f"relay {name} stopped: {exc}")
        finally:
            stop.set()
            try:
                src.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                dst.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    t1 = threading.Thread(
        target=pump, args=(client, upstream),
        kwargs={"to_upstream": True, "name": "app->openai"}, daemon=True,
    )
    t2 = threading.Thread(
        target=pump, args=(upstream, client),
        kwargs={"to_upstream": False, "name": "openai->app"}, daemon=True,
    )
    t1.start()
    t2.start()
    if on_ready:
        try:
            on_ready(sender)
        except Exception as exc:
            log(f"on_ready error: {exc}")
    t1.join()
    t2.join()
    try:
        client.close()
    except OSError:
        pass
    try:
        upstream.close()
    except OSError:
        pass
