"""Cross-cutting API protections: request IDs, body-size cap, security headers, rate limiting.

The rate limiter is in-process (one worker); it bounds abuse of a single instance and is not a
substitute for edge limits in a multi-instance deployment. ``X-Forwarded-For`` is deliberately not
trusted; behind a reverse proxy the client address must be configured explicitly (future work).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("factorymind.api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

SECURITY_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
    (b"cross-origin-resource-policy", b"same-origin"),
]


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after: int


class SlidingWindowLimiter:
    """Counts weighted events per key over a sliding window."""

    def __init__(self, window_s: float = 60.0) -> None:
        self._window = window_s
        self._events: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[tuple[float, int]]:
        q = self._events[key]
        while q and now - q[0][0] >= self._window:
            q.popleft()
        return q

    def _decision(self, q: deque[tuple[float, int]], limit: int, now: float) -> RateDecision:
        total = sum(w for _, w in q)
        if total < limit:
            return RateDecision(True, 0)
        retry = int(self._window - (now - q[0][0])) + 1 if q else 1
        return RateDecision(False, max(retry, 1))

    def allow(self, key: str, limit: int, cost: int = 1) -> RateDecision:
        """Count one request (of ``cost``) unless the limit is already reached."""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now)
            decision = self._decision(q, limit, now)
            if decision.allowed:
                q.append((now, cost))
            return decision

    def blocked(self, key: str, limit: int) -> RateDecision:
        """Check without counting."""
        now = time.monotonic()
        with self._lock:
            return self._decision(self._prune(key, now), limit, now)

    def record(self, key: str, cost: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now).append((now, cost))


class SecurityMiddleware:
    """Pure ASGI middleware: request ID, body cap (413) and response security headers."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self._max = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope["headers"]}
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self._max:
            await self._reject(send, request_id)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max:
                    # FastAPI re-raises its own HTTPException from body parsing (others become 400)
                    raise HTTPException(status_code=413, detail="Request body too large")
            return message

        started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                extra: list[tuple[bytes, bytes]] = [
                    *SECURITY_HEADERS,
                    (b"x-request-id", request_id.encode("ascii")),
                ]
                names = {n for n, _ in extra}
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() not in names]
                message = {**message, "headers": [*kept, *extra]}
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_headers)
        except Exception:
            log.exception("unhandled error (request %s)", request_id)
            if started:
                raise
            await self._reject(send, request_id, status=500, detail="Internal server error")

    async def _reject(
        self,
        send: Send,
        request_id: str,
        *,
        status: int = 413,
        detail: str = "Request body too large",
    ) -> None:
        body = json.dumps({"detail": detail, "request_id": request_id}).encode()
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-request-id", request_id.encode("ascii")),
            *SECURITY_HEADERS,
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def request_id_of(request: Any) -> str:
    return str(getattr(request.state, "request_id", "-"))
