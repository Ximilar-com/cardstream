"""Ximilar stream sessions — an optional record of the show on the Ximilar platform.

Off unless ``--ximilar-stream`` is passed: without it the identify call is
still the only thing that leaves this machine. With it, every identification
that survives the result threshold is ALSO queued for the session API
(``/cardstream/v2/`` on api.ximilar.com, same API key), so the show can be
reviewed afterwards — what was shown, when, how sure the match was and what
it was worth.

Three pieces, each testable without a network:

* :func:`session_item` — one analyzer identification as the upload item the
  API validates: renamed fields, text clipped to the API's limits, links
  reduced to plain text. One malformed field would reject the whole batch, so
  this is where the payload is made safe, once.
* :class:`SessionApi` — the four HTTP calls (create, get, upload, close) and
  the one rule that sorts every upload reply into what happens next.
* :class:`SessionRecorder` — a queue drained in batches by a background
  thread. Every item carries a random ``event_id``, and the API skips the ids
  a session already has, so a batch whose reply was lost is simply sent
  again. A network error, 429 or 5xx keeps the items and backs off; a
  rejected batch (400) is dropped and logged; a closed, forbidden or missing
  session stops the uploads — never the show.
"""

from __future__ import annotations

import enum
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from cardstream.core.identify_client import DEFAULT_HTTP_TIMEOUT, auth_headers

DEFAULT_SESSION_URL = "https://api.ximilar.com/cardstream/v2"
# The --ximilar-stream value that starts a session instead of resuming one.
NEW_SESSION = "NEW"
# The API's platform choices; "other" is its default too.
PLATFORMS = ("whatnot", "tiktok", "ebay", "fanatics", "youtube", "twitch", "other")
DEFAULT_PLATFORM = "other"

# Per upload request — the API refuses a larger batch outright.
MAX_BATCH = 500
# Identifications kept while the API is unreachable. Beyond this the OLDEST
# go first: a show that outlives a long outage keeps its most recent cards.
MAX_PENDING = 5000
DEFAULT_FLUSH_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
# How long a clean exit waits for the last uploads before giving up.
DEFAULT_CLOSE_TIMEOUT = 10.0

# CardstreamIdentification's column limits. A longer value is a 400 for the
# whole batch, so it is clipped here rather than discovered there.
_TEXT_LIMITS = {
    "name": 255,
    "full_name": 500,
    "set_name": 255,
    "set_code": 50,
    "card_number": 50,
    "series": 255,
    "year": 20,
    "subcategory": 100,
}
_CONFIDENCE = frozenset({"high", "medium", "low"})
_MAX_LIST = 10
_MAX_LINK = 2048
_SESSION_NAME_LIMIT = 255
_GAME_LIMIT = 100


class SessionError(Exception):
    """A session could not be started or resumed — user-facing, at startup."""


def parse_session_spec(value: str) -> str:
    """``NEW`` (any case) or a session id, canonicalised; ValueError otherwise."""
    text = value.strip()
    if text.upper() == NEW_SESSION:
        return NEW_SESSION
    try:
        return str(uuid.UUID(text))
    except ValueError:
        raise ValueError(
            f"{value!r} is neither {NEW_SESSION} nor a session id"
        ) from None


def _text(value: object, limit: int) -> str:
    if value is None or isinstance(value, dict | list):
        return ""
    return str(value)[:limit]


def session_item(
    ident: dict[str, Any], id_type: str, seen: float, event_id: str | None = None
) -> dict[str, Any]:
    """One analyzer identification as an upload item.

    ``ident`` is the flattened dict the identify target returns
    (``Identification.to_dict()`` plus the analyzer's ``elapsed_ms``);
    ``seen`` is the wall-clock time the call fired, in epoch seconds. The two
    renames — ``set`` → ``set_name``, ``confidence_tier`` → ``confidence`` —
    are the API's names. Anything the API would reject is dropped or clipped
    instead, because one bad field fails the whole batch.
    """
    item: dict[str, Any] = {
        "event_id": event_id or uuid.uuid4().hex,
        "seen": datetime.fromtimestamp(seen, tz=UTC).isoformat(),
        "id_type": id_type,
    }
    source = {**ident, "set_name": ident.get("set")}
    for key, limit in _TEXT_LIMITS.items():
        item[key] = _text(source.get(key), limit)

    distance = ident.get("distance")
    if (
        isinstance(distance, int | float)
        and not isinstance(distance, bool)
        and math.isfinite(distance)
        and distance >= 0
    ):
        item["distance"] = float(distance)

    tier = ident.get("confidence_tier")
    tier = getattr(tier, "value", tier)
    if tier in _CONFIDENCE:
        item["confidence"] = tier

    links = ident.get("links")
    if isinstance(links, dict):
        item["links"] = {
            str(name): url[:_MAX_LINK]
            for name, url in links.items()
            if isinstance(url, str)
        }
    for key in ("alternatives", "price_stats"):
        entries = ident.get(key)
        if isinstance(entries, list):
            item[key] = [e for e in entries if isinstance(e, dict)][:_MAX_LIST]
    return item


class Outcome(enum.Enum):
    """What an upload reply means for the items that were in it."""

    STORED = "stored"  # 2xx: the API has them (created or already known)
    RETRY = "retry"  # network, 429, 5xx: keep them, try again later
    REJECTED = "rejected"  # 400 and friends: this batch will never pass
    STOPPED = "stopped"  # closed / forbidden / gone: no upload will pass


@dataclass(frozen=True)
class UploadReply:
    outcome: Outcome
    detail: str = ""
    created: int = 0
    duplicates: int = 0


# Status codes after which no later upload can succeed for this session.
_STOP_STATUSES = frozenset({401, 403, 404, 409})


def _detail(response: requests.Response) -> str:
    """The API's own explanation, when it gave one — else the start of the body."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return f"HTTP {response.status_code}: {body['detail']}"
    return f"HTTP {response.status_code}: {str(body)[:200]}"


class SessionApi:
    """The session endpoints, with the same auth headers as the identify call.

    Stateless ``requests`` calls per request, like the identify client: a
    batch goes out every few seconds at most, so a pooled Session buys
    nothing.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_SESSION_URL,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        if not api_key:
            raise SessionError(
                "--ximilar-stream needs a Ximilar API key — set XIMILAR_API_KEY "
                "or pass --api-key"
            )
        self._headers = auth_headers(api_key)
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def url(self, *parts: str) -> str:
        return "/".join([self.base_url, *parts]) + "/"

    def _send(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> requests.Response:
        return requests.request(
            method, url, json=payload, headers=self._headers, timeout=self._timeout
        )

    def _session(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """A create/get call whose failure should stop startup with a reason."""
        try:
            response = self._send(method, url, payload)
        except requests.RequestException as exc:
            raise SessionError(f"session API unreachable ({url}): {exc}") from None
        if not response.ok:
            raise SessionError(f"session API refused ({url}): {_detail(response)}")
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("id"), str):
            raise SessionError(f"session API sent no session ({url})")
        return body

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._session("POST", self.url("session"), payload)

    def get(self, session_id: str) -> dict[str, Any]:
        return self._session("GET", self.url("session", session_id))

    def upload(self, session_id: str, items: list[dict[str, Any]]) -> UploadReply:
        url = self.url("session", session_id, "identifications")
        try:
            response = self._send("POST", url, {"identifications": items})
        except requests.RequestException as exc:
            return UploadReply(Outcome.RETRY, f"connection error: {exc}")
        if response.ok:
            try:
                body = response.json()
            except ValueError:
                body = {}
            body = body if isinstance(body, dict) else {}
            created, duplicates = body.get("created"), body.get("duplicates")
            return UploadReply(
                Outcome.STORED,
                created=created if isinstance(created, int) else len(items),
                duplicates=duplicates if isinstance(duplicates, int) else 0,
            )
        code = response.status_code
        if code == 429 or code >= 500:
            return UploadReply(Outcome.RETRY, _detail(response))
        if code in _STOP_STATUSES:
            return UploadReply(Outcome.STOPPED, _detail(response))
        return UploadReply(Outcome.REJECTED, _detail(response))

    def close(self, session_id: str) -> str | None:
        """Close the session; None on success, else why it could not be."""
        try:
            response = self._send("POST", self.url("session", session_id, "close"))
        except requests.RequestException as exc:
            return f"connection error: {exc}"
        return None if response.ok else _detail(response)


def open_session(
    api: SessionApi,
    spec: str,
    *,
    name: str = "",
    game: str = "",
    platform: str = DEFAULT_PLATFORM,
    client: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Start (``spec`` = NEW) or resume (``spec`` = a session id) a session.

    Returns ``(session, created)``. Only a LIVE session can be resumed — a
    closed one refuses uploads, which is better said at startup than
    discovered after the first batch.
    """
    spec = parse_session_spec(spec)
    if spec == NEW_SESSION:
        payload = {
            "name": name[:_SESSION_NAME_LIMIT],
            "game": game[:_GAME_LIMIT],
            "platform": platform,
            "client": client or {},
        }
        return api.create(payload), True
    session = api.get(spec)
    if session.get("status") != "live":
        raise SessionError(
            f"session {spec} is {session.get('status', 'not live')} — "
            f"start a new one with --ximilar-stream {NEW_SESSION}"
        )
    return session, False


class SessionRecorder:
    """Queues identifications and uploads them in the background.

    ``record`` never blocks on the network and never raises, so the identify
    thread that calls it cannot be held up by the API. ``close`` drains the
    queue (retrying transient failures until ``timeout``) and then closes the
    session unless ``close_session`` is off — the resume-after-restart case.
    """

    def __init__(
        self,
        api: SessionApi,
        session_id: str,
        *,
        close_session: bool = True,
        flush_seconds: float = DEFAULT_FLUSH_SECONDS,
        max_pending: int = MAX_PENDING,
        log: Callable[[str], None] = print,
        start: bool = True,
    ) -> None:
        self.session_id = session_id
        self._api = api
        self._close_session = close_session
        self._flush_seconds = flush_seconds
        self._max_pending = max_pending
        self._log = log
        self._pending: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()  # guards _pending, counters and flags
        self._flushing = threading.Lock()  # one batch in flight at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._stopped: str | None = None  # why uploads ended, once they have
        self._overflowing = False
        self.recorded = 0  # identifications queued
        self.stored = 0  # accepted by the API (created + already known)
        self.dropped = 0  # given up on: rejected, overflowed or stopped
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(
                target=self._run, name="cardstream-session", daemon=True
            )
            self._thread.start()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def stopped(self) -> str | None:
        """Why uploads stopped for good, or None while they still run."""
        return self._stopped

    def record(
        self, ident: dict[str, Any], id_type: str, seen: float | None = None
    ) -> None:
        """Queue one identification; ``seen`` defaults to now (epoch seconds)."""
        item = session_item(ident, id_type, time.time() if seen is None else seen)
        with self._lock:
            if self._closed or self._stopped is not None:
                return
            if len(self._pending) >= self._max_pending:
                self._pending.popleft()
                self.dropped += 1
                if not self._overflowing:
                    self._overflowing = True
                    self._log(
                        f"[session] upload queue full ({self._max_pending}) — "
                        "dropping the oldest identifications until the API "
                        "is reachable again"
                    )
            self._pending.append(item)
            self.recorded += 1
            full = len(self._pending) >= MAX_BATCH
        if full:
            self._wake.set()

    def flush(self) -> bool:
        """Upload everything queued, batch by batch.

        False means a transient failure left items queued (try again later);
        True means the queue is empty or will never drain (uploads stopped).
        """
        with self._flushing:
            while True:
                with self._lock:
                    if self._stopped is not None or not self._pending:
                        return True
                    batch = [
                        self._pending[i]
                        for i in range(min(MAX_BATCH, len(self._pending)))
                    ]
                reply = self._api.upload(self.session_id, batch)
                if reply.outcome is Outcome.RETRY:
                    self._log(
                        f"[session] upload failed ({reply.detail}) — keeping "
                        f"{self.pending} identification(s), retrying"
                    )
                    return False
                self._settle(batch, reply)

    def _settle(self, batch: list[dict[str, Any]], reply: UploadReply) -> None:
        """Take a finished batch off the queue and account for it."""
        sent = {item["event_id"] for item in batch}
        with self._lock:
            # By id, not by position: record() may have dropped the oldest
            # items from the head while this batch was on the wire.
            self._pending = deque(i for i in self._pending if i["event_id"] not in sent)
            if reply.outcome is Outcome.STORED:
                self.stored += len(batch)
                self._overflowing = False
                return
            self.dropped += len(batch)
            if reply.outcome is Outcome.STOPPED:
                self._stopped = reply.detail
                self.dropped += len(self._pending)
                self._pending.clear()
        if reply.outcome is Outcome.STOPPED:
            self._log(
                f"[session] uploads stopped ({reply.detail}) — the show goes on, "
                "but nothing more is saved to the session"
            )
        else:
            self._log(
                f"[session] upload rejected ({reply.detail}) — dropped "
                f"{len(batch)} identification(s)"
            )

    def _run(self) -> None:
        delay = self._flush_seconds
        while not self._stop.is_set():
            self._wake.wait(delay)
            self._wake.clear()
            if self._stop.is_set():
                return
            ok = self.flush()
            delay = self._flush_seconds if ok else min(delay * 2, MAX_BACKOFF_SECONDS)

    def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        """Drain the queue, then close the session (unless told to keep it).

        Idempotent. Identifications recorded after this are ignored — an
        identify call still in flight when the show ends is not waited for.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        deadline = time.monotonic() + timeout
        while not self.flush():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))

        if self.pending:
            self._log(
                f"[session] {self.pending} identification(s) could not be uploaded"
            )
        summary = (
            f"[session] {self.stored} identification(s) saved to {self.session_id}"
        )
        if self._stopped is not None:
            self._log(summary)
        elif self._close_session:
            failure = self._api.close(self.session_id)
            if failure is None:
                self._log(f"{summary} — session closed")
            else:
                self._log(f"{summary} — could not close it ({failure})")
        else:
            self._log(
                f"{summary} — left open; resume it with --ximilar-stream "
                f"{self.session_id}"
            )
