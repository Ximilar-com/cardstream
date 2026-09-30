"""Ximilar stream sessions — an optional record of the show on the Ximilar platform.

Off unless ``--ximilar-stream`` is passed: without it the identify call is
still the only thing that leaves this machine. With it, the rows of the show's
history (one per card shown, exactly as the page lists them, see
:mod:`cardstream.core.show_history`) and the number of paid identify calls are
ALSO saved to a session through the session API (``/cardstream/v2/`` on
api.ximilar.com, same API key), so the show can be reviewed afterwards — what
was shown, when, for how long, how sure the match was and what it was worth.

Three pieces, each testable without a network:

* :func:`session_item` — one history row as the upload item the API
  validates: renamed fields, text clipped to the API's limits, links reduced
  to plain text. One malformed field would reject the whole batch, so this is
  where the payload is made safe, once. Images never leave the machine.
* :class:`SessionApi` — the HTTP calls (create, get, upload, report calls,
  close) and the one rule that sorts every reply into what happens next.
* :class:`SessionRecorder` — a queue of rows keyed by their ``event_id``,
  drained in batches by a background thread. A row re-sent with the same id
  is an update on the API's side (its duration and call count only grow), so
  a row that changes is simply queued again and a batch whose reply was lost
  is simply sent again. A network error, 429 or 5xx keeps the rows and backs
  off; a rejected batch (400) is dropped and logged; a closed, forbidden or
  missing session stops the uploads — never the show.
"""

from __future__ import annotations

import enum
import itertools
import math
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from cardstream.core.identify_client import DEFAULT_HTTP_TIMEOUT, auth_headers
from cardstream.core.show_history import DEFAULT_MIN_CARD_TIME, HistoryRow, ShowHistory

DEFAULT_SESSION_URL = "https://api.ximilar.com/cardstream/v2"
# The --ximilar-stream value that starts a session instead of resuming one.
NEW_SESSION = "NEW"
# The API's platform choices; "other" is its default too.
PLATFORMS = ("whatnot", "tiktok", "ebay", "fanatics", "youtube", "twitch", "other")
DEFAULT_PLATFORM = "other"

# Per upload request — the API refuses a larger batch outright.
MAX_BATCH = 500
# Rows kept while the API is unreachable. Beyond this the OLDEST go first: a
# show that outlives a long outage keeps its most recent cards.
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
    ident: dict[str, Any],
    id_type: str,
    seen: float,
    event_id: str | None = None,
    *,
    duration: float | None = None,
    calls: int | None = None,
) -> dict[str, Any]:
    """One history row as an upload item.

    ``ident`` is the row's identification, the flattened dict the identify
    target returns (``Identification.to_dict()`` plus what the analyzer adds);
    ``seen`` is when the card appeared, in epoch seconds; ``duration`` its
    seconds on stream and ``calls`` the paid calls behind the row. The two
    renames — ``set`` → ``set_name``, ``confidence_tier`` → ``confidence`` —
    are the API's names. Only declared fields are copied, so the page's
    thumbnail never leaves the machine, and anything the API would reject is
    dropped or clipped instead, because one bad field fails the whole batch.
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

    if duration is not None and math.isfinite(duration) and duration >= 0:
        item["duration"] = round(float(duration), 3)
    if calls is not None:
        item["calls"] = max(0, int(calls))
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


def _failure(response: requests.Response) -> UploadReply:
    """What a non-2xx reply to an upload or a call report means for it."""
    code = response.status_code
    if code == 429 or code >= 500:
        return UploadReply(Outcome.RETRY, _detail(response))
    if code in _STOP_STATUSES:
        return UploadReply(Outcome.STOPPED, _detail(response))
    return UploadReply(Outcome.REJECTED, _detail(response))


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
        return _failure(response)

    def report_calls(self, session_id: str, run: str, calls: int) -> UploadReply:
        """This run's paid identify calls so far; the API keeps each run's highest."""
        url = self.url("session", session_id, "calls")
        try:
            response = self._send("POST", url, {"run": run, "calls": calls})
        except requests.RequestException as exc:
            return UploadReply(Outcome.RETRY, f"connection error: {exc}")
        return UploadReply(Outcome.STORED) if response.ok else _failure(response)

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
    """Uploads a session's history rows and paid-call count in the background.

    Each analyzer gets its own :class:`ShowHistory` from :meth:`history` (the
    rows of one stream) and counts its paid calls with :meth:`count_call`.
    Neither touches the network nor raises, so the frame loop and the identify
    thread are never held up by the API. Rows are queued by ``event_id``: a row
    that changes before it went out replaces its queued version, and one that
    changes after goes out again, which the API turns into an update.

    ``close`` closes the rows still on stream, drains the queue (retrying
    transient failures until ``timeout``), reports the final call count and then
    closes the session unless ``close_session`` is off — the
    resume-after-restart case.
    """

    def __init__(
        self,
        api: SessionApi,
        session_id: str,
        *,
        close_session: bool = True,
        min_card_time: float = DEFAULT_MIN_CARD_TIME,
        split_results: bool = False,
        flush_seconds: float = DEFAULT_FLUSH_SECONDS,
        max_pending: int = MAX_PENDING,
        log: Callable[[str], None] = print,
        start: bool = True,
    ) -> None:
        self.session_id = session_id
        # This process's run of the session: the API keeps one call count per
        # run, so a resumed session adds this run's calls to the earlier ones.
        self.run_id = uuid.uuid4().hex
        self._api = api
        self._close_session = close_session
        self._min_card_time = min_card_time
        self._split_results = split_results
        self._flush_seconds = flush_seconds
        self._max_pending = max_pending
        self._log = log
        # event_id -> (version, item), oldest first. A newer version of a row
        # replaces the older one in place, keeping the row's position.
        self._pending: dict[str, tuple[int, dict[str, Any]]] = {}
        self._version = 0
        self._histories: weakref.WeakSet[ShowHistory] = weakref.WeakSet()
        self._lock = threading.Lock()  # guards the queue, counters and flags
        self._flushing = threading.Lock()  # one request in flight at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closing = False
        self._closed = False
        self._stopped: str | None = None  # why uploads ended, once they have
        self._overflowing = False
        self._paid_calls = 0
        self._reported_calls = 0
        self._saved: set[str] = set()  # event_ids the API has accepted
        self.dropped = 0  # row versions given up on: rejected, overflowed or stopped
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

    @property
    def rows_saved(self) -> int:
        """Rows (cards shown) the session has accepted at least once."""
        with self._lock:
            return len(self._saved)

    @property
    def paid_calls(self) -> int:
        with self._lock:
            return self._paid_calls

    def history(self, id_type: Callable[[], str]) -> ShowHistory:
        """A history for one analyzer's stream, whose rows this recorder saves."""
        history = ShowHistory(
            self.record_row,
            id_type=id_type,
            min_card_time=self._min_card_time,
            split_results=self._split_results,
        )
        with self._lock:
            self._histories.add(history)
        return history

    def record_row(self, row: HistoryRow, duration: float) -> None:
        """Queue a new or changed history row (the ShowHistory callback)."""
        self.queue(
            session_item(
                row.identification,
                row.id_type,
                row.seen,
                event_id=row.event_id,
                duration=duration,
                calls=row.calls,
            )
        )

    def queue(self, item: dict[str, Any]) -> None:
        """Queue an upload item; a queued item with its event_id is replaced."""
        event_id = item["event_id"]
        with self._lock:
            if self._closed or self._stopped is not None:
                return
            self._version += 1
            if (
                event_id not in self._pending
                and len(self._pending) >= self._max_pending
            ):
                del self._pending[next(iter(self._pending))]
                self.dropped += 1
                if not self._overflowing:
                    self._overflowing = True
                    self._log(
                        f"[session] upload queue full ({self._max_pending}) — "
                        "dropping the oldest rows until the API is reachable again"
                    )
            self._pending[event_id] = (self._version, item)
            full = len(self._pending) >= MAX_BATCH
        if full:
            self._wake.set()

    def count_call(self) -> None:
        """One paid identify call fired, matched or not."""
        with self._lock:
            if not self._closed and self._stopped is None:
                self._paid_calls += 1

    def flush(self) -> bool:
        """Upload every queued row, batch by batch, then the call count.

        False means a transient failure left work queued (try again later);
        True means everything queued went out (rows that changed meanwhile go
        with the next flush) or never will (uploads stopped).
        """
        with self._flushing:
            with self._lock:
                # One pass over what is queued now: a row that keeps changing
                # while its batch is on the wire waits for the next flush
                # instead of holding this one forever.
                rounds = len(self._pending) // MAX_BATCH + 1
            for _ in range(rounds):
                with self._lock:
                    if self._stopped is not None:
                        return True
                    batch = list(itertools.islice(self._pending.items(), MAX_BATCH))
                if not batch:
                    break
                reply = self._api.upload(
                    self.session_id, [item for _, (_, item) in batch]
                )
                if reply.outcome is Outcome.RETRY:
                    self._log(
                        f"[session] upload failed ({reply.detail}) — keeping "
                        f"{self.pending} row(s), retrying"
                    )
                    return False
                self._settle(batch, reply)
            return self._report_calls()

    def _report_calls(self) -> bool:
        with self._lock:
            calls = self._paid_calls
            if calls == self._reported_calls:
                return True
        reply = self._api.report_calls(self.session_id, self.run_id, calls)
        if reply.outcome is Outcome.RETRY:
            self._log(f"[session] reporting calls failed ({reply.detail}) — retrying")
            return False
        with self._lock:
            # Even a rejected report is not retried: the same count would be
            # rejected again, forever.
            self._reported_calls = calls
            if reply.outcome is Outcome.STOPPED:
                self._stopped = reply.detail
        if reply.outcome is not Outcome.STORED:
            self._log(f"[session] paid calls not saved ({reply.detail})")
        return True

    def _settle(
        self, batch: list[tuple[str, tuple[int, dict[str, Any]]]], reply: UploadReply
    ) -> None:
        """Take a finished batch off the queue and account for it."""
        with self._lock:
            for event_id, (version, _) in batch:
                # Only the version that was sent: a row that changed while the
                # batch was on the wire must still go out.
                queued = self._pending.get(event_id)
                if queued is not None and queued[0] == version:
                    del self._pending[event_id]
            if reply.outcome is Outcome.STORED:
                self._saved.update(event_id for event_id, _ in batch)
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
                f"{len(batch)} row(s)"
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
        """Close the open rows, drain the queue, then close the session.

        Idempotent. The session is left live with ``close_session`` off. Rows
        and calls recorded after this are ignored — an identify call still in
        flight when the show ends is not waited for.
        """
        with self._lock:
            if self._closing:
                return
            self._closing = True
            histories = list(self._histories)
        # Rows still on stream end with the show; their final durations are
        # queued before the queue is shut.
        for history in histories:
            history.finish()
        with self._lock:
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
            self._log(f"[session] {self.pending} row(s) could not be uploaded")
        summary = (
            f"[session] {self.rows_saved} card(s) and {self._reported_calls} paid "
            f"call(s) saved to {self.session_id}"
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
