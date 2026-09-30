"""The show's history as the page lists it: the rows --ximilar-stream saves.

A Python twin of the history list in ``webui/shared/overlay.js``
(``handleResult``, ``_addHistory``, ``_closeEntry``, ``_resumeEntry``,
``_revealIfEarned``), fed with the same snapshots the page renders, so a
session holds exactly the rows the page shows:

* **One row per card.** Consecutive identifications of the same card (full
  name, name, set, number) are one row. A later paid call for the card the row
  already shows only raises the row's call count.
* **Time on stream.** It counts from when the card appeared, not from when the
  identification landed, and stops while the frame is empty. In merge mode (the
  default) the same card coming back before a different one appears resumes
  the row and its clock. With ``split_results`` every appearance is its own row.
* **Only cards that earned it.** A row is kept only once its card has been on
  stream for ``min_card_time``, so a card glimpsed mid-swap never appears, on
  the page or in the session.

Rows go to ``emit`` when they earn their place and again whenever they change:
the card leaves, comes back or is identified again, and every
``refresh_seconds`` while it stays. They always keep the same ``event_id``, so
the session updates the last row instead of adding one.

Stdlib only; the clocks are injectable so the tests need no sleeping.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# The page's --min-card-time default; the web flag reads it from here.
DEFAULT_MIN_CARD_TIME = 1.0
# How often a row still on stream is re-sent, so a live session's duration
# does not sit at its first value while one card is held for minutes.
DEFAULT_REFRESH_SECONDS = 10.0

_EMPTY = "empty"


def card_key(identification: dict[str, Any]) -> str:
    """The page's history key: two identifications with it are the same card."""
    parts = ("full_name", "name", "set", "card_number")
    return "|".join(str(identification.get(part) or "") for part in parts)


@dataclass
class HistoryRow:
    """One row of the history: a card shown on stream."""

    event_id: str
    # The FIRST identification of the row, which is what the page's row shows.
    identification: dict[str, Any]
    id_type: str
    # Wall-clock time (epoch seconds) the card appeared.
    seen: float
    # Paid calls whose match is this row: 0 for a card shown from the
    # analyzer's memory, which --split-results can turn into a row of its own.
    calls: int = 1
    shown: bool = False
    _accumulated: float = 0.0
    _open_since: float | None = None
    _last_emit: float | None = field(default=None, repr=False)

    def duration(self, now: float) -> float:
        """Seconds on stream so far, the current visit included."""
        running = now - self._open_since if self._open_since is not None else 0.0
        return self._accumulated + running


class ShowHistory:
    """Tracks the rows of one analyzer's stream; see the module docstring.

    ``observe`` is called with every snapshot the analyzer produces and
    ``finish`` when the stream stops. ``emit(row, duration)`` receives each new
    or changed row that has earned its place.
    """

    def __init__(
        self,
        emit: Callable[[HistoryRow, float], None],
        *,
        id_type: Callable[[], str],
        min_card_time: float = DEFAULT_MIN_CARD_TIME,
        split_results: bool = False,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._emit_row = emit
        self._id_type = id_type
        self._min_card_time = min_card_time
        self._split = split_results
        self._refresh = refresh_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._last_key: str | None = None
        self._open: HistoryRow | None = None  # the row of the card in frame
        self._last: HistoryRow | None = None  # the frozen row it may resume
        self._present_since: float | None = None
        # Identification dicts already counted as a paid call. Each call
        # returns a new dict and snapshots keep handing the same one back, so
        # the object is the call. Held, not only their ids, so an id is never
        # reused by a later dict.
        self._counted: dict[int, dict[str, Any]] = {}

    def observe(
        self,
        state: object,
        identification: dict[str, Any] | None,
        now: float | None = None,
    ) -> None:
        """One analysed frame: its state and the identification it shows."""
        with self._lock:
            now = self._clock() if now is None else now
            empty = str(getattr(state, "value", state)) == _EMPTY
            if not empty and self._present_since is None:
                self._present_since = now
            if identification:
                self._add(identification, now)
            if empty:
                # Card lost: freeze its time. With split_results the next
                # appearance opens its own row; otherwise it resumes this one.
                self._close(now)
                if self._split:
                    self._last_key = None
                    self._last = None
                self._present_since = None
            elif self._open is not None:
                self._tick(self._open, now)

    def finish(self, now: float | None = None) -> None:
        """The stream stopped: that loses the card as surely as removing it."""
        with self._lock:
            now = self._clock() if now is None else now
            self._close(now)
            self._last_key = None
            self._last = None
            self._present_since = None

    def _is_new_call(self, identification: dict[str, Any]) -> bool:
        if id(identification) in self._counted:
            return False
        self._counted[id(identification)] = identification
        return True

    def _add(self, identification: dict[str, Any], now: float) -> None:
        new_call = self._is_new_call(identification)
        key = card_key(identification)
        if key == self._last_key:
            row = self._open or self._last
            if row is None:
                return
            if new_call:
                row.calls += 1
            if self._open is None:
                # Merge mode: the card that left is back; its clock resumes.
                row._open_since = now
                self._open = row
                if not self._reveal_if_earned(row, now) and new_call:
                    self._emit(row, now)
            elif new_call and row.shown:
                self._emit(row, now)
            return

        self._last_key = key
        # A card swapped in place (no empty frame between the two) starts its
        # own clock now; one arriving into an empty frame counts from when it
        # appeared, not from when its identification landed.
        if self._open is not None or self._present_since is None:
            started = now
        else:
            started = self._present_since
        self._close(now)
        row = HistoryRow(
            event_id=uuid.uuid4().hex,
            identification=identification,
            id_type=self._id_type(),
            seen=self._wall_clock() - (now - started),
            calls=1 if new_call else 0,
            _open_since=started,
        )
        self._open = row
        self._reveal_if_earned(row, now)

    def _close(self, now: float) -> None:
        """Freeze the open row at the moment its card was lost."""
        row = self._open
        if row is None or row._open_since is None:
            return
        row._accumulated += now - row._open_since
        row._open_since = None
        self._open = None
        self._last = row
        # A card that made the time gets its row even if it left between two
        # frames; one that did not stays unsent.
        if not self._reveal_if_earned(row, now) and row.shown:
            self._emit(row, now)

    def _tick(self, row: HistoryRow, now: float) -> None:
        if self._reveal_if_earned(row, now):
            return
        if (
            row.shown
            and row._last_emit is not None
            and now - row._last_emit >= self._refresh
        ):
            self._emit(row, now)

    def _reveal_if_earned(self, row: HistoryRow, now: float) -> bool:
        """Send a row the first time its card has been on stream long enough."""
        if row.shown or row.duration(now) < self._min_card_time:
            return False
        row.shown = True
        self._emit(row, now)
        return True

    def _emit(self, row: HistoryRow, now: float) -> None:
        row._last_emit = now
        self._emit_row(row, round(row.duration(now), 3))
