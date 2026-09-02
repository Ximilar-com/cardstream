"""The per-card decision engine.

"One paid call per distinct card, gated by cheap local signals" lives here and
only here: the *decisions* are pure, synchronous, I/O-free and fully
unit-testable, while the driver
(:class:`cardstream.client.analyzer.SmartAnalyzer`) owns the scheduling —
detection inline, identification on a thread. Keeping the two apart is what
lets the whole state machine be tested without a camera, a network or a clock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from cardstream.core.models import BoundingBox, CardState, DetectionResult
from cardstream.core.motion import hamming, phash
from cardstream.core.quad import map_quad


@dataclass(frozen=True)
class DetectIntervals:
    """Detection throttle tiers. Three are picked by the free motion gate:
    moving scene -> fast; static with a card -> medium (keep tracking);
    static and empty -> slow heartbeat (a card appearing creates motion,
    which bumps us back to fast). The fourth, ``tracking``, wins over all of
    them while a visual tracker is locked on the card — detection then only
    re-syncs the tracker (drift correction / card-still-there check)."""

    moving: float
    idle_with_card: float
    empty: float
    tracking: float = 2.0


# Shared defaults. Both drivers configure these independently (env on the
# server, flags on the client) but had no reason to disagree on the starting
# value, and each was typing its own literal.
DEFAULT_PHASH_DISTANCE = 10  # hamming above this = a different card
DEFAULT_CALL_TIMEOUT_SECONDS = 20.0  # watchdog: force-clear a hung call


class CallGuard:
    """One throttled, in-flight-guarded call slot (detect or identify).

    Tracks the in-flight flag and the last-start timestamp (which doubles as
    the throttle/cooldown anchor), and provides the watchdog that force-clears
    a stuck flag so a hung upstream call can never freeze the pipeline —
    the call itself is not cancelled, only forgotten.
    """

    def __init__(
        self,
        name: str,
        timeout_seconds: float,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self._timeout = timeout_seconds
        self._log = log
        self.in_flight = False
        # -inf, not 0.0: time.monotonic() counts from boot on Linux, so on a
        # young machine 0.0 would throttle the FIRST call as if one had just
        # fired. -inf is the same "never called" sentinel expire() uses.
        self.last_start_ts = float("-inf")

    def watchdog(self, now: float) -> None:
        if self.in_flight and (now - self.last_start_ts) > self._timeout:
            if self._log is not None:
                self._log(f"{self.name} watchdog reset (call exceeded timeout)")
            self.in_flight = False

    def ready(self, now: float, interval: float) -> bool:
        return not self.in_flight and (now - self.last_start_ts) >= interval

    def begin(self, now: float) -> None:
        self.in_flight = True
        self.last_start_ts = now

    def end(self) -> None:
        self.in_flight = False

    def expire(self) -> None:
        """Void the throttle anchor so the next ``ready()`` passes regardless
        of interval (an in-flight call still blocks)."""
        self.last_start_ts = float("-inf")


class IdentityGate(ABC):
    """Decides whether a settled card is the SAME one (skip the paid call) or
    NEW. ``decide`` computes the signature and compares; ``commit`` stores it.

    Commit-on-fire encodes the no-retry policy: the signature is remembered
    the moment an identify is *launched*, so a card the upstream can't identify
    is not hammered — it retries naturally once the card changes.
    """

    @abstractmethod
    def decide(self, crop_bgr: np.ndarray) -> tuple[bool, Any]:
        """Return (is_new_card, signature_token)."""

    @abstractmethod
    def commit(self, token: Any) -> None:
        """Store ``token`` as the signature of the card being identified."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the stored signature — the next card counts as NEW."""


class PhashGate(IdentityGate):
    """Perceptual-hash identity gate (free, no ML deps)."""

    def __init__(
        self,
        distance_threshold: int = DEFAULT_PHASH_DISTANCE,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._threshold = distance_threshold
        self._on_debug = on_debug
        self._last: np.uint64 | None = None

    def decide(self, crop_bgr: np.ndarray) -> tuple[bool, Any]:
        h = phash(crop_bgr)
        if self._last is None:
            return True, h
        dist = hamming(h, self._last)
        if self._on_debug is not None:
            self._on_debug(f"[gate] phash_dist={dist} (threshold {self._threshold})")
        return dist > self._threshold, h

    def commit(self, token: Any) -> None:
        self._last = token

    def reset(self) -> None:
        self._last = None


@dataclass(frozen=True)
class Snapshot:
    """The core's current belief about the card in frame.

    ``quad`` is the card's corners when a segmentor found them — carried along
    by a tracker update, cleared when the card goes — and None from every box
    locator. Note that ``==`` between two snapshots that both carry corners is
    numpy-elementwise and will not reduce to a bool; nothing here does that,
    and the equality is here for the all-None empty case.
    """

    state: CardState
    bbox: BoundingBox | None
    # IdentificationLike: Identification (parsed) or the client's dict.
    identification: Any | None
    quad: np.ndarray | None


class DecisionCore:
    """Synchronous per-card decision state — the whole call policy, in one place.

    :class:`~cardstream.client.analyzer.SmartAnalyzer` feeds it these events
    and reads back what to do:

    * ``on_frame(settled, score, now)`` -> True = launch a detection now
    * ``on_detection(det, settled, now)`` -> True = launch an identify for it
    * ``on_track(ok, bbox)`` -> per-frame tracker outcome (tracker drivers)
    * ``on_identify_done(ident)`` -> records the result (None = failed)

    plus ``snapshot()`` for the current (state, bbox, identification).
    """

    def __init__(
        self,
        intervals: DetectIntervals,
        motion_threshold: float,
        gate: IdentityGate,
        cooldown_seconds: float,
        call_timeout_seconds: float,
        log: Callable[[str], None] | None = None,
        use_tracker: bool = False,
        forget_after_seconds: float = 0.0,
        retry_unmatched_seconds: float = 0.0,
    ) -> None:
        self._intervals = intervals
        self._motion_threshold = motion_threshold
        self._gate = gate
        self._cooldown = cooldown_seconds
        self._use_tracker = use_tracker
        # A card gone this long stops counting as "the same card": the gate
        # signature is dropped so the next one is identified from scratch.
        # 0 disables it — the signature then survives any gap.
        self._forget_after = forget_after_seconds
        self._absent_since: float | None = None
        # The no-retry policy, softened: a card whose call came back with
        # nothing is normally never asked about again while it stays put (the
        # gate committed its signature on fire). After this many seconds the
        # signature is dropped so the SAME card gets one more attempt.
        # 0 keeps the strict policy.
        self._retry_unmatched = retry_unmatched_seconds
        self._unmatched = False
        # One channel: the driver hands its ``on_log`` in and the terminal and
        # the browser's debug panel both see every line, faults and routine
        # events alike.
        self._log = log
        self.detect = CallGuard("detection", call_timeout_seconds, log)
        self.identify = CallGuard("identify", call_timeout_seconds, log)

        self.present = False
        self.settled = False
        self.tracking = False  # a visual tracker is locked on the card
        self.last_bbox: BoundingBox | None = None
        # The card's four corners when the locator is a SEGMENTOR, else None —
        # the overlay draws the real outline instead of the hull. A tracker
        # update MOVES them with its box (see on_track); only losing the card
        # clears them.
        self.last_quad: np.ndarray | None = None
        # IdentificationLike: Identification (server) or dict (client).
        self.last_ident: Any | None = None

    def on_frame(self, settled: bool, score: float, now: float) -> bool:
        """Run the watchdogs and the tiered throttle; True = detect now."""
        self.settled = settled
        self.detect.watchdog(now)
        self.identify.watchdog(now)

        if self.tracking:
            # The tracker follows the card (even a moving one); detection only
            # needs to re-sync it. A lost track expires the throttle, so the
            # fallback detect fires immediately, not a full interval later.
            interval = self._intervals.tracking
        elif score >= self._motion_threshold:
            interval = self._intervals.moving
        elif self.present:
            interval = self._intervals.idle_with_card
        else:
            interval = self._intervals.empty
        if self.detect.ready(now, interval):
            self.detect.begin(now)
            return True
        return False

    def on_track(self, ok: bool, bbox: BoundingBox | None) -> None:
        """Record a per-frame tracker update while ``tracking`` is set."""
        if ok and bbox is not None:
            # Carry the corners along with the box rather than dropping them.
            # Dropping was the safe-looking choice, but with a segmentor the
            # tracking re-sync interval is 2s: at 30 fps the real outline
            # showed on one frame in sixty and the overlay flickered
            # polygon<->rect. map_quad is as approximate as the box, and the
            # box is exactly what the page falls back to drawing — and a fresh
            # detection re-syncs both. Nothing PAID is cut from this: an
            # identify only ever fires straight after a detection.
            if self.last_quad is not None and self.last_bbox is not None:
                self.last_quad = map_quad(self.last_quad, self.last_bbox, bbox)
            self.last_bbox = bbox
        elif self.tracking:
            self.on_tracker_failed()

    def on_tracker_failed(self) -> None:
        """Tracker lost the card (or broke): fall back to the motion tiers and
        let the next ``on_frame`` fire a detection immediately."""
        self.tracking = False
        self.detect.expire()

    def abort_detection(self) -> None:
        """A launched detection died (detector raised): clear the in-flight
        flag so the throttle is not jammed."""
        self.detect.end()

    def on_detection(
        self, det: DetectionResult | None, settled: bool, now: float
    ) -> bool:
        """Record a detection outcome; True = launch an identify for ``det``.

        ``settled`` is the motion-gate verdict captured when the detection was
        *scheduled* — for an async detector the scene may have changed since.
        """
        self.detect.end()
        if det is None:
            # Card gone. Keep the gate signature + last_ident so the SAME card
            # returning is shown instantly without a fresh paid call — unless
            # it stays away longer than ``forget_after_seconds`` (below).
            if self.present or self._absent_since is None:
                self._absent_since = now
            self.present = False
            self.tracking = False
            self.last_bbox = None
            self.last_quad = None
            return False

        if self._absent_since is not None:
            gone = now - self._absent_since
            self._absent_since = None
            if self._forget_after and gone >= self._forget_after:
                # Long gap: probably a different card, or the same one worth
                # re-checking. Start clean rather than trusting a stale
                # signature that would veto the call.
                self._gate.reset()
                self.last_ident = None
                self._unmatched = False
                if self._log is not None:
                    self._log(
                        f"[gate] card away {gone:.1f}s (> {self._forget_after:.0f}s) "
                        "— forgetting the last card, analysing fresh"
                    )

        self.present = True
        # The detector box is ground truth: the driver re-inits its tracker
        # from it, which corrects any drift accumulated since the last sync.
        self.tracking = self._use_tracker
        self.last_bbox = det.bbox
        self.last_quad = det.quad
        if not settled:
            return False
        if not self.identify.ready(now, self._cooldown):
            return False

        if (
            self._retry_unmatched
            and self._unmatched
            and (now - self.identify.last_start_ts) >= self._retry_unmatched
        ):
            # Forget the unmatched card so the gate below calls it NEW. Cleared
            # first: a second failure re-arms it from the new call's timestamp,
            # which is what spaces repeats out instead of stacking them.
            self._unmatched = False
            self._gate.reset()
            if self._log is not None:
                self._log(
                    f"[gate] no match {self._retry_unmatched:.0f}s ago — "
                    "trying this card once more"
                )

        changed, token = self._gate.decide(det.crop)
        if not changed:
            # The same card, still in frame: it is already paid for.
            return False

        self.identify.begin(now)
        self._gate.commit(token)
        if changed:
            # Different card -> drop the stale name so the overlay doesn't show
            # the previous card while we re-identify. (Same card -> keep it.)
            self.last_ident = None
        return True

    def on_identify_done(self, ident: Any | None) -> None:
        """Record an identify result. ``None`` (failed / no match) keeps the
        committed signature, so the same card is not retried."""
        self.identify.end()
        # Arms --retry-unmatched. A failed call and a match dropped by the
        # result threshold are the same thing here: the card on the table has
        # no name, and asking again later is the only way it gets one.
        self._unmatched = ident is None
        if ident is not None:
            self.last_ident = ident

    def snapshot(self) -> Snapshot:
        """What the core currently believes, for the driver to render.

        A named shape rather than a tuple: it has already grown once (``quad``)
        and every caller destructured it positionally.
        """
        if not self.present:
            return Snapshot(CardState.EMPTY, None, None, None)
        if self.identify.in_flight:
            state = CardState.IDENTIFYING
        elif self.last_ident is not None:
            state = CardState.IDENTIFIED
        elif self.settled:
            state = CardState.SETTLED
        else:
            state = CardState.MOVING
        return Snapshot(state, self.last_bbox, self.last_ident, self.last_quad)
