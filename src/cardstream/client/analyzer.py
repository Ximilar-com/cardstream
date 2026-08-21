"""SmartAnalyzer — the client-side driver over the shared DecisionCore.

Motion gate, card detection and the same-card identity gate all run locally
(free); the only network call is one ``/identify`` per distinct card. The
state machine (three-tier detect throttle, identity-gate policy, cooldown /
watchdog / no-retry) lives in
:mod:`cardstream.core.engine`; this class owns the client's scheduling:
detection is inline-synchronous, the identify request runs in a small
background thread so the capture loop never blocks.

Card removed: the gate signature + identification are kept, so the SAME card
returning is shown instantly without another paid call.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cardstream.client.embedders import Embedder, EmbeddingGate
from cardstream.client.identify_target import IdentifyTarget
from cardstream.core.detection_filters import make_detection_filters
from cardstream.core.detectors import CardDetector
from cardstream.core.engine import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_PHASH_DISTANCE,
    DecisionCore,
    DetectIntervals,
    PhashGate,
)
from cardstream.core.image_store import ImageStore
from cardstream.core.imaging import FramePair
from cardstream.core.models import AnalysisResult, BoundingBox, DetectionResult
from cardstream.core.motion import MotionGate
from cardstream.core.quad import expand_quad, paid_quad
from cardstream.core.tracking import ObjectTracker, make_tracker

# The analysis downscale, named so the web layer's own signatures can default
# to it instead of repeating the number in three more places.
DEFAULT_ANALYSIS_WIDTH = 960


@dataclass(frozen=True)
class AnalyzerConfig:
    """The tuning knobs, and the single source of their defaults — argparse in
    client/common.py reads every one of them off this dataclass."""

    gate: str = "embedding"  # "embedding" | "phash"
    similarity_threshold: float = 0.85  # cosine sim below this = new card
    phash_threshold: int = DEFAULT_PHASH_DISTANCE
    result_threshold: float = (
        0.9  # drop matches with distance above this (1.0 = keep all)
    )
    analysis_width: int = (
        DEFAULT_ANALYSIS_WIDTH  # local analysis downscale; 0 = full res
    )
    # A detection narrower than this FRACTION of the analysed frame — in either
    # dimension — is ignored, as if nothing had been detected. 0 = accept any
    # box. A fraction rather than pixels so it means the same thing whatever
    # --width and the camera resolution are.
    min_card_fraction: float = 0.1
    # Shortest side over longest, so it reads the same whether the card is held
    # portrait or landscape. A card is ~0.71; a fragment (a corner clipped by
    # the frame edge, a sleeve lip) is well under. 0 = accept any shape.
    min_card_aspect: float = 0.4
    # Grow the located card by this fraction on every side before cutting the
    # crop that is PAID FOR — nothing else sees it (not the gate, not the
    # overlay, not the filters). 0 = send exactly what was located.
    detection_expansion: float = 0.0
    cooldown_seconds: float = 2.0
    forget_after_seconds: float = 2.0  # card away this long = analyse fresh (0 = never)
    # A card whose identify came back with nothing is asked about again after
    # this long, rather than being left alone until it changes. Short on
    # purpose: an unnamed card in frame is a card the show still needs a name
    # for, and the miss is usually a bad look (glare, a hand across the art)
    # that the next frame already fixes. 0 = never retry.
    retry_unmatched_seconds: float = 0.5
    motion_threshold: float = 8.0
    still_frames_required: int = 2
    detect_interval_seconds: float = 0.1
    idle_detect_interval_seconds: float = 0.2
    empty_detect_interval_seconds: float = 0.2
    tracker_model: str | None = None  # path to vitTracker .onnx; set = enable tracking
    tracker_score_threshold: float = 0.3  # below this the track counts as lost
    tracking_detect_interval_seconds: float = 2.0  # re-sync detect while tracking
    # No flag on purpose: the watchdog only force-clears a call that hung past
    # this, which is a fault path, not a tuning knob. Settable here so a test
    # (or an embedder of this package) can shorten it.
    call_timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS
    debug: bool = False


# AnalyzerConfig fields the settings dialog may change on a RUNNING analyzer.
# Everything else is read once at construction, so retuning it would be a lie.
LIVE_FIELDS = frozenset({"result_threshold"})


class SmartAnalyzer:
    def __init__(
        self,
        detector: CardDetector,
        embedder: Embedder | None,
        identify_client: IdentifyTarget,
        config: AnalyzerConfig | None = None,
        on_result: Callable[[dict], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        run_async: bool = True,
        tracker: ObjectTracker | None = None,
        store: ImageStore | None = None,
    ) -> None:
        cfg = config or AnalyzerConfig()
        if cfg.gate == "embedding" and embedder is None:
            raise ValueError("gate='embedding' requires an embedder")
        self._detector = detector
        # Trackers are stateful per card, so each analyzer owns its own
        # instance (web_client builds one analyzer per connection). The
        # parameter is an override for tests.
        self._tracker = (
            tracker
            if tracker is not None
            else make_tracker(cfg.tracker_model, cfg.tracker_score_threshold)
        )
        self._identify_client = identify_client
        # --store-images in `frame` mode: the identify path only ever sees the
        # crop, so the whole picture has to be kept from here. In `object` mode
        # save_frame is a no-op and the identify client does the keeping.
        self._store = store
        self._cfg = cfg
        # Paid identify calls this analyzer has fired — the point of the whole
        # state machine, so the UI shows it. Counted on fire, not on success:
        # a call that returns no match was still spent.
        self.identify_calls = 0
        # (w, h) of the frame the last detection ran on — the space bboxes are
        # in, so the browser needs it to draw them. Dimensions only: holding
        # the frame itself would pin a full-resolution copy per connection.
        self.analysis_size: tuple[int, int] | None = None
        self._on_result = on_result
        # Log sink: the CLI leaves it None (plain stdout); the browser UI
        # injects a callback that also pushes lines to the page.
        self._on_log = on_log
        # run_async=False runs /identify inline — for tests and determinism.
        self._run_async = run_async

        # Built once, like the gate's thresholds: baked into the objects, not
        # re-read per frame (and LIVE_FIELDS says as much).
        self._filters = make_detection_filters(
            cfg.min_card_fraction, cfg.min_card_aspect
        )
        self._motion = MotionGate(cfg.motion_threshold, cfg.still_frames_required)
        self._prev_settled: bool | None = None  # for motion-transition debug logs
        # The detection whose crop the gate is currently deciding on — lets the
        # gate debug line carry the detector's confidence too.
        self._gate_det = None
        on_debug = self._gate_debug if cfg.debug else None
        gate = (
            EmbeddingGate(embedder, cfg.similarity_threshold, on_debug=on_debug)
            if cfg.gate == "embedding"
            else PhashGate(cfg.phash_threshold, on_debug=on_debug)
        )
        self._core = DecisionCore(
            intervals=DetectIntervals(
                moving=cfg.detect_interval_seconds,
                idle_with_card=cfg.idle_detect_interval_seconds,
                empty=cfg.empty_detect_interval_seconds,
                tracking=cfg.tracking_detect_interval_seconds,
            ),
            motion_threshold=cfg.motion_threshold,
            gate=gate,
            cooldown_seconds=cfg.cooldown_seconds,
            call_timeout_seconds=cfg.call_timeout_seconds,
            log=self._log,
            use_tracker=self._tracker is not None,
            forget_after_seconds=cfg.forget_after_seconds,
            retry_unmatched_seconds=cfg.retry_unmatched_seconds,
        )

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            self._on_log(msg)
        else:
            print(msg)

    def _gate_debug(self, msg: str) -> None:
        """Gate debug lines, augmented with the detector's confidence."""
        det = self._gate_det
        if det is not None and det.prob is not None:
            msg = f"{msg} det_prob={det.prob:.2f}"
        self._log(msg)

    @property
    def result_threshold(self) -> float:
        """Read-only — retune via :meth:`tune` so the config stays the truth."""
        return self._cfg.result_threshold

    def tune(self, **fields: object) -> None:
        """Swap live-tunable config values on a running analyzer.

        The config stays frozen — it IS the startup default — so this replaces
        it wholesale rather than shadowing individual fields with instance
        attributes. LIVE_FIELDS is also the honest list of what CANNOT be
        retuned: gate thresholds are baked into the gate object and detect
        intervals into DecisionCore, so changing them here would silently do
        nothing.
        """
        if unknown := set(fields) - LIVE_FIELDS:
            raise ValueError(f"not live-tunable: {', '.join(sorted(unknown))}")
        self._cfg = dataclasses.replace(self._cfg, **fields)

    def process(self, frame_bgr: np.ndarray) -> AnalysisResult:
        # Everything local runs on the downscaled frame; only the crop handed to
        # the paid id endpoint is cut from the original (pair.crop below).
        pair = FramePair.from_frame(frame_bgr, self._cfg.analysis_width)
        frame_bgr = pair.analysis
        if self._cfg.debug and pair.size != self.analysis_size:
            fw, fh = pair.full_size
            aw, ah = pair.size
            self._log(
                f"[frame] {fw}x{fh} -> analysis {aw}x{ah} (scale {1 / pair.scale:.2f})"
            )
        self.analysis_size = pair.size

        settled, score = self._motion.update(frame_bgr)
        now = time.monotonic()

        # Log motion-gate TRANSITIONS only — per-frame lines would flood at 30 fps.
        if self._cfg.debug and settled != self._prev_settled:
            state = "settled" if settled else "moving"
            self._log(
                f"[motion] {state} (score={score:.1f}, threshold "
                f"{self._cfg.motion_threshold})"
            )
        self._prev_settled = settled

        if self._tracker is not None and self._core.tracking:
            ok, bbox = self._tracker.update(frame_bgr)
            self._core.on_track(ok, bbox)
            if self._cfg.debug and not ok:
                self._log("[track] lost — forcing re-detect")

        if self._core.on_frame(settled, score, now):
            self._detect_tick(pair, settled, now)

        return self._snapshot()

    def _detect_tick(self, pair: FramePair, settled: bool, now: float) -> None:
        """One detection: run it, judge it, and pay for it if the core says so."""
        was_present = self._core.present
        try:
            det = self._detector.detect(pair.analysis)
        except BaseException:
            self._core.abort_detection()  # a raising detector must not jam the throttle
            raise

        if det is not None and (reason := self._rejected(det.bbox, pair.size)):
            # Dropped to None rather than flagged: a sliver of card is not a
            # card, and the rest of the pipeline already knows what "nothing
            # detected" means.
            det = None
            if self._cfg.debug:
                self._log(reason)

        self._gate_det = det
        fire = self._core.on_detection(det, settled, now)
        if det is not None:
            self._init_tracker(pair.analysis, det)
        if self._cfg.debug:
            if det is None and was_present:
                self._log("[detect] card lost")
            elif det is not None and not was_present:
                conf = f" det_prob={det.prob:.2f}" if det.prob is not None else ""
                self._log(f"[detect] card found bbox={det.bbox.as_list()}{conf}")
        if fire:
            self._identify_detection(pair, det)

    def _rejected(self, bbox: BoundingBox, frame_size: tuple[int, int]) -> str | None:
        """The first rule that says this box is not a card, or None."""
        for rule in self._filters:
            if reason := rule.reject(bbox, frame_size):
                return reason
        return None

    def _identify_detection(self, pair: FramePair, det: DetectionResult) -> None:
        """Send the crop for this detection — the one step that costs money."""
        # Cut from the ORIGINAL frame: the gate already had its (cheap,
        # analysis-space) look via det.crop, and both pair.crop and pair.warp
        # return an owned array, so the identify thread can outlive this frame.
        # A locator that found the card's CORNERS (a segmentor) gets a deskewed
        # crop, tight at the card edge; a box locator gets the square cut.
        crop = self._identify_crop(pair, det)
        if self._store is not None:
            # Inline, not on the identify thread: pair.full belongs to the
            # caller and may be reused for the next frame, while pair.crop()
            # hands the thread an owned copy.
            self._store.save_frame(pair.full)
        if crop is None:
            self._log("[identify] degenerate crop — skipping")
            self._core.on_identify_done(None)
            return
        if self._cfg.debug:
            self._log(self._identify_log(crop, deskewed=det.quad is not None))
        if self._run_async:
            threading.Thread(target=self._identify, args=(crop,), daemon=True).start()
        else:
            self._identify(crop)

    def _identify_crop(
        self, pair: FramePair, det: DetectionResult
    ) -> np.ndarray | None:
        """The pixels that go on the wire, cut from the ORIGINAL frame.

        --detection-expansion is applied HERE and nowhere else, so it changes
        what is IDENTIFIED without disturbing what was LOCATED: the identity
        gate keeps comparing the tight crop (a padded one would dilute the
        SAME-vs-NEW decision with background) and the overlay keeps drawing the
        box the model actually returned.

        Deskew or square cut is the only branch — the growth itself follows
        ``core.quad.paid_quad``, which is what _crop_outline draws.
        """
        grow = self._cfg.detection_expansion
        if det.quad is not None:
            return pair.warp(expand_quad(det.quad, grow))
        return pair.crop(det.bbox.expanded(grow))

    def _identify_log(self, crop_bgr: np.ndarray, deskewed: bool = False) -> str:
        """The debug line that proves which resolution and shape was actually sent."""
        h, w = crop_bgr.shape[:2]
        how = " deskewed" if deskewed else ""
        if self._cfg.detection_expansion:
            how += f" +{self._cfg.detection_expansion:.0%}"
        return f"[identify] new card — calling identify crop={w}x{h}{how}"

    def _init_tracker(self, frame_bgr: np.ndarray, det: DetectionResult) -> None:
        """Re-seed the tracker from a fresh detection (detector = ground truth)."""
        if self._tracker is None:
            return
        try:
            self._tracker.init(frame_bgr, det.bbox)
            if self._cfg.debug:
                self._log(f"[track] init bbox={det.bbox.as_list()}")
        except Exception as exc:
            # A broken tracker must not kill the capture loop — fall back to
            # plain motion-tiered detection for this card.
            self._core.on_tracker_failed()
            self._log(f"[track] init failed ({type(exc).__name__}: {exc})")

    def _identify(self, crop_bgr: np.ndarray) -> None:
        ident = None
        self.identify_calls += 1
        started = time.monotonic()
        try:
            ident = self._identify_client.identify(crop_bgr)
            if ident is not None:
                dist = ident.get("distance")
                dist = float(dist) if isinstance(dist, (int, float)) else 1.0
                if dist > self._cfg.result_threshold:
                    self._log(
                        f"[identify] dist {dist:.3f} > result threshold "
                        f"{self._cfg.result_threshold} — dropping match"
                    )
                    ident = None
        except Exception as exc:
            # Surface client failures — a raising client must not kill the
            # daemon thread silently and leave the state reverting unexplained.
            self._log(f"[identify] {type(exc).__name__}: {exc}")
        finally:
            self._core.on_identify_done(ident)
        if ident is None:
            # The gate signature was committed on fire: don't hammer a card the
            # upstream can't identify — it retries when the card changes.
            self._log("[identify] no match / failed; not retrying this card")
            return
        # Wall time of the whole identify call (network included) — the UI
        # shows it next to the distance.
        ident["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        if self._on_result is not None:
            self._on_result(ident)

    def _snapshot(self) -> AnalysisResult:
        snap = self._core.snapshot()
        return AnalysisResult(
            state=snap.state,
            bbox=snap.bbox,
            identification=snap.identification,
            quad=snap.quad,
            crop_quad=self._crop_outline(snap.bbox, snap.quad),
        )

    def _crop_outline(
        self, bbox: BoundingBox | None, quad: np.ndarray | None
    ) -> np.ndarray | None:
        """Where the paid crop WILL be cut, when that differs from what was
        located — i.e. only under --detection-expansion.

        Always four corners, so the page draws one kind of shape whichever
        locator is running. ``core.quad.paid_quad`` is the shared statement of
        how the expansion applies, so this cannot promise a region
        _identify_crop would not cut.
        """
        grow = self._cfg.detection_expansion
        return paid_quad(bbox, quad, grow) if grow else None
