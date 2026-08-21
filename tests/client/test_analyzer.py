"""Gate behaviour of SmartAnalyzer: exactly one /identify per distinct card."""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import (
    FakeTracker,
    make_frame,
    unit_vec,
)
from _helpers import (
    make_smart_analyzer as make_analyzer,
)
from cardstream.core.detectors import CardDetector
from cardstream.core.models import BoundingBox, CardState, DetectionResult
from cardstream.core.quad import quad_bbox


def settle(analyzer, frames=6):
    """Feed identical frames until the motion gate reports settled."""
    snap = None
    for _ in range(frames):
        snap = analyzer.process(make_frame())
    return snap


def test_one_identify_per_settled_card(fake_detector, fake_embedder, fake_identify):
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    snap = settle(analyzer, frames=10)
    assert fake_identify.calls == 1  # many settled frames, one paid call
    assert snap.state == CardState.IDENTIFIED
    assert snap.identification["full_name"] == "Charizard"
    assert snap.bbox.as_list() == [10, 10, 50, 70]
    # The identify call's wall time is stamped onto the result for the UI.
    assert isinstance(snap.identification["elapsed_ms"], int)
    assert snap.identification["elapsed_ms"] >= 0


def test_new_embedding_triggers_second_identify(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    settle(analyzer)
    assert fake_identify.calls == 1

    fake_embedder.embedding = unit_vec(1)  # orthogonal -> cosine 0.0 -> new card
    fake_identify.result = {"full_name": "Pikachu", "set": "Base", "distance": 0.1}
    snap = settle(analyzer)
    assert fake_identify.calls == 2
    assert snap.identification["full_name"] == "Pikachu"


def test_same_card_returning_uses_cache(fake_detector, fake_embedder, fake_identify):
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    settle(analyzer)
    assert fake_identify.calls == 1

    fake_detector.detection = None  # card removed
    snap = settle(analyzer)
    assert snap.state == CardState.EMPTY

    fake_detector.detection = type(fake_detector)().detection  # same card back
    snap = settle(analyzer)
    assert fake_identify.calls == 1  # no new paid call
    assert snap.state == CardState.IDENTIFIED
    assert snap.identification["full_name"] == "Charizard"


def test_result_threshold_drops_distant_match(
    fake_detector, fake_embedder, fake_identify
):
    shown = []
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        on_result=shown.append,
        result_threshold=0.8,
    )
    fake_identify.result = {"full_name": "Fighting Energy", "distance": 0.872}
    snap = settle(analyzer)
    assert fake_identify.calls == 1
    assert shown == []  # never surfaced to the UI
    assert snap.identification is None
    settle(analyzer)
    assert fake_identify.calls == 1  # no-retry policy still applies


def test_identify_calls_counts_every_fired_call(
    fake_detector, fake_embedder, fake_identify
):
    """The badge counts what was SPENT: a dropped match still cost a call."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, result_threshold=0.8
    )
    assert analyzer.identify_calls == 0
    fake_identify.result = {"full_name": "Fighting Energy", "distance": 0.872}
    settle(analyzer)
    assert analyzer.identify_calls == 1  # fired, match dropped
    settle(analyzer)
    assert analyzer.identify_calls == 1  # same card -> no new call


def test_result_threshold_keeps_close_match(
    fake_detector, fake_embedder, fake_identify
):
    shown = []
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        on_result=shown.append,
        result_threshold=0.8,
    )
    fake_identify.result = {"full_name": "Charizard", "distance": 0.15}
    snap = settle(analyzer)
    assert [r["full_name"] for r in shown] == ["Charizard"]
    assert snap.identification["full_name"] == "Charizard"


def test_cooldown_suppresses_rapid_second_call(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, cooldown_seconds=60.0
    )
    settle(analyzer)
    assert fake_identify.calls == 1

    fake_embedder.embedding = unit_vec(1)  # new card, but inside the cooldown
    settle(analyzer)
    assert fake_identify.calls == 1


def test_moving_scene_never_identifies(fake_detector, fake_embedder, fake_identify):
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    for i in range(10):  # alternating frames -> never settles
        analyzer.process(make_frame(fill=(i % 2) * 200))
    assert fake_identify.calls == 0
    assert fake_embedder.calls == 0  # gate never even embeds


def test_failed_identify_not_retried_for_same_card(
    fake_detector, fake_embedder, fake_identify
):
    fake_identify.result = None  # server: no match / upstream failure
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    snap = settle(analyzer, frames=10)
    assert fake_identify.calls == 1  # embedding was stored -> no hammering
    assert snap.identification is None

    fake_embedder.embedding = unit_vec(1)  # a NEW card retries naturally
    fake_identify.result = {"full_name": "Pikachu"}
    settle(analyzer)
    assert fake_identify.calls == 2


def test_on_log_receives_debug_lines(fake_detector, fake_embedder, fake_identify):
    fake_detector.detection.prob = 0.7  # a detector that reports confidence
    logs = []
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, on_log=logs.append, debug=True
    )
    settle(analyzer, frames=8)
    assert any("[motion] moving" in line for line in logs)  # first frame
    assert any("[motion] settled" in line for line in logs)  # after N still frames
    assert any("[detect] card found" in line for line in logs)
    assert any("[identify] new card" in line for line in logs)
    # gate line carries both the gate metric and the detector confidence
    assert any("[gate] cos_sim" in line and "det_prob=0.70" in line for line in logs)

    fake_detector.detection = None
    settle(analyzer)
    assert any("[detect] card lost" in line for line in logs)


def test_phash_gate_without_embedder(fake_detector, fake_identify):
    analyzer = make_analyzer(fake_detector, None, fake_identify, gate="phash")
    settle(analyzer, frames=10)
    assert fake_identify.calls == 1  # same crop -> same hash -> one call


# --- visual tracker -----------------------------------------------------------


def test_tracker_carries_bbox_and_suppresses_detection(
    fake_detector, fake_embedder, fake_identify
):
    from cardstream.core.models import BoundingBox

    tracker = FakeTracker(results=[(True, BoundingBox(42, 43, 50, 70))])
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        tracker=tracker,
        tracking_detect_interval_seconds=60.0,  # a re-sync must never fire in-test
    )
    snap = settle(analyzer, frames=10)
    assert fake_detector.calls == 1  # one detect, tracker took over
    assert tracker.inits == [fake_detector.detection.bbox]
    assert tracker.updates == 9  # every frame after the detect
    assert snap.bbox.as_list() == [42, 43, 50, 70]  # fresh tracked bbox, not stale


def test_tracker_lost_forces_redetect(fake_detector, fake_embedder, fake_identify):
    from cardstream.core.models import BoundingBox

    tracker = FakeTracker(results=[(True, BoundingBox(42, 43, 50, 70)), (False, None)])
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        tracker=tracker,
        tracking_detect_interval_seconds=60.0,
    )
    analyzer.process(make_frame())  # detect #1 -> tracker init
    analyzer.process(make_frame())  # tracked ok
    assert fake_detector.calls == 1
    analyzer.process(make_frame())  # tracker lost -> detect NOW
    assert fake_detector.calls == 2
    assert len(tracker.inits) == 2  # re-seeded by the new detection


def test_forget_after_re_identifies_a_card_that_stayed_away(
    monkeypatch, fake_detector, fake_embedder, fake_identify
):
    """End-to-end through the analyzer: the gate is cleared after the gap, so
    the returning card costs a fresh call even though it looks identical."""
    from cardstream.client import analyzer as analyzer_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(analyzer_mod.time, "monotonic", lambda: clock["t"])
    a = make_analyzer(
        fake_detector, fake_embedder, fake_identify, forget_after_seconds=5.0
    )

    settle(a)
    assert fake_identify.calls == 1

    fake_detector.detection = None  # card leaves
    settle(a)
    clock["t"] += 6.0  # ...for longer than 5 s
    fake_detector.detection = type(fake_detector)().detection  # identical card back
    settle(a)
    assert fake_identify.calls == 2  # analysed fresh


def test_forget_after_zero_keeps_todays_behaviour(
    monkeypatch, fake_detector, fake_embedder, fake_identify
):
    from cardstream.client import analyzer as analyzer_mod

    clock = {"t": 0.0}
    monkeypatch.setattr(analyzer_mod.time, "monotonic", lambda: clock["t"])
    a = make_analyzer(
        fake_detector, fake_embedder, fake_identify, forget_after_seconds=0.0
    )

    settle(a)
    fake_detector.detection = None
    settle(a)
    clock["t"] += 600.0
    fake_detector.detection = type(fake_detector)().detection
    settle(a)
    assert fake_identify.calls == 1  # same card, no matter the gap


# --- analysis frame vs identification crop ------------------------------------
#
# Detection, motion, tracking and the identity gate run on a downscaled frame;
# the crop that costs money is cut from the original. These tests are the only
# guard on that split — everything above uses 320x240 frames, which are below
# the analysis width and so are never downscaled at all.


def big_frame(w=1920, h=1080, fill=40):
    return np.full((h, w, 3), fill, dtype=np.uint8)


def settle_big(analyzer, frames=6, **kw):
    snap = None
    for _ in range(frames):
        snap = analyzer.process(big_frame(**kw))
    return snap


def test_identify_crop_is_cut_from_the_full_frame(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, analysis_width=480
    )
    settle_big(analyzer)
    assert fake_identify.calls == 1
    # The detector's box is 50x70 in a 480-wide analysis frame; the frame is
    # 1920 wide, so the crop must come back 4x bigger — not resized, RE-CUT.
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (70 * 4, 50 * 4)


def test_detector_and_tracker_see_the_analysis_frame(
    fake_detector, fake_embedder, fake_identify
):
    tracker = FakeTracker()
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, tracker=tracker, analysis_width=480
    )
    settle_big(analyzer)
    assert fake_detector.frames_seen[0] == (480, 270)  # not (1920, 1080)
    assert analyzer.analysis_size == (480, 270)
    assert tracker.inits == [fake_detector.detection.bbox]  # analysis-space bbox


def test_gate_still_sees_the_small_crop(fake_detector, fake_embedder, fake_identify):
    """Gate embeddings are compared against one stored from an earlier frame,
    so the gate stays on the fixed-width analysis crop."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, analysis_width=480
    )
    settle_big(analyzer)
    assert fake_embedder.crop_shapes[0] == (50, 70)  # the detector's own crop


def test_analysis_width_zero_analyses_at_full_resolution(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, analysis_width=0
    )
    settle_big(analyzer)
    assert fake_detector.frames_seen[0] == (1920, 1080)
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (70, 50)  # scale 1: box as-is


def test_small_frame_is_never_upscaled(fake_detector, fake_embedder, fake_identify):
    """A 320x240 source under a 960 target is analysed as-is."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, analysis_width=960
    )
    settle(analyzer)
    assert fake_detector.frames_seen[0] == (320, 240)
    assert analyzer.analysis_size == (320, 240)
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (70, 50)


def test_identify_crop_does_not_alias_the_frame(
    fake_detector, fake_embedder, fake_identify
):
    """The crop outlives the frame it came from (identify runs in a thread), so
    it must own its pixels."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, analysis_width=480
    )
    frame = big_frame()
    for _ in range(6):
        analyzer.process(frame)
    (crop,) = fake_identify.crops
    before = crop.copy()
    frame[:] = 0  # next capture overwrites
    assert np.array_equal(crop, before)
    assert not np.shares_memory(crop, frame)


class CenteredCardDetector(CardDetector):
    """Reports the rectangle textured_card_frame() draws, scaled to whatever
    frame it is handed — a stand-in for a model, so this test needs no weights."""

    name = "centered"

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x0, y0 = round(w * (1420 / 3840)), round(h * (380 / 2160))
        cw, ch = round(w * (1000 / 3840)), round(h * (1400 / 2160))
        return DetectionResult(
            bbox=BoundingBox(x=x0, y=y0, w=cw, h=ch),
            crop=frame_bgr[y0 : y0 + ch, x0 : x0 + cw].copy(),
        )


def textured_card_frame(w=3840, h=2160, card_w=1000, card_h=1400):
    """A card-shaped rectangle carrying fine, high-frequency detail — the kind
    of print (set codes, card numbers) that only survives at full resolution."""
    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    x0, y0 = (w - card_w) // 2, (h - card_h) // 2
    card = np.full((card_h, card_w, 3), 230, dtype=np.uint8)
    card[::4, :] = 30  # 4-px stripes: gone once you downscale by 4
    card[:, ::4] = 30
    frame[y0 : y0 + card_h, x0 : x0 + card_w] = card
    return frame


def test_identify_crop_keeps_detail_a_downscale_would_have_destroyed(fake_identify):
    """The point of the whole change: the crop must be RE-CUT from the original,
    not the analysis crop scaled back up. Upscaling restores the pixel count but
    not the detail, so compare sharpness, not just shape."""
    import cv2

    from cardstream.core.imaging import FramePair

    frame = textured_card_frame()
    analyzer = make_analyzer(
        CenteredCardDetector(), None, fake_identify, gate="phash", analysis_width=960
    )
    for _ in range(6):
        analyzer.process(frame)
    assert fake_identify.calls == 1
    crop = fake_identify.crops[0]

    # Same box, the old way: cut from the analysis frame and scaled back up.
    pair = FramePair.from_frame(frame, 960)
    det = CenteredCardDetector().detect(pair.analysis)
    small = cv2.resize(
        det.crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    def sharpness(img):
        return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    assert crop.shape[1] > 900  # full-resolution pixels
    assert sharpness(crop) > 3 * sharpness(small)  # ...carrying real detail


# --- live tuning -------------------------------------------------------------


def test_tune_replaces_the_frozen_config_rather_than_shadowing_it(
    fake_detector, fake_embedder, fake_identify
):
    """The settings dialog retunes running analyzers. Before, that meant a
    mutable instance attribute shadowing a frozen config field — which works
    for exactly one knob and silently diverges from cfg for every reader that
    kept using self._cfg."""
    from cardstream.client.analyzer import LIVE_FIELDS

    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    assert analyzer.result_threshold == analyzer._cfg.result_threshold

    analyzer.tune(result_threshold=0.25)
    assert analyzer.result_threshold == 0.25
    assert analyzer._cfg.result_threshold == 0.25  # the config IS the truth
    assert "result_threshold" in LIVE_FIELDS


def test_result_threshold_is_read_only(fake_detector, fake_embedder, fake_identify):
    # Assigning would re-create the shadow attribute the property exists to
    # prevent, and _identify reads the config.
    with pytest.raises(AttributeError):
        make_analyzer(
            fake_detector, fake_embedder, fake_identify
        ).result_threshold = 0.1


def test_tune_refuses_knobs_that_would_silently_do_nothing(
    fake_detector, fake_embedder, fake_identify
):
    """Gate thresholds are baked into the gate object and detect intervals into
    DecisionCore, so accepting them here would look like it worked."""
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    with pytest.raises(ValueError, match="not live-tunable"):
        analyzer.tune(motion_threshold=99.0)
    with pytest.raises(ValueError, match="not live-tunable"):
        analyzer.tune(detect_interval_seconds=5.0)
    assert analyzer._cfg.motion_threshold != 99.0


# --- --min-card-size ---------------------------------------------------------

# FakeDetector's box is 50x70 in a 320x240 frame: 0.156 of the width, 0.292 of
# the height. So a threshold between those two rejects on WIDTH alone, which is
# the "too narrow horizontally" case the flag exists for.


def test_a_box_under_the_threshold_is_ignored_entirely(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.2
    )
    snap = settle(analyzer, frames=10)
    assert fake_identify.calls == 0  # never paid for a sliver
    assert snap.state == CardState.EMPTY  # and the card reads as absent
    assert snap.bbox is None
    assert fake_detector.calls > 0  # detection still ran; its result was dropped


def test_a_box_above_the_threshold_is_untouched(
    fake_detector, fake_embedder, fake_identify
):
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.1
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 1


def test_zero_accepts_every_box(fake_detector, fake_embedder, fake_identify):
    """The escape hatch for wide framing, where real cards are a small part of
    the shot and the fraction would reject them."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.0
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 1


def test_either_dimension_is_enough_to_reject(
    fake_detector, fake_embedder, fake_identify
):
    """Short but wide is rejected too — 'narrow' means either axis."""
    fake_detector.detection = DetectionResult(
        bbox=BoundingBox(x=0, y=0, w=300, h=20),  # 0.94 wide, 0.08 tall
        crop=np.full((20, 300, 3), 128, dtype=np.uint8),
    )
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.15
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 0


def test_the_rejection_is_logged_under_debug(
    fake_detector, fake_embedder, fake_identify
):
    lines = []
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        on_log=lines.append,
        min_card_fraction=0.2,
        debug=True,
    )
    settle(analyzer, frames=4)
    assert any("[size]" in line and "50x70" in line for line in lines)


# --- --store-images-type frame -----------------------------------------------


def test_frame_mode_stores_the_full_frame_not_the_crop(
    fake_detector, fake_embedder, fake_identify, tmp_path
):
    """The identify path only ever sees the crop, so the whole picture has to
    come from the analyzer — and it must be the ORIGINAL, not the downscale."""
    import cv2

    from cardstream.core.image_store import FRAME, ImageStore

    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        store=ImageStore(tmp_path, FRAME),
        analysis_width=160,
    )
    settle(analyzer, frames=10)

    assert fake_identify.calls == 1
    (written,) = list(tmp_path.iterdir())
    saved = cv2.imread(str(written))
    assert saved.shape[:2] == (240, 320)  # make_frame()'s full size, not 160-wide


def test_object_mode_writes_nothing_from_the_analyzer(
    fake_detector, fake_embedder, fake_identify, tmp_path
):
    """In object mode the crop is kept by the identify client, which the fake
    stands in for here — so the analyzer itself must leave the folder empty."""
    from cardstream.core.image_store import OBJECT, ImageStore

    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, store=ImageStore(tmp_path, OBJECT)
    )
    settle(analyzer, frames=10)
    assert not list(tmp_path.iterdir())


# --- --min-card-aspect-ratio -------------------------------------------------


def _boxed(w, h):
    return DetectionResult(
        bbox=BoundingBox(x=0, y=0, w=w, h=h),
        crop=np.full((h, w, 3), 128, dtype=np.uint8),
    )


def test_the_real_sliver_is_rejected(fake_detector, fake_embedder, fake_identify):
    """214x702 — the crop that actually reached Ximilar before this existed.
    Its aspect is 0.30 against a card's ~0.71."""
    fake_detector.detection = _boxed(214, 702)
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.0
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 0


def test_a_real_card_shape_passes_either_way_up(
    fake_detector, fake_embedder, fake_identify
):
    """0.71 portrait and 1.4 landscape are the SAME card, so both must pass —
    that is why the ratio is short-side-over-long, not width-over-height."""
    for w, h in [(550, 702), (702, 550)]:
        fake_detector.detection = _boxed(w, h)
        analyzer = make_analyzer(
            fake_detector, fake_embedder, fake_identify, min_card_fraction=0.0
        )
        fake_identify.calls = 0
        settle(analyzer, frames=10)
        assert fake_identify.calls == 1, f"{w}x{h} was rejected"


def test_a_wide_sliver_is_rejected_too(fake_detector, fake_embedder, fake_identify):
    """The orientation-blind half: 700x200 is 0.29, as wrong as 200x700."""
    fake_detector.detection = _boxed(700, 200)
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, min_card_fraction=0.0
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 0


def test_zero_accepts_any_shape(fake_detector, fake_embedder, fake_identify):
    fake_detector.detection = _boxed(214, 702)
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        min_card_fraction=0.0,
        min_card_aspect=0.0,
    )
    settle(analyzer, frames=10)
    assert fake_identify.calls == 1


def test_the_aspect_rejection_is_logged_under_debug(
    fake_detector, fake_embedder, fake_identify
):
    lines = []
    fake_detector.detection = _boxed(214, 702)
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        on_log=lines.append,
        min_card_fraction=0.0,
        debug=True,
    )
    settle(analyzer, frames=4)
    assert any("[aspect]" in line and "0.30" in line for line in lines)


# --- a locator that found CORNERS gets a deskewed identify crop ----------------


class TiltedCardSegmentor(CardDetector):
    """Stand-in for a segmentor: reports the four corners of the tilted card
    that tilted_card_frame() draws, in analysis-frame coords."""

    name = "fake-segmentor"

    def __init__(self, card=(400, 600), angle=25.0):
        self.card = card
        self.angle = angle

    def detect(self, frame_bgr):
        import cv2

        from cardstream.core.quad import mask_to_quad, quad_bbox, warp_quad

        h, w = frame_bgr.shape[:2]
        cw, ch = self.card[0] * w / 1920, self.card[1] * h / 1080
        mask = np.zeros((h, w), np.uint8)
        box = cv2.boxPoints(((w / 2, h / 2), (cw, ch), self.angle))
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
        quad = mask_to_quad(mask.astype(bool))
        return DetectionResult(
            bbox=quad_bbox(quad), crop=warp_quad(frame_bgr, quad), prob=0.9, quad=quad
        )


def tilted_card_frame(w=1920, h=1080, card=(400, 600), angle=25.0):
    """A card-shaped rectangle rotated in the frame — the case an axis-aligned
    box cannot cut cleanly."""
    import cv2

    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    box = cv2.boxPoints(((w / 2, h / 2), card, angle)).astype(np.int32)
    cv2.fillPoly(frame, [box], (230, 230, 230))
    return frame


def test_a_quad_routes_the_paid_crop_through_warp(fake_embedder, fake_identify):
    """The whole point of --segmentor: the crop that costs money is deskewed
    and tight, not the square cut with background wedges in the corners."""
    analyzer = make_analyzer(
        TiltedCardSegmentor(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
    )
    for _ in range(10):
        analyzer.process(tilted_card_frame())
    (crop,) = fake_identify.crops
    h, w = crop.shape[:2]
    # Full resolution (the frame was analysed at 960 but cut from 1920) and the
    # CARD's dimensions, not the tilt-inflated hull's.
    assert (w, h) == pytest.approx((400, 600), abs=12)
    # Tight: almost every pixel is card. The same card through the box path
    # lands nearer 80%.
    assert (crop > 128).mean() > 0.95


def test_a_box_locator_still_gets_the_square_cut(
    fake_detector, fake_embedder, fake_identify
):
    """quad=None is what every box detector reports, and it must keep taking
    the pair.crop path untouched."""
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    settle(analyzer, frames=10)
    assert fake_detector.detection.quad is None
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (70, 50)  # exactly the detector's bbox


def test_the_deskew_is_marked_in_the_debug_log(fake_embedder, fake_identify):
    """The debug line is how you tell which locator paid for the call."""
    lines = []
    analyzer = make_analyzer(
        TiltedCardSegmentor(),
        fake_embedder,
        fake_identify,
        on_log=lines.append,
        debug=True,
        analysis_width=960,
    )
    for _ in range(10):
        analyzer.process(tilted_card_frame())
    assert any("calling identify" in line and "deskewed" in line for line in lines)


# --- --detection-expansion: grow the PAID crop, and only that -----------------


class MidFrameDetector(CardDetector):
    """A 50x70 box well clear of the frame edges, so expansion has room."""

    name = "mid-frame"

    def __init__(self) -> None:
        self.bbox = BoundingBox(x=100, y=80, w=50, h=70)
        crop = np.full((70, 50, 3), 128, dtype=np.uint8)
        self.detection = DetectionResult(bbox=self.bbox, crop=crop)

    def detect(self, frame_bgr):
        return self.detection


def test_expansion_grows_the_box_locators_paid_crop(fake_embedder, fake_identify):
    """0.25 on a 50x70 box pushes each edge out by 12 / 18 -> 74x106."""
    analyzer = make_analyzer(
        MidFrameDetector(), fake_embedder, fake_identify, detection_expansion=0.25
    )
    settle(analyzer, frames=10)
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (106, 74)


def test_an_expanded_box_is_clamped_to_the_frame(
    fake_detector, fake_embedder, fake_identify
):
    """Growing a card already at the frame edge cannot invent pixels — the cut
    stops at the border rather than failing or wrapping."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, detection_expansion=0.25
    )
    settle(analyzer, frames=10)
    (crop,) = fake_identify.crops
    # bbox (10,10,50,70) grown by 12/18 starts at (-2,-8): clamped to (0,0).
    assert crop.shape[:2] == (98, 72)


def test_no_expansion_by_default(fake_detector, fake_embedder, fake_identify):
    """The default must send exactly what was located."""
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    settle(analyzer, frames=10)
    (crop,) = fake_identify.crops
    assert crop.shape[:2] == (70, 50)


def test_expansion_grows_a_segmentors_deskewed_crop(fake_embedder, fake_identify):
    """The quad's corners are pushed out from the centre, so the crop is still
    deskewed — just with a margin of context around the card."""
    tight = make_analyzer(
        TiltedCardSegmentor(), fake_embedder, fake_identify, analysis_width=960
    )
    for _ in range(10):
        tight.process(tilted_card_frame())
    (plain,) = fake_identify.crops

    fake_identify.crops.clear()
    grown = make_analyzer(
        TiltedCardSegmentor(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
        detection_expansion=0.2,
    )
    for _ in range(10):
        grown.process(tilted_card_frame())
    (padded,) = fake_identify.crops

    ph, pw = plain.shape[:2]
    gh, gw = padded.shape[:2]
    assert (gw, gh) == pytest.approx((pw * 1.4, ph * 1.4), rel=0.05)
    # Still deskewed: the card is centred with background only at the edges,
    # so the middle stays card and the corners pick up the dark surround.
    assert padded[gh // 2, gw // 2].mean() > 128
    assert padded[2, 2].mean() < 128


def test_expansion_leaves_the_gate_and_the_overlay_alone(
    fake_detector, fake_embedder, fake_identify
):
    """It changes what is IDENTIFIED, not what was LOCATED — a padded gate crop
    would dilute the SAME-vs-NEW decision with background."""
    analyzer = make_analyzer(
        fake_detector, fake_embedder, fake_identify, detection_expansion=0.25
    )
    snap = settle(analyzer, frames=10)
    assert snap.bbox.as_list() == [10, 10, 50, 70]  # the overlay's box
    assert fake_detector.detection.crop.shape[:2] == (70, 50)  # the gate's crop


def test_expansion_is_marked_in_the_debug_log(
    fake_detector, fake_embedder, fake_identify
):
    lines = []
    analyzer = make_analyzer(
        fake_detector,
        fake_embedder,
        fake_identify,
        detection_expansion=0.1,
        on_log=lines.append,
        debug=True,
    )
    settle(analyzer, frames=10)
    assert any("calling identify" in line and "+10%" in line for line in lines)


def bordered_card_frame(w=1920, h=1080, card=(200, 280)):
    """A white card on a dark field, ringed by a distinctly coloured border.

    The ring sits just OUTSIDE the card, so it is the evidence that an expanded
    crop really pulled in surrounding pixels rather than just being scaled up.
    """
    import cv2

    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cw, ch = card
    cv2.rectangle(frame, (cx - cw, cy - ch), (cx + cw, cy + ch), (0, 0, 255), 60)
    cv2.rectangle(frame, (cx - cw, cy - ch), (cx + cw, cy + ch), (235, 235, 235), -1)
    return frame


class FullResBoxDetector(CardDetector):
    """Reports the bordered card's box in ANALYSIS coords, whatever the scale."""

    name = "full-res-box"

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        cw, ch = round(w * (200 / 1920)), round(h * (280 / 1080))
        bbox = BoundingBox(x=w // 2 - cw, y=h // 2 - ch, w=2 * cw, h=2 * ch)
        return DetectionResult(
            bbox=bbox,
            crop=frame_bgr[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w].copy(),
        )


def test_expansion_is_applied_before_the_cut_from_the_ORIGINAL_frame(
    fake_embedder, fake_identify
):
    """The order the flag promises: locate on the analysis frame, expand there,
    rescale, then cut from the full-resolution original. A crop expanded AFTER
    the cut would come back at analysis resolution instead."""
    analyzer = make_analyzer(
        FullResBoxDetector(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
        detection_expansion=0.25,
    )
    for _ in range(10):
        analyzer.process(bordered_card_frame())
    (crop,) = fake_identify.crops
    # Analysis box is 200x280 at 960 wide; +25% a side -> 300x420; x2 back to
    # the 1920-wide original -> 600x840. Not the 300x420 an analysis-space cut
    # would have produced.
    h, w = crop.shape[:2]
    assert (w, h) == pytest.approx((600, 840), abs=4)


def test_expansion_pulls_in_real_surrounding_pixels(fake_embedder, fake_identify):
    """Not just a bigger canvas: the ring that sits outside the card has to
    appear in the expanded crop and be absent from the tight one."""

    def run(grow):
        fake_identify.crops.clear()
        a = make_analyzer(
            FullResBoxDetector(),
            fake_embedder,
            fake_identify,
            analysis_width=960,
            detection_expansion=grow,
        )
        for _ in range(10):
            a.process(bordered_card_frame())
        return fake_identify.crops[0]

    tight, grown = run(0.0), run(0.25)

    # The border is drawn in red (BGR): count strongly-red pixels in each.
    def red_fraction(img):
        b, g, r = (
            img[:, :, 0].astype(int),
            img[:, :, 1].astype(int),
            img[:, :, 2].astype(int),
        )
        return ((r > 150) & (g < 100) & (b < 100)).mean()

    assert red_fraction(tight) < 0.01  # tight cut is card, no ring
    assert red_fraction(grown) > 0.10  # expanded cut brought the ring in


def test_expansion_of_one_is_accepted_end_to_end(fake_embedder, fake_identify):
    """The inclusive top of the range must actually work, not just parse — the
    warp/crop guards have to have headroom for a card that triples."""
    analyzer = make_analyzer(
        MidFrameDetector(), fake_embedder, fake_identify, detection_expansion=1.0
    )
    settle(analyzer, frames=10)
    (crop,) = fake_identify.crops
    assert crop.size > 0


def test_the_expanded_crop_is_what_reaches_the_wire_and_the_store(
    tmp_path, fake_embedder, monkeypatch
):
    """The whole chain in one go: expand -> cut from the original -> b64 on the
    wire -> the same bytes on disk. --store-images-type object exists to answer
    "what did I actually pay for", so it has to show the EXPANDED crop."""
    import base64

    import cv2

    from cardstream.client.ximilar_api import DirectXimilarClient
    from cardstream.core import ximilar as core_ximilar
    from cardstream.core.identify_options import IdentifyOptions
    from cardstream.core.image_store import OBJECT, ImageStore

    captured = {}

    class _Resp:
        status_code, ok, text = 200, True, "{}"

        def json(self):
            return {"records": [{"_status": {"code": 200}}]}

    def fake_post(url, json, headers, timeout):
        captured["b64"] = json["records"][0]["_base64"]
        captured["objects"] = json["records"][0]["_objects"]
        return _Resp()

    monkeypatch.setattr(core_ximilar.requests, "post", fake_post)

    store = ImageStore(str(tmp_path), kind=OBJECT)
    client = DirectXimilarClient("key", IdentifyOptions("tcg"), store=store)
    analyzer = make_analyzer(
        FullResBoxDetector(),
        fake_embedder,
        client,
        analysis_width=960,
        detection_expansion=0.25,
        store=store,
    )
    for _ in range(10):
        analyzer.process(bordered_card_frame())

    posted = cv2.imdecode(
        np.frombuffer(base64.b64decode(captured["b64"]), np.uint8), cv2.IMREAD_COLOR
    )
    h, w = posted.shape[:2]
    assert (w, h) == pytest.approx((600, 840), abs=4)  # expanded, full-res
    # The endpoint is told the card fills the sent image, whatever we expanded to.
    assert captured["objects"][0]["bound_box"] == [0, 0, w, h]

    (saved,) = sorted(tmp_path.glob("*.jpg"))
    on_disk = cv2.imread(str(saved))
    assert on_disk.shape == posted.shape


# --- the overlay's two outlines: located vs what gets paid for ----------------


def test_no_crop_outline_without_expansion(fake_detector, fake_embedder, fake_identify):
    """With nothing to compare, the page draws one outline — sending a second
    identical shape would just be a heavier line."""
    analyzer = make_analyzer(fake_detector, fake_embedder, fake_identify)
    assert settle(analyzer, frames=10).crop_quad is None


def test_the_crop_outline_is_the_expanded_box(fake_embedder, fake_identify):
    """A box locator still reports four corners, so the page has one kind of
    shape to draw whichever locator is running."""
    analyzer = make_analyzer(
        MidFrameDetector(), fake_embedder, fake_identify, detection_expansion=0.25
    )
    snap = settle(analyzer, frames=10)
    assert snap.quad is None  # a box locator found it...
    assert snap.crop_quad is not None  # ...but the crop is a quad
    assert quad_bbox(snap.crop_quad).as_list() == [88, 62, 74, 106]
    assert snap.bbox.as_list() == [100, 80, 50, 70]  # located outline untouched


def test_the_crop_outline_matches_the_crop_that_is_actually_cut(
    fake_embedder, fake_identify
):
    """The green outline is a promise about what gets paid for — if the drawn
    shape and the cut crop could drift apart, the overlay would be lying."""
    analyzer = make_analyzer(
        FullResBoxDetector(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
        detection_expansion=0.25,
    )
    for _ in range(10):
        analyzer.process(bordered_card_frame())
    snap = analyzer.process(bordered_card_frame())
    (crop,) = fake_identify.crops

    outline = quad_bbox(snap.crop_quad)
    ch, cw = crop.shape[:2]
    # The outline is in analysed-frame coords; the crop was cut from the
    # original, twice the size at --width 960 on a 1920 frame.
    assert (outline.w * 2, outline.h * 2) == pytest.approx((cw, ch), abs=4)


def test_the_crop_outline_follows_a_segmentors_corners(fake_embedder, fake_identify):
    """Expanded from the quad, not from its hull — otherwise the green outline
    would sit square while the crop it describes is tilted."""
    analyzer = make_analyzer(
        TiltedCardSegmentor(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
        detection_expansion=0.2,
    )
    snap = None
    for _ in range(10):
        snap = analyzer.process(tilted_card_frame())
    assert snap.crop_quad is not None
    # Same centre, same tilt, 1.4x the edges.
    assert np.allclose(snap.crop_quad.mean(axis=0), snap.quad.mean(axis=0), atol=1.0)
    located = np.linalg.norm(snap.quad[1] - snap.quad[0])
    grown = np.linalg.norm(snap.crop_quad[1] - snap.crop_quad[0])
    assert grown == pytest.approx(located * 1.4, rel=0.02)


def test_both_outlines_serialize_for_the_page(fake_embedder, fake_identify):
    import json

    analyzer = make_analyzer(
        TiltedCardSegmentor(),
        fake_embedder,
        fake_identify,
        analysis_width=960,
        detection_expansion=0.2,
    )
    snap = None
    for _ in range(10):
        snap = analyzer.process(tilted_card_frame())
    d = snap.to_dict()
    assert len(d["quad"]) == 4 and len(d["crop_quad"]) == 4
    assert all(isinstance(v, int) for pt in d["crop_quad"] for v in pt)
    json.dumps(d)
