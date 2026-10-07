"""DecisionCore / CallGuard / PhashGate — the shared state machine, unit-tested
once with a fake clock (SmartAnalyzer is a thin shell over this)."""

from __future__ import annotations

import numpy as np

from _helpers import textured_crop
from cardstream.core.engine import (
    CallGuard,
    DecisionCore,
    DetectIntervals,
    PhashGate,
    Snapshot,
)
from cardstream.core.models import BoundingBox, CardState, DetectionResult


def make_core(
    moving=0.3,
    idle=1.0,
    empty=3.0,
    tracking=2.0,
    cooldown=0.0,
    timeout=20.0,
    log=None,
    use_tracker=False,
    forget_after=0.0,
    retry_unmatched=0.0,
    send_blurred_after=0.0,
) -> DecisionCore:
    return DecisionCore(
        intervals=DetectIntervals(
            moving=moving, idle_with_card=idle, empty=empty, tracking=tracking
        ),
        motion_threshold=4.0,
        gate=PhashGate(distance_threshold=10),
        cooldown_seconds=cooldown,
        call_timeout_seconds=timeout,
        log=log,
        use_tracker=use_tracker,
        forget_after_seconds=forget_after,
        retry_unmatched_seconds=retry_unmatched,
        send_blurred_after_seconds=send_blurred_after,
    )


def det(seed=0) -> DetectionResult:
    crop = textured_crop(seed=seed)
    return DetectionResult(bbox=BoundingBox(10, 10, 64, 64), crop=crop)


# --- three-tier detection throttle ------------------------------------------


def test_moving_scene_uses_fast_interval():
    core = make_core(moving=0.3, idle=1.0, empty=3.0)
    assert core.on_frame(settled=False, score=9.0, now=100.0) is True  # first fire
    assert core.on_detection(det(), settled=False, now=100.0) is False  # not settled
    assert core.on_frame(settled=False, score=9.0, now=100.2) is False  # < 0.3s
    assert core.on_frame(settled=False, score=9.0, now=100.31) is True  # >= 0.3s


def test_static_with_card_uses_idle_interval():
    core = make_core(moving=0.3, idle=1.0, empty=3.0)
    assert (
        core.on_frame(settled=True, score=0.0, now=100.0) is True
    )  # empty tier, first fire
    core.on_detection(det(), settled=True, now=100.0)  # card present now
    assert (
        core.on_frame(settled=True, score=0.0, now=100.5) is False
    )  # < 1.0s idle tier
    assert core.on_frame(settled=True, score=0.0, now=101.01) is True  # >= 1.0s


def test_static_empty_uses_slow_heartbeat():
    core = make_core(moving=0.3, idle=1.0, empty=3.0)
    assert core.on_frame(settled=True, score=0.0, now=100.0) is True
    core.on_detection(None, settled=True, now=100.0)  # nothing there
    assert core.on_frame(settled=True, score=0.0, now=101.5) is False  # < 3s empty tier
    assert core.on_frame(settled=True, score=0.0, now=103.01) is True  # >= 3s


def test_no_detect_while_one_is_in_flight():
    core = make_core(moving=0.0)
    assert core.on_frame(settled=False, score=9.0, now=100.0) is True
    # detection still in flight -> never a second one, even at interval 0
    assert core.on_frame(settled=False, score=9.0, now=105.0) is False


# --- tracking tier ------------------------------------------------------------


def test_tracking_tier_stretches_detection_even_while_moving():
    core = make_core(moving=0.3, idle=1.0, empty=3.0, tracking=2.0, use_tracker=True)
    assert core.on_frame(settled=False, score=9.0, now=100.0) is True
    core.on_detection(det(), settled=False, now=100.0)  # card found -> tracking
    assert core.tracking is True
    # tracker locked: even a moving scene waits for the 2s re-sync interval
    assert core.on_frame(settled=False, score=9.0, now=100.4) is False
    assert core.on_frame(settled=False, score=9.0, now=101.9) is False
    assert core.on_frame(settled=False, score=9.0, now=102.01) is True


def test_on_track_refreshes_snapshot_bbox():
    core = make_core(use_tracker=True)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(), settled=True, now=100.0)
    moved = BoundingBox(30, 40, 64, 64)
    core.on_track(True, moved)
    assert core.snapshot().bbox == moved
    assert core.tracking is True


def test_tracker_lost_forces_immediate_detect():
    core = make_core(moving=0.3, tracking=60.0, use_tracker=True)
    core.on_frame(settled=False, score=9.0, now=100.0)
    core.on_detection(det(), settled=False, now=100.0)
    assert (
        core.on_frame(settled=False, score=0.0, now=100.1) is False
    )  # within tracking tier
    core.on_track(False, None)  # score dropped
    assert core.tracking is False
    assert (
        core.on_frame(settled=False, score=0.0, now=100.2) is True
    )  # fires NOW, no interval wait
    # the card is still 'present' with its last bbox until detection resolves
    assert core.snapshot().state is not CardState.EMPTY


def test_detection_none_clears_tracking():
    core = make_core(moving=0.0, use_tracker=True)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(), settled=True, now=100.0)
    assert core.tracking is True
    core.on_frame(settled=True, score=9.0, now=105.0)
    core.on_detection(None, settled=True, now=105.0)  # card removed
    assert core.tracking is False
    assert core.snapshot().state is CardState.EMPTY


def test_tracker_failed_forces_immediate_detect():
    """Driver-side tracker failure (e.g. init raised) behaves like a lost
    track: tracking cleared AND the throttle expired, not a full-tier wait."""
    core = make_core(moving=0.3, tracking=60.0, use_tracker=True)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(), settled=True, now=100.0)
    assert core.tracking is True
    core.on_tracker_failed()
    assert core.tracking is False
    assert core.on_frame(settled=True, score=0.0, now=100.1) is True  # no interval wait


def test_without_tracker_detection_never_enters_tracking():
    core = make_core(use_tracker=False)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(), settled=True, now=100.0)
    assert core.tracking is False


# --- watchdog ---------------------------------------------------------------


def test_watchdog_clears_stuck_detect_flag():
    warnings: list[str] = []
    core = make_core(moving=0.0, timeout=20.0, log=warnings.append)
    assert core.on_frame(settled=False, score=9.0, now=100.0) is True
    # hung call: 21s later the watchdog clears the flag and detection resumes
    assert core.on_frame(settled=False, score=9.0, now=121.0) is True
    assert any("detection watchdog" in w for w in warnings)


def test_watchdog_clears_stuck_identify_flag():
    warnings: list[str] = []
    core = make_core(moving=0.0, timeout=20.0, log=warnings.append)
    core.on_frame(settled=True, score=9.0, now=100.0)
    assert (
        core.on_detection(det(), settled=True, now=100.0) is True
    )  # identify launched
    assert core.identify.in_flight is True
    core.on_frame(settled=True, score=9.0, now=121.0)
    assert core.identify.in_flight is False
    assert any("identify watchdog" in w for w in warnings)


# --- identity gate policy ---------------------------------------------------


def test_same_card_is_not_reidentified():
    core = make_core(moving=0.0)
    core.on_frame(settled=True, score=9.0, now=100.0)
    assert core.on_detection(det(seed=1), settled=True, now=100.0) is True
    core.on_identify_done({"full_name": "Charizard"})
    core.on_frame(settled=True, score=9.0, now=101.0)
    assert core.on_detection(det(seed=1), settled=True, now=101.0) is False  # same card
    core.on_frame(settled=True, score=9.0, now=102.0)
    assert core.on_detection(det(seed=2), settled=True, now=102.0) is True  # new card


def test_cooldown_suppresses_identify():
    """Two DIFFERENT cards back to back: the second waits out the cooldown."""
    core = make_core(moving=0.0, cooldown=5.0)
    core.on_frame(settled=True, score=9.0, now=100.0)
    assert core.on_detection(det(seed=1), settled=True, now=100.0) is True
    core.on_identify_done({"x": 1})
    core.on_frame(settled=True, score=9.0, now=102.0)
    assert (
        core.on_detection(det(seed=2), settled=True, now=102.0) is False
    )  # in cooldown
    core.on_frame(settled=True, score=9.0, now=105.1)
    assert core.on_detection(det(seed=2), settled=True, now=105.1) is True


def test_failed_identify_not_retried_for_same_card():
    core = make_core(moving=0.0)
    core.on_frame(settled=True, score=9.0, now=100.0)
    assert core.on_detection(det(seed=1), settled=True, now=100.0) is True
    core.on_identify_done(None)  # upstream failed
    core.on_frame(settled=True, score=9.0, now=101.0)
    # signature was committed on fire -> same card does not re-fire
    assert core.on_detection(det(seed=1), settled=True, now=101.0) is False
    core.on_frame(settled=True, score=9.0, now=102.0)
    assert core.on_detection(det(seed=2), settled=True, now=102.0) is True


def test_new_card_drops_stale_identification():
    core = make_core(moving=0.0)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(seed=1), settled=True, now=100.0)
    core.on_identify_done({"full_name": "Charizard"})
    assert core.last_ident is not None
    core.on_frame(settled=True, score=9.0, now=101.0)
    assert core.on_detection(det(seed=2), settled=True, now=101.0) is True
    assert core.last_ident is None  # stale name dropped


def test_unsettled_detection_never_identifies():
    core = make_core(moving=0.0)
    core.on_frame(settled=False, score=9.0, now=100.0)
    assert core.on_detection(det(), settled=False, now=100.0) is False


# --- card-gone / snapshot ladder --------------------------------------------


def test_card_gone_keeps_identity_for_instant_return():
    core = make_core(moving=0.0)
    core.on_frame(settled=True, score=9.0, now=100.0)
    core.on_detection(det(seed=1), settled=True, now=100.0)
    core.on_identify_done({"full_name": "Charizard"})

    core.on_frame(settled=True, score=9.0, now=101.0)
    core.on_detection(None, settled=True, now=101.0)  # card removed
    assert core.snapshot().state is CardState.EMPTY

    core.on_frame(settled=True, score=9.0, now=102.0)
    assert (
        core.on_detection(det(seed=1), settled=True, now=102.0) is False
    )  # same card back: no call
    snap = core.snapshot()
    state, ident = snap.state, snap.identification
    assert state is CardState.IDENTIFIED and ident == {"full_name": "Charizard"}


def test_snapshot_state_ladder():
    core = make_core(moving=0.0)
    assert core.snapshot().state is CardState.EMPTY

    core.on_frame(settled=False, score=9.0, now=100.0)
    core.on_detection(det(), settled=False, now=100.0)
    assert core.snapshot().state is CardState.MOVING

    core.on_frame(settled=True, score=0.0, now=100.1)
    assert core.snapshot().state is CardState.SETTLED

    core.on_frame(settled=True, score=9.0, now=101.0)
    core.on_detection(det(), settled=True, now=101.0)
    assert core.snapshot().state is CardState.IDENTIFYING  # identify in flight

    core.on_identify_done({"full_name": "X"})
    assert core.snapshot().state is CardState.IDENTIFIED


# --- CallGuard basics --------------------------------------------------------


def test_call_guard_ready_begin_end():
    g = CallGuard("x", timeout_seconds=10.0)
    assert g.ready(now=100.0, interval=1.0) is True
    g.begin(100.0)
    assert g.ready(now=200.0, interval=1.0) is False  # in flight
    g.end()
    assert g.ready(now=100.5, interval=1.0) is False  # inside interval
    assert g.ready(now=101.0, interval=1.0) is True


def test_phash_gate_boundary():
    """Distance <= threshold reads as the same card; > threshold as new."""
    gate = PhashGate(distance_threshold=10)
    crop = textured_crop(seed=5)
    changed, token = gate.decide(crop)
    assert changed is True  # nothing committed yet
    gate.commit(token)
    changed, _ = gate.decide(crop)  # identical crop -> distance 0
    assert changed is False
    changed, _ = gate.decide(textured_crop(seed=6))  # different card
    assert changed is True


# --- forgetting a card that stayed away ---------------------------------------
#
# The gate normally remembers the last card across a dropout, so waving one card
# in and out is free. Past `forget_after_seconds` that stops being the useful
# assumption: the scene has moved on, and trusting a stale signature would veto
# a call that should happen.


def test_long_absence_forgets_the_card_and_re_identifies():
    core = make_core(forget_after=5.0)
    assert core.on_detection(det(0), settled=True, now=0.0)  # identified
    core.on_identify_done({"full_name": "Charizard"})  # (driver does this)
    assert not core.on_detection(det(0), settled=True, now=1.0)  # same card, no call

    core.on_detection(None, settled=True, now=2.0)  # card lost
    assert not core.on_detection(det(0), settled=True, now=5.0)  # back after 3s: same

    core.on_detection(None, settled=True, now=6.0)  # lost again
    assert core.on_detection(det(0), settled=True, now=12.0)  # back after 6s: fresh


def test_short_dropouts_never_forget():
    """The flicker case: detection loses the card for a moment every second."""
    core = make_core(forget_after=5.0)
    assert core.on_detection(det(0), settled=True, now=0.0)
    core.on_identify_done({"full_name": "Charizard"})
    now = 0.0
    for _ in range(10):  # 10 lost/found cycles, 1s apart
        core.on_detection(None, settled=True, now=now + 0.5)
        assert not core.on_detection(det(0), settled=True, now=now + 1.0)
        now += 1.0


def test_forget_clears_the_shown_identification_too():
    core = make_core(forget_after=5.0)
    core.on_detection(det(0), settled=True, now=0.0)
    core.on_identify_done({"full_name": "Charizard"})
    assert core.snapshot().identification == {"full_name": "Charizard"}

    core.on_detection(None, settled=True, now=1.0)
    core.on_detection(det(0), settled=True, now=10.0)
    # Stale name dropped: the overlay must not show the old card while the
    # fresh identify is still in flight.
    assert core.last_ident is None


def test_forget_after_zero_remembers_across_any_gap():
    core = make_core(forget_after=0.0)
    assert core.on_detection(det(0), settled=True, now=0.0)
    core.on_identify_done({"full_name": "Charizard"})
    core.on_detection(None, settled=True, now=1.0)
    assert not core.on_detection(det(0), settled=True, now=1000.0)


def test_forget_warns_so_the_extra_call_is_explainable():
    lines = []
    core = make_core(forget_after=5.0, log=lines.append)
    core.on_detection(det(0), settled=True, now=0.0)
    core.on_detection(None, settled=True, now=1.0)
    core.on_detection(det(0), settled=True, now=9.0)
    assert any("forgetting the last card" in line for line in lines)
    assert any("8.0s" in line for line in lines)


def test_routine_gate_events_and_faults_share_one_channel():
    """One log sink: the driver's on_log, which reaches both the terminal and
    the browser's debug panel. Routine events and faults are distinguished by
    their wording, not by which callback they went down."""
    lines = []
    core = make_core(forget_after=5.0, log=lines.append)

    core.on_detection(det(), settled=True, now=0.0)
    core.on_identify_done({"full_name": "Charizard"})
    core.on_detection(None, settled=False, now=1.0)  # card leaves
    core.on_detection(det(), settled=True, now=20.0)  # ...and comes back late
    assert any("forgetting the last card" in m for m in lines)

    core.detect.begin(0.0)
    core.detect.watchdog(9999.0)
    assert any("watchdog" in m for m in lines)


def _identify_once(core, crop, now):
    """Drive one full settled-card cycle; returns whether an identify fired."""
    core.on_frame(settled=True, score=0.0, now=now)
    fired = core.on_detection(crop, settled=True, now=now)
    return fired


def test_retry_unmatched_zero_never_asks_again():
    """The strict commit-on-fire policy: the signature is remembered even when
    nothing came back, so the same card sitting there is not hammered. This is
    what --retry-unmatched 0 restores."""
    core = make_core(idle=0.0, empty=0.0)
    card = det(seed=1)
    assert _identify_once(core, card, now=100.0) is True
    core.on_identify_done(None)  # no match
    assert _identify_once(core, card, now=200.0) is False  # 100s later, still nothing


def test_retry_unmatched_asks_once_more_after_the_delay():
    core = make_core(idle=0.0, empty=0.0, retry_unmatched=5.0)
    card = det(seed=1)
    assert _identify_once(core, card, now=100.0) is True
    core.on_identify_done(None)
    assert _identify_once(core, card, now=104.9) is False  # inside the window
    assert _identify_once(core, card, now=105.0) is True  # at the window


def test_a_matched_card_is_never_retried():
    """The retry is for cards with no name — a good match must still cost one
    call however long it stays in frame."""
    core = make_core(idle=0.0, empty=0.0, retry_unmatched=5.0)
    card = det(seed=1)
    assert _identify_once(core, card, now=100.0) is True
    core.on_identify_done({"full_name": "Charizard"})
    assert _identify_once(core, card, now=999.0) is False


def test_repeats_are_spaced_from_the_last_attempt_not_the_first():
    core = make_core(idle=0.0, empty=0.0, retry_unmatched=5.0)
    card = det(seed=1)
    _identify_once(core, card, now=100.0)
    core.on_identify_done(None)
    assert _identify_once(core, card, now=105.0) is True  # first retry
    core.on_identify_done(None)
    assert _identify_once(core, card, now=106.0) is False  # not stacked up
    assert _identify_once(core, card, now=110.0) is True  # 5s after the retry


def test_the_retry_is_announced():
    notes = []
    core = make_core(idle=0.0, empty=0.0, log=notes.append, retry_unmatched=5.0)
    card = det(seed=1)
    _identify_once(core, card, now=100.0)
    core.on_identify_done(None)
    _identify_once(core, card, now=105.0)
    assert any("once more" in n for n in notes)


def test_the_cooldown_still_wins_over_a_due_retry():
    """--cooldown is the outer throttle: a retry cannot jump it."""
    core = make_core(idle=0.0, empty=0.0, cooldown=30.0, retry_unmatched=5.0)
    card = det(seed=1)
    assert _identify_once(core, card, now=100.0) is True
    core.on_identify_done(None)
    assert _identify_once(core, card, now=105.0) is False  # retry due, cooldown says no
    assert _identify_once(core, card, now=130.0) is True


# --- the segmentor's corners ride along with the box --------------------------

QUAD = np.float32([[10, 20], [60, 22], [58, 120], [8, 118]])


def seg_det(seed=0) -> DetectionResult:
    """A detection from a SEGMENTOR: the same box, plus the four corners."""
    d = det(seed)
    d.quad = QUAD
    return d


def test_snapshot_carries_the_quad_a_segmentor_found():
    core = make_core()
    core.on_detection(seg_det(), settled=True, now=1.0)
    snap = core.snapshot()
    state, bbox, quad = snap.state, snap.bbox, snap.quad
    assert state is not CardState.EMPTY
    assert bbox is not None
    assert quad is not None and len(quad) == 4


def test_snapshot_quad_is_none_for_a_box_locator():
    """Every detector leaves it unset, and the overlay then draws a rect."""
    core = make_core()
    core.on_detection(det(), settled=True, now=1.0)
    assert core.snapshot().quad is None


def test_a_tracker_update_carries_the_quad_with_the_box():
    """A tracker reports a MOVED BOX; the corners inside it move the same way.

    Dropping them looked safer but meant that with a segmentor + a tracker the
    real outline appeared only on a re-sync frame (2s apart), so the page
    flickered polygon<->rect for a whole second at a time.
    """
    core = make_core(use_tracker=True)
    core.on_detection(seg_det(), settled=True, now=1.0)
    before = core.snapshot().quad
    assert before is not None

    # Same size, shifted by (+40, +30): a pure translation, which the mapping
    # reproduces exactly.
    moved = BoundingBox(x=50, y=40, w=64, h=64)
    core.on_track(True, moved)
    snap = core.snapshot()
    assert snap.bbox == moved
    assert np.allclose(snap.quad, before + np.float32([40, 30]))


def test_a_tracker_update_keeps_a_box_locator_cornerless():
    """No corners to carry: a box locator still reports quad=None."""
    core = make_core(use_tracker=True)
    core.on_detection(det(), settled=True, now=1.0)
    core.on_track(True, BoundingBox(x=50, y=40, w=64, h=64))
    assert core.snapshot().quad is None


def test_losing_the_card_clears_the_quad():
    core = make_core()
    core.on_detection(seg_det(), settled=True, now=1.0)
    core.on_detection(None, settled=True, now=2.0)
    assert core.snapshot() == Snapshot(CardState.EMPTY, None, None, None)


# --- holding a call until the crop is sharp ----------------------------------
#
# The driver measures; the core decides what a "no" means. These pin that a
# held call is not a spent one, and when the core stops waiting.


def blurred():
    return False


def in_focus():
    return True


def _detect(core, crop, now, sharp, settled=True):
    core.on_frame(settled=settled, score=0.0, now=now)
    return core.on_detection(crop, settled=settled, now=now, sharp=sharp)


def test_a_blurred_card_holds_the_call_and_spends_nothing():
    """Held, not fired: no signature committed and no cooldown started, so the
    very next detection that finds the card sharp pays for it — once."""
    core = make_core(idle=0.0, empty=0.0, cooldown=30.0)
    card = det(seed=1)
    assert _detect(core, card, 100.0, blurred) is False
    assert core.holding
    assert not core.identify.in_flight
    assert core.snapshot().state is CardState.SETTLED  # still a card, still there

    assert _detect(core, card, 100.1, in_focus) is True  # a 30s cooldown did not apply
    assert not core.holding
    core.on_identify_done({"full_name": "Clefairy"})
    assert _detect(core, card, 200.0, in_focus) is False  # and it is paid for


def test_nobody_checking_means_nothing_is_held():
    core = make_core(idle=0.0, empty=0.0)
    assert _detect(core, det(seed=1), 100.0, sharp=None) is True


def test_sharpness_is_only_asked_when_a_call_would_otherwise_fire():
    """The answer costs a full-resolution cut, so a card that is moving,
    cooling down or already paid for must never ask the question."""
    asked = []

    def sharp():
        asked.append(1)
        return True

    core = make_core(idle=0.0, empty=0.0, cooldown=5.0)
    assert _detect(core, det(seed=1), 100.0, sharp, settled=False) is False
    assert asked == []  # still moving

    assert _detect(core, det(seed=1), 100.1, sharp) is True
    assert len(asked) == 1
    core.on_identify_done({"full_name": "Clefairy"})

    assert _detect(core, det(seed=2), 101.0, sharp) is False
    assert len(asked) == 1  # a new card, but inside the cooldown
    assert _detect(core, det(seed=1), 110.0, sharp) is False
    assert len(asked) == 1  # the same card: the gate already said no


def test_a_held_call_keeps_the_previous_cards_name():
    """ "Changed" was decided on a blurred crop — exactly when the gate cannot
    be trusted. Dropping the name there would lose it for good whenever the
    card was merely refocusing: the sharp frame matches the signature and
    never fires again."""
    core = make_core(idle=0.0, empty=0.0)
    assert _detect(core, det(seed=1), 100.0, in_focus) is True
    core.on_identify_done({"full_name": "Clefairy"})

    assert _detect(core, det(seed=2), 101.0, blurred) is False
    snap = core.snapshot()
    assert snap.state is CardState.IDENTIFIED
    assert snap.identification == {"full_name": "Clefairy"}

    # The same card after all, now back in focus: no call, and the name stayed.
    assert _detect(core, det(seed=1), 101.2, in_focus) is False
    assert core.snapshot().identification == {"full_name": "Clefairy"}
    assert not core.holding


def test_a_card_held_too_long_is_sent_as_it_is():
    notes = []
    core = make_core(idle=0.0, empty=0.0, log=notes.append, send_blurred_after=3.0)
    card = det(seed=1)
    assert _detect(core, card, 100.0, blurred) is False
    assert _detect(core, card, 101.5, blurred) is False
    assert _detect(core, card, 102.9, blurred) is False
    assert notes == []
    assert _detect(core, card, 103.0, blurred) is True  # 3s of nothing better
    assert any("[sharp]" in n and "sending it anyway" in n for n in notes)
    assert not core.holding


def test_send_blurred_after_zero_waits_for_a_sharp_frame_forever():
    core = make_core(idle=0.0, empty=0.0, send_blurred_after=0.0)
    card = det(seed=1)
    for now in (100.0, 110.0, 1000.0):
        assert _detect(core, card, now, blurred) is False
    assert _detect(core, card, 1000.1, in_focus) is True


def test_a_hold_is_a_streak_and_anything_else_ends_it():
    """The timeout counts CONSECUTIVE held detections. Were it a clock that
    only some exits stopped, a card lifted for three seconds without leaving
    the frame would leave it running — and the next blurred card would be sent
    the instant it settled."""
    core = make_core(idle=0.0, empty=0.0, send_blurred_after=3.0)
    assert _detect(core, det(seed=1), 100.0, blurred) is False
    assert core.holding

    assert _detect(core, det(seed=1), 101.0, blurred, settled=False) is False
    assert not core.holding  # moving again: the streak is over

    assert _detect(core, det(seed=2), 104.0, blurred) is False  # NOT sent at once
    assert _detect(core, det(seed=2), 106.9, blurred) is False
    assert _detect(core, det(seed=2), 107.0, blurred) is True  # 3s of ITS OWN


def test_losing_the_card_ends_the_hold():
    core = make_core(idle=0.0, empty=0.0, send_blurred_after=3.0)
    assert _detect(core, det(seed=1), 100.0, blurred) is False
    assert _detect(core, None, 100.5, blurred) is False
    assert not core.holding
    assert _detect(core, det(seed=1), 103.5, blurred) is False  # a fresh 3s


def test_a_due_retry_that_is_blurred_still_costs_nothing():
    """--retry-unmatched resets the gate BEFORE sharpness is asked. Harmless:
    the emptied gate keeps answering NEW, so the retry simply waits for focus."""
    core = make_core(idle=0.0, empty=0.0, retry_unmatched=5.0)
    card = det(seed=1)
    assert _detect(core, card, 100.0, in_focus) is True
    core.on_identify_done(None)  # no match

    assert _detect(core, card, 105.0, blurred) is False  # retry due, crop blurred
    assert _detect(core, card, 105.2, blurred) is False
    assert _detect(core, card, 105.4, in_focus) is True  # the one retry
    core.on_identify_done({"full_name": "Clefairy"})
    assert _detect(core, card, 111.0, in_focus) is False


def test_a_held_call_does_not_wait_out_the_tracker_resync():
    """Only a detection can retry a held call, and with a tracker locked on
    they come every `tracking` seconds — a long time to leave a card unnamed."""
    core = make_core(moving=0.3, idle=1.0, tracking=30.0, use_tracker=True)
    assert core.on_frame(settled=True, score=0.0, now=100.0) is True
    assert core.on_detection(det(seed=1), True, 100.0, sharp=blurred) is False
    assert core.tracking and core.holding
    assert core.on_frame(settled=True, score=0.0, now=100.5) is False  # idle tier
    assert core.on_frame(settled=True, score=0.0, now=101.0) is True  # not 30s

    # Sent: the tracker's slow tier is back.
    assert core.on_detection(det(seed=1), True, 101.0, sharp=in_focus) is True
    core.on_identify_done({"full_name": "Clefairy"})
    assert core.on_frame(settled=True, score=0.0, now=110.0) is False
    assert core.on_frame(settled=True, score=0.0, now=131.0) is True
