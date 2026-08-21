"""Detector backends: factory routing, crop margins, and the ONNX
post-processing numerics (via a fake session — no onnxruntime, no weights)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cardstream.core import detectors as detectors_mod
from cardstream.core.detectors import (
    RfDetrOnnxDetector,
    RfDetrSegOnnxDetector,
    RtDetrOnnxDetector,
    _crop_with_margin,
    make_detector,
    make_segmentor,
)

# --- factory -----------------------------------------------------------------


def test_make_detector_names_the_flag_when_a_model_is_missing():
    """Every family needs a model and none has a usable default here, so the
    error has to name the flag that fixes it."""
    for family in ("rfdetr", "rtdetr"):
        with pytest.raises(ValueError, match="--detector-model"):
            make_detector(family, model=None)


def test_make_detector_rejects_unknown():
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("magic")


def test_make_detector_rtdetr_requires_model():
    with pytest.raises(ValueError, match="needs a model path"):
        make_detector("rtdetr")


def test_make_detector_rtdetr_onnx_routing(monkeypatch):
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (
            _FakeRtDetrSession(np.zeros((1, 1)), np.zeros((1, 4))),
            "pixel_values",
            64,
        ),
    )
    det = make_detector("rtdetr", model="card.onnx")
    assert isinstance(det, RtDetrOnnxDetector)


def test_make_detector_rfdetr_onnx_routing(monkeypatch):
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (
            _FakeRfDetrSession(np.zeros((1, 4)), np.zeros((1, 2))),
            "input",
            64,
        ),
    )
    det = make_detector("rfdetr", model="card.onnx")
    assert isinstance(det, RfDetrOnnxDetector)


def test_make_detector_rfdetr_rejects_non_onnx():
    """rfdetr has no transformers fallback — the error points at the exporter."""
    with pytest.raises(ValueError, match="export_onnx"):
        make_detector("rfdetr", model="models/card-rfdetr-small/final")


# --- _crop_with_margin -------------------------------------------------------


def test_crop_with_margin_pads_and_clamps():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _crop_with_margin(frame, 10, 10, 60, 80)
    assert det is not None
    x, y, w, h = det.bbox.as_list()
    # 4% margin of a 50x70 box = 2x2 px, clamped inside the frame
    assert (x, y) == (8, 8)
    assert (w, h) == (54, 74)
    assert det.crop.shape == (74, 54, 3)


def test_crop_with_margin_clamps_at_frame_edges():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    det = _crop_with_margin(frame, -5, -5, 60, 60)
    assert det is not None
    assert det.bbox.as_list() == [0, 0, 50, 50]


def test_crop_with_margin_rejects_degenerate_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _crop_with_margin(frame, 10, 10, 12, 12) is None


# --- RtDetrOnnxDetector numerics (fake session) -------------------------------


class _FakeRtDetrSession:
    """RT-DETR export shape: run() returns [logits (1,Q,C), boxes (1,Q,4)]."""

    def __init__(self, logits: np.ndarray, boxes: np.ndarray):
        self._logits = logits.astype(np.float32)
        self._boxes = boxes.astype(np.float32)
        self.received = None

    def run(self, _outputs, feeds):
        self.received = feeds
        return [self._logits[None], self._boxes[None]]


def _rtdetr_detector(
    monkeypatch, logits, boxes, size=64, **kwargs
) -> RtDetrOnnxDetector:
    session = _FakeRtDetrSession(np.array(logits), np.array(boxes))
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "pixel_values", size),
    )
    det = RtDetrOnnxDetector("fake.onnx", **kwargs)
    det._fake_session = session
    return det


def test_rtdetr_decodes_normalized_box_on_nonsquare_frame(monkeypatch):
    # Frame 128 wide x 64 high; the model resizes without letterboxing, so
    # normalized cxcywh denormalizes by the ORIGINAL frame dimensions.
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    logits = [[4.0], [-10.0], [-10.0]]  # query 0 wins, sigmoid(4)=0.982
    boxes = [[0.5, 0.5, 0.25, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
    det = _rtdetr_detector(monkeypatch, logits, boxes, conf=0.5).detect(frame)
    assert det is not None
    assert det.prob == pytest.approx(1 / (1 + np.exp(-4.0)))
    # cxcywh (0.5,0.5,0.25,0.5) -> corners (48,16)-(80,48), then the 4% margin
    assert det.bbox.as_list() == [47, 15, 34, 34]
    assert det.crop.shape == (34, 34, 3)


def test_rtdetr_feeds_resized_normalized_blob(monkeypatch):
    frame = np.full((32, 96, 3), 255, dtype=np.uint8)
    det = _rtdetr_detector(
        monkeypatch, [[4.0]], [[0.5, 0.5, 0.5, 0.5]], size=64, conf=0.5
    )
    det.detect(frame)
    blob = det._fake_session.received["pixel_values"]
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert blob.max() == pytest.approx(1.0)


def test_rtdetr_below_threshold_returns_none(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # logit 0 -> sigmoid 0.5, below the 0.6 threshold
    det = _rtdetr_detector(monkeypatch, [[0.0]], [[0.5, 0.5, 0.5, 0.5]], conf=0.6)
    assert det.detect(frame) is None


def test_rtdetr_class_filter(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    logits = [[-10.0, 4.0]]  # only class 1 is confident
    boxes = [[0.5, 0.5, 0.5, 0.5]]
    assert (
        _rtdetr_detector(monkeypatch, logits, boxes, conf=0.5, class_ids=(0,)).detect(
            frame
        )
        is None
    )
    det = _rtdetr_detector(monkeypatch, logits, boxes, conf=0.5, class_ids=(1,)).detect(
        frame
    )
    assert det is not None


# --- RfDetrOnnxDetector numerics (fake session) --------------------------------


class _FakeRfDetrSession:
    """RF-DETR export shape: NAMED outputs dets/labels(/masks), order varies."""

    def __init__(self, dets, labels, extra=None, order=("dets", "labels")):
        self._outs = {
            "dets": np.asarray(dets, np.float32)[None],
            "labels": np.asarray(labels, np.float32)[None],
        }
        for name, value in (extra or {}).items():
            self._outs[name] = np.asarray(value, np.float32)[None]
        self._order = order
        self.received = None

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self._order]

    def run(self, _outputs, feeds):
        self.received = feeds
        return [self._outs[n] for n in self._order]


def _rfdetr_detector(
    monkeypatch, dets, labels, size=64, extra=None, order=("dets", "labels"), **kwargs
) -> RfDetrOnnxDetector:
    session = _FakeRfDetrSession(dets, labels, extra=extra, order=order)
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "input", size),
    )
    det = RfDetrOnnxDetector("fake.onnx", **kwargs)
    det._fake_session = session
    return det


def test_rfdetr_decodes_normalized_box_on_nonsquare_frame(monkeypatch):
    # Same geometry as the rtdetr test, but the labels carry the extra
    # no-object column that must not take part in scoring.
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    labels = [[4.0, -10.0], [-10.0, -10.0], [-10.0, -10.0]]  # (Q, C+1), C=1
    dets = [[0.5, 0.5, 0.25, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
    det = _rfdetr_detector(monkeypatch, dets, labels, conf=0.5).detect(frame)
    assert det is not None
    assert det.prob == pytest.approx(1 / (1 + np.exp(-4.0)))
    # cxcywh (0.5,0.5,0.25,0.5) -> corners (48,16)-(80,48), then the 4% margin
    assert det.bbox.as_list() == [47, 15, 34, 34]
    assert det.crop.shape == (34, 34, 3)


def test_rfdetr_drops_no_object_column(monkeypatch):
    # Only the LAST (no-object) logit is hot: with it dropped, nothing scores.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det = _rfdetr_detector(
        monkeypatch, [[0.5, 0.5, 0.5, 0.5]], [[-10.0, 6.0]], conf=0.5
    )
    assert det.detect(frame) is None


def test_rfdetr_matches_outputs_by_name(monkeypatch):
    # Outputs handed over in reversed order must still decode correctly.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det = _rfdetr_detector(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[4.0, -10.0]],
        order=("labels", "dets"),
        conf=0.5,
    ).detect(frame)
    assert det is not None
    assert det.prob == pytest.approx(1 / (1 + np.exp(-4.0)))


def test_rfdetr_ignores_masks_output(monkeypatch):
    # A segmentation export carries a third `masks` output; detection decoding
    # must not care.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    det = _rfdetr_detector(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[4.0, -10.0]],
        extra={"masks": np.zeros((1, 8, 8))},
        order=("dets", "labels", "masks"),
        conf=0.5,
    ).detect(frame)
    assert det is not None


def test_rfdetr_feeds_imagenet_normalized_blob(monkeypatch):
    frame = np.full((32, 96, 3), 255, dtype=np.uint8)
    det = _rfdetr_detector(
        monkeypatch, [[0.5, 0.5, 0.5, 0.5]], [[4.0, -10.0]], size=64, conf=0.5
    )
    det.detect(frame)
    blob = det._fake_session.received["input"]
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    # A white frame normalizes to (1 - mean) / std per channel.
    assert blob[0, 0].max() == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-4)
    assert blob[0, 2].max() == pytest.approx((1.0 - 0.406) / 0.225, rel=1e-4)


def test_rfdetr_below_threshold_returns_none(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    # logit 0 -> sigmoid 0.5, below the 0.6 threshold
    det = _rfdetr_detector(
        monkeypatch, [[0.5, 0.5, 0.5, 0.5]], [[0.0, -10.0]], conf=0.6
    )
    assert det.detect(frame) is None


def test_rfdetr_class_filter(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = [[-10.0, 4.0, -10.0]]  # C=2 + no-object; class 1 confident
    dets = [[0.5, 0.5, 0.5, 0.5]]
    assert (
        _rfdetr_detector(monkeypatch, dets, labels, conf=0.5, class_ids=(0,)).detect(
            frame
        )
        is None
    )
    det = _rfdetr_detector(monkeypatch, dets, labels, conf=0.5, class_ids=(1,)).detect(
        frame
    )
    assert det is not None


# --- RfDetrSegOnnxDetector: masks -> corners -----------------------------------


def _mask_plane(size, centre, rect, angle) -> np.ndarray:
    """One query's mask logits: +10 inside a rotated rect, -10 outside.

    Logits rather than a 0/1 mask because that is what the export emits, and
    the threshold is at logit 0 (== sigmoid 0.5).
    """
    import cv2

    plane = np.full((size, size), -10.0, np.float32)
    box = cv2.boxPoints((centre, rect, angle)).astype(np.int32)
    cv2.fillPoly(plane, [box], 10.0)
    return plane


def _rfdetr_segmentor(
    monkeypatch,
    dets,
    labels,
    masks,
    size=64,
    order=("dets", "labels", "masks"),
    **kwargs,
) -> RfDetrSegOnnxDetector:
    session = _FakeRfDetrSession(dets, labels, extra={"masks": masks}, order=order)
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "input", size),
    )
    seg = RfDetrSegOnnxDetector("fake.onnx", **kwargs)
    seg._fake_session = session
    return seg


def test_a_detection_export_is_refused_as_a_segmentor(monkeypatch):
    """It would load and run perfectly well — it just has no boundary to give,
    so this has to fail at startup rather than silently every frame."""
    session = _FakeRfDetrSession([[0.5, 0.5, 0.5, 0.5]], [[4.0, -10.0]])
    monkeypatch.setattr(
        detectors_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "input", 64),
    )
    with pytest.raises(ValueError, match="detection export"):
        RfDetrSegOnnxDetector("boxes.onnx")


def test_rfdetr_seg_derives_a_quad_from_the_winning_querys_mask(monkeypatch):
    """Masks are indexed by QUERY, so the loser's mask must not leak in."""
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    labels = [[-10.0, -10.0], [4.0, -10.0]]  # query 1 wins
    dets = [[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.3, 0.6]]
    masks = np.stack(
        [
            _mask_plane(32, (16, 16), (30, 30), 0),  # loser: near-full frame
            _mask_plane(32, (16, 16), (10, 20), 0),  # winner: a portrait card
        ]
    )
    det = _rfdetr_segmentor(monkeypatch, dets, labels, masks, conf=0.5).detect(frame)
    assert det is not None and det.quad is not None
    # The winner's 10x20-of-32 rect scales to 40x80 of the 128 px frame.
    assert det.bbox.w == pytest.approx(40, abs=4)
    assert det.bbox.h == pytest.approx(80, abs=4)


def test_rfdetr_seg_returns_a_deskewed_crop_for_the_gate(monkeypatch):
    """det.crop is what the identity gate compares (engine.decide), so under a
    segmentor it is the card square-on — the embedding then stops moving as the
    card tilts in the hand."""
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    masks = _mask_plane(32, (16, 16), (10, 20), 30)[None]  # a TILTED card
    det = _rfdetr_segmentor(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[4.0, -10.0]],
        masks,
        conf=0.5,
    ).detect(frame)
    assert det is not None
    ch, cw = det.crop.shape[:2]
    # Deskewed to the card's own 10x20-of-32 -> 40x80, NOT the inflated hull
    # an axis-aligned cut of a 30-degree tilt would give.
    assert (cw, ch) == pytest.approx((40, 80), abs=5)
    assert det.bbox.w > cw and det.bbox.h > ch


def test_rfdetr_seg_quad_is_in_analysis_frame_pixels(monkeypatch):
    """The quad shares bbox's coordinate space — FramePair.warp scales it up."""
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    masks = _mask_plane(32, (16, 16), (10, 20), 0)[None]
    det = _rfdetr_segmentor(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[4.0, -10.0]],
        masks,
        conf=0.5,
    ).detect(frame)
    assert det.quad.shape == (4, 2)
    assert det.quad[:, 0].min() >= 0 and det.quad[:, 0].max() <= 128
    from cardstream.core.quad import quad_bbox

    assert quad_bbox(det.quad).as_list() == det.bbox.as_list()


def test_rfdetr_seg_falls_back_to_its_own_box_when_the_mask_is_empty(monkeypatch):
    """A mask that yields no shape is not worth losing the card over — degrade
    to detector behaviour (quad=None) rather than dropping the detection."""
    frame = np.zeros((64, 128, 3), dtype=np.uint8)
    masks = np.full((1, 32, 32), -10.0, np.float32)  # nothing above threshold
    det = _rfdetr_segmentor(
        monkeypatch,
        [[0.5, 0.5, 0.25, 0.5]],
        [[4.0, -10.0]],
        masks,
        conf=0.5,
    ).detect(frame)
    assert det is not None
    assert det.quad is None
    assert det.bbox.as_list() == [47, 15, 34, 34]  # the box path, margin and all


def test_rfdetr_seg_below_threshold_returns_none(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    masks = _mask_plane(32, (16, 16), (10, 20), 0)[None]
    det = _rfdetr_segmentor(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[0.0, -10.0]],
        masks,
        conf=0.6,
    )
    assert det.detect(frame) is None


def test_rfdetr_seg_matches_outputs_by_name(monkeypatch):
    """Same name-matching as the detector: export order must not matter."""
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    masks = _mask_plane(32, (16, 16), (10, 20), 0)[None]
    det = _rfdetr_segmentor(
        monkeypatch,
        [[0.5, 0.5, 0.5, 0.5]],
        [[4.0, -10.0]],
        masks,
        order=("masks", "labels", "dets"),
        conf=0.5,
    ).detect(frame)
    assert det is not None and det.quad is not None


def test_make_segmentor_names_the_flag_when_a_model_is_missing():
    for bad in (None, "", "model.pth"):
        with pytest.raises(ValueError, match="--segmentor-model"):
            make_segmentor("rfdetr", model=bad)


def test_make_segmentor_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown segmentor"):
        make_segmentor("rtdetr", model="m.onnx")


# --- the export contract is matched by NAME, and says so when it cannot -------


def test_rfdetr_raises_when_an_output_name_is_missing(monkeypatch):
    """No positional fallback: guessing would read logits as boxes and locate
    a card nowhere near the card, silently, on every frame."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = [[4.0, -10.0]]
    dets = [[0.5, 0.5, 0.25, 0.5]]
    # An export that named its outputs the RT-DETR way instead.
    det = _rfdetr_detector(
        monkeypatch,
        dets,
        labels,
        order=("boxes", "logits"),
        extra={"boxes": dets, "logits": labels},
    )
    with pytest.raises(ValueError, match="not an RF-DETR export"):
        det.detect(frame)


def test_rfdetr_class_filter_ignores_ids_the_model_has_no_column_for(monkeypatch):
    """--detector-classes 5 against a single-class model used to IndexError."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    labels = [[4.0, -10.0]]  # (Q, C+1), C=1
    dets = [[0.5, 0.5, 0.25, 0.5]]
    det = _rfdetr_detector(monkeypatch, dets, labels, conf=0.5, class_ids=(5,))
    assert det.detect(frame) is None  # nothing kept, but no crash
