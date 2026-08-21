"""VitTracker wrapper + make_tracker factory — offline, cv2 tracker faked
(the same pattern as the fake-ONNX-session detector tests)."""

from __future__ import annotations

import cv2
import pytest

from _helpers import make_frame
from cardstream.core.models import BoundingBox
from cardstream.core.tracking import VitTracker, make_tracker


class _FakeCvTracker:
    """Stands in for the cv2.TrackerVit instance."""

    def __init__(self) -> None:
        self.inited_with = None
        self.result = (True, (10, 20, 50, 70))
        self.score = 0.9

    def init(self, frame, box) -> None:
        self.inited_with = box

    def update(self, frame):
        return self.result

    def getTrackingScore(self):
        return self.score


@pytest.fixture()
def fake_cv(monkeypatch):
    fake = _FakeCvTracker()
    captured = {}

    def create(params):
        captured["net"] = params.net
        return fake

    monkeypatch.setattr(cv2, "TrackerVit_create", create)
    return fake, captured


def test_make_tracker_without_model_is_disabled():
    assert make_tracker(None) is None
    assert make_tracker("") is None


def test_vit_tracker_loads_model_and_inits_with_xywh(fake_cv):
    fake, captured = fake_cv
    t = make_tracker("model.onnx")
    assert captured["net"] == "model.onnx"
    t.init(make_frame(), BoundingBox(5, 6, 70, 90))
    assert fake.inited_with == (5, 6, 70, 90)


def test_update_locked_returns_bbox(fake_cv):
    _fake, _ = fake_cv
    t = VitTracker("model.onnx", score_threshold=0.3)
    ok, bbox = t.update(make_frame())
    assert ok is True
    assert bbox == BoundingBox(10, 20, 50, 70)


def test_update_low_score_reads_as_lost(fake_cv):
    fake, _ = fake_cv
    t = VitTracker("model.onnx", score_threshold=0.3)
    fake.score = 0.1
    assert t.update(make_frame()) == (False, None)


def test_update_failed_or_degenerate_box_reads_as_lost(fake_cv):
    fake, _ = fake_cv
    t = VitTracker("model.onnx")
    fake.result = (False, (0, 0, 0, 0))
    assert t.update(make_frame()) == (False, None)
    fake.result = (True, (10, 10, 0, 5))  # zero-width box
    assert t.update(make_frame()) == (False, None)


def test_missing_cv2_support_raises_clear_error(monkeypatch):
    monkeypatch.delattr(cv2, "TrackerVit_create", raising=False)
    with pytest.raises(RuntimeError, match="opencv"):
        VitTracker("model.onnx")
