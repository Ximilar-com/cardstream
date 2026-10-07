"""cardstream-client behaviour that is not about the pipeline it builds."""

from __future__ import annotations

import sys

import cv2

from _helpers import FakeDetector, FakeIdentifyClient, printed_card
from cardstream.client import stream_client
from cardstream.client.analyzer import AnalyzerConfig
from cardstream.client.common import Pipeline


def test_a_still_image_is_never_held_for_a_sharper_frame(monkeypatch, tmp_path):
    """A still cannot come into focus. Holding it would cost the default
    timeout before its one answer — and with --send-blurred-after 0 it would
    exit ten seconds later having asked nothing at all."""
    soft = cv2.GaussianBlur(printed_card(), (0, 0), 8)
    still = tmp_path / "soft.png"
    cv2.imwrite(str(still), soft)

    card_box = FakeDetector()  # reports its 50x70 box, well inside the still
    identify = FakeIdentifyClient()
    pipeline = Pipeline(
        detector=card_box,
        embedder=None,
        identify_client=identify,
        # The shipped sharpness settings, and nothing else in the way of a
        # six-frame run: the tiny fake box, the throttles, the cooldown.
        config=AnalyzerConfig(
            gate="phash",
            min_card_fraction=0.0,
            cooldown_seconds=0.0,
            detect_interval_seconds=0.0,
            idle_detect_interval_seconds=0.0,
            empty_detect_interval_seconds=0.0,
        ),
        description="fake",
    )
    assert pipeline.config.min_sharpness > 0  # the hold IS on for a live source
    monkeypatch.setattr(stream_client, "build_pipeline", lambda args: pipeline)
    monkeypatch.setattr(stream_client, "print_banner", lambda *a, **k: None)

    def six_frames(analyzer, frame, fps, loop, print_state):
        for _ in range(6):
            analyzer.process(frame.copy())

    monkeypatch.setattr(stream_client, "_run_still_image", six_frames)
    monkeypatch.setattr(sys, "argv", ["cardstream-client", "--source", str(still)])
    stream_client.main()
    assert identify.calls == 1
