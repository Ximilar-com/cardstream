"""Both entrypoints close the --ximilar-stream session when the run ends."""

from __future__ import annotations

import sys
import types

import pytest

from _helpers import (
    SESSION_ID,
    FakeDetector,
    FakeIdentifyClient,
    FakeSessionApi,
    make_frame,
)
from cardstream.client import banner, common, sources, stream_client
from cardstream.client.analyzer import AnalyzerConfig
from cardstream.client.common import Pipeline
from cardstream.core.ximilar_session import SessionRecorder


@pytest.fixture
def session_pipeline(monkeypatch):
    """A pipeline of fakes whose session records to a FakeSessionApi."""
    api = FakeSessionApi()
    recorder = SessionRecorder(api, SESSION_ID, start=False, log=lambda _: None)
    pipeline = Pipeline(
        detector=FakeDetector(),
        embedder=None,
        identify_client=FakeIdentifyClient(),
        config=AnalyzerConfig(gate="phash"),
        description="fake",
        recorder=recorder,
    )
    monkeypatch.setattr(banner, "print_banner", lambda *a, **k: None)
    monkeypatch.setattr(stream_client, "print_banner", lambda *a, **k: None)
    return pipeline, api


def test_ctrl_c_in_the_headless_client_closes_the_session(
    session_pipeline, monkeypatch, tmp_path
):
    import cv2

    pipeline, api = session_pipeline
    still = tmp_path / "card.png"
    cv2.imwrite(str(still), make_frame())
    monkeypatch.setattr(stream_client, "build_pipeline", lambda args: pipeline)

    def interrupted(analyzer, *args):
        pipeline.recorder.record({"full_name": "Charizard"}, "tcg")
        raise KeyboardInterrupt

    monkeypatch.setattr(stream_client, "_run_still_image", interrupted)
    monkeypatch.setattr(sys, "argv", ["cardstream-client", "--source", str(still)])
    stream_client.main()
    assert [i["full_name"] for i in api.uploads[0]] == ["Charizard"]
    assert api.closed == [SESSION_ID]


def test_stopping_the_web_server_closes_the_session(session_pipeline, monkeypatch):
    from cardstream.client import web_client

    pipeline, api = session_pipeline
    monkeypatch.setattr(common, "build_pipeline", lambda args: pipeline)
    monkeypatch.setattr(sources, "make_source", lambda *a, **k: None)
    # uvicorn.run returns once Ctrl-C has shut the server down.
    ran = []
    fake_uvicorn = types.SimpleNamespace(run=lambda app, **kwargs: ran.append(app))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(sys, "argv", ["cardstream-web", "--no-browser"])
    web_client.main()
    assert ran, "the app was never served"
    assert api.closed == [SESSION_ID]
