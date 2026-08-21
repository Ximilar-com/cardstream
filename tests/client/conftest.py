"""Client-suite fixtures. Helpers live in tests/_helpers.py (one shared copy)."""

from __future__ import annotations

import pytest

from _helpers import (
    FakeDetector,
    FakeEmbedder,
    FakeIdentifyClient,
    start_fake_ws_source,
)


@pytest.fixture
def ws_source():
    """Factory for fake WS frame sources; every started server is stopped at
    teardown so no thread or event loop outlives the test."""
    stops = []

    def factory(jpeg: bytes, prelude: tuple = ()) -> str:
        url, stop = start_fake_ws_source(jpeg, prelude)
        stops.append(stop)
        return url

    yield factory
    for stop in stops:
        stop()


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_identify() -> FakeIdentifyClient:
    return FakeIdentifyClient()
