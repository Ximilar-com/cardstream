"""Shared offline test helpers for the whole suite.

Importable from any test directory (``pythonpath = ["tests"]`` in the root
pyproject) so client and core tests use ONE copy of the synthetic frames,
fake clients and fake pipeline components. Everything here is offline: no
network, no torch/onnx, no API key.
"""

from __future__ import annotations

import numpy as np

from cardstream.client.embedders import Embedder
from cardstream.client.identify_target import IdentifyTarget
from cardstream.core.detectors import CardDetector
from cardstream.core.models import (
    BoundingBox,
    ConfidenceTier,
    DetectionResult,
    Identification,
)

# --- Synthetic frames / crops ----------------------------------------------


def make_frame(w: int = 320, h: int = 240, fill: int = 0) -> np.ndarray:
    """A flat BGR frame. Identical frames feed the motion gate to 'settled'."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def textured_crop(seed: int = 0, size: int = 64) -> np.ndarray:
    """A deterministic, high-entropy BGR crop. Different seeds yield crops with a
    large pHash Hamming distance, so they read as distinct 'cards'."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def jpeg_bytes(frame: np.ndarray | None = None) -> bytes:
    """Encode ``frame`` (default: a plain make_frame()) as JPEG bytes."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame if frame is not None else make_frame())
    assert ok
    return buf.tobytes()


def write_mjpg_avi(path, frames: int = 30, w: int = 320, h: int = 240) -> str:
    """Write a looping-friendly MJPG AVI of flat frames; returns the path."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    assert writer.isOpened(), "MJPG writer unavailable in this OpenCV build"
    for _ in range(frames):
        writer.write(make_frame(w=w, h=h))
    writer.release()
    return str(path)


def card_frame(w: int = 640, h: int = 480, angle: float = 0.0) -> np.ndarray:
    """A dark frame with a bright card-shaped rectangle (aspect ~0.71),
    optionally rotated."""
    import cv2

    frame = np.full((h, w, 3), 20, dtype=np.uint8)
    box = cv2.boxPoints(((w / 2, h / 2), (200, 280), angle)).astype(np.int32)
    cv2.fillPoly(frame, [box], (230, 230, 230))
    return frame


async def wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """Poll ``predicate`` until true or ``timeout`` — replaces fixed sleeps."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return bool(predicate())


# --- Results: identification / detection ------------------------------------


def make_identification(
    full_name: str = "Charizard", distance: float = 0.1
) -> Identification:
    return Identification(
        name=full_name,
        full_name=full_name,
        set="Base",
        set_code="BS",
        card_number="4",
        series="Base",
        year="1999",
        subcategory="Pokemon",
        distance=distance,
        confidence_tier=ConfidenceTier.HIGH,
        links={"ximilar": "https://example.com/card"},
        alternatives=[],
    )


def make_detection(crop: np.ndarray, prob: float | None = 0.9) -> DetectionResult:
    """A DetectionResult covering a fixed box."""
    h, w = crop.shape[:2]
    return DetectionResult(bbox=BoundingBox(x=10, y=10, w=w, h=h), crop=crop, prob=prob)


# --- Client: analyzer factory -------------------------------------------------

# Deterministic tuning shared by every SmartAnalyzer test: zero throttles and
# cooldown (detection/identify fire on every eligible frame), quick settle.
DETERMINISTIC_ANALYZER_CFG = {
    # The size filter is frame-relative and the synthetic frames are tiny, so
    # it would reject every fake detection; tests that care set it explicitly.
    # The aspect filter is left ON — the fakes are card-shaped (~0.71), so it
    # stays an honest check that they look like cards.
    "min_card_fraction": 0.0,
    "cooldown_seconds": 0.0,
    "still_frames_required": 2,
    "detect_interval_seconds": 0.0,
    "idle_detect_interval_seconds": 0.0,
    "empty_detect_interval_seconds": 0.0,
}


def make_smart_analyzer(
    detector,
    embedder,
    identify_client,
    on_result=None,
    on_log=None,
    tracker=None,
    store=None,
    **cfg_overrides,
):
    """A SmartAnalyzer with the deterministic test config (identify inline).

    ``cfg_overrides`` are AnalyzerConfig fields and win over the base config.
    """
    from cardstream.client.analyzer import AnalyzerConfig, SmartAnalyzer

    cfg = {**DETERMINISTIC_ANALYZER_CFG, **cfg_overrides}
    return SmartAnalyzer(
        detector=detector,
        embedder=embedder,
        identify_client=identify_client,
        config=AnalyzerConfig(**cfg),
        on_result=on_result,
        on_log=on_log,
        run_async=False,  # identify inline -> deterministic snapshots
        tracker=tracker,
        store=store,
    )


# --- Client: fake pipeline components ---------------------------------------


def unit_vec(axis: int, dim: int = 8) -> np.ndarray:
    """Orthogonal unit vectors — cosine similarity 1.0 to self, 0.0 to others."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


class FakeDetector(CardDetector):
    """Returns the preset Detection (set ``detection = None`` for 'no card')."""

    name = "fake"

    def __init__(self) -> None:
        crop = np.full((70, 50, 3), 128, dtype=np.uint8)
        self.detection: DetectionResult | None = DetectionResult(
            bbox=BoundingBox(x=10, y=10, w=50, h=70), crop=crop
        )
        self.calls = 0
        # (w, h) of every frame detect() was handed — tells the analysis frame
        # apart from the full-resolution one.
        self.frames_seen: list[tuple[int, int]] = []

    def detect(self, frame_bgr):
        self.calls += 1
        h, w = frame_bgr.shape[:2]
        self.frames_seen.append((w, h))
        return self.detection


class FakeTracker:
    """Scripted ObjectTracker: pop the next (ok, bbox) from ``results`` on each
    update (repeating the last one when exhausted); records init calls."""

    name = "fake"

    def __init__(self, results: list | None = None) -> None:
        self.results = list(results or [])
        self.inits: list[BoundingBox] = []
        self.updates = 0

    def init(self, frame_bgr, bbox) -> None:
        self.inits.append(bbox)

    def update(self, frame_bgr):
        self.updates += 1
        if len(self.results) > 1:
            return self.results.pop(0)
        if self.results:
            return self.results[0]
        return True, BoundingBox(x=12, y=12, w=50, h=70)


class FakeEmbedder(Embedder):
    """Returns the preset embedding vector."""

    name = "fake"

    def __init__(self) -> None:
        self.embedding = unit_vec(0)
        self.calls = 0
        # Shapes the identity gate saw — it must keep gating on the cheap
        # analysis-space crop, not the full-resolution one.
        self.crop_shapes: list[tuple[int, int]] = []

    def embed(self, crop_bgr):
        self.calls += 1
        h, w = crop_bgr.shape[:2]
        self.crop_shapes.append((w, h))
        return self.embedding


class FakeIdentifyClient(IdentifyTarget):
    """Stands in for either identify target; counts calls.

    Inherits the real ``options`` handling so the settings endpoint exercises
    the same code path it does in production.
    """

    def __init__(self) -> None:
        super().__init__()
        self.result: dict | None = {
            "full_name": "Charizard",
            "set": "Base",
            "distance": 0.1,
        }
        self.calls = 0
        # The crops actually sent — the only way to prove which frame they were
        # cut from, and that they are owned rather than views on a live frame.
        self.crops: list[np.ndarray] = []

    def identify(self, crop_bgr):
        self.calls += 1
        self.crops.append(crop_bgr)
        return self.result


def start_fake_ws_source(jpeg: bytes, prelude: tuple = ()):
    """A local WS server pushing ``prelude`` messages once, then the same JPEG
    frame forever. Returns ``(url, stop)`` — call ``stop()`` to shut the server
    and its thread down (the ``ws_source`` conftest fixture does this)."""
    import asyncio
    import threading

    import websockets

    holder = {}
    ready = threading.Event()

    def run() -> None:
        async def handler(ws):
            try:
                for msg in prelude:
                    await ws.send(msg)
                while True:
                    await ws.send(jpeg)
                    await asyncio.sleep(0.02)
            except Exception:
                pass

        async def amain():
            stop_event = asyncio.Event()
            holder["stop_event"] = stop_event
            holder["loop"] = asyncio.get_running_loop()
            server = await websockets.serve(handler, "127.0.0.1", 0)
            holder["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await stop_event.wait()
            server.close()
            await server.wait_closed()

        asyncio.run(amain())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5), "fake ws source did not start"

    def stop() -> None:
        holder["loop"].call_soon_threadsafe(holder["stop_event"].set)
        thread.join(timeout=5)

    return f"ws://127.0.0.1:{holder['port']}", stop
