"""Browser-UI transport: frames over /ws -> local analyzer -> JSON snapshots.

Skipped automatically when fastapi isn't installed (it's in the base
requirements but optional for the headless CLI path).
"""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from _helpers import (
    jpeg_bytes,
    make_frame,
    make_smart_analyzer,
    write_mjpg_avi,
)
from cardstream.client.sources import CaptureSource, WsJpegSource
from cardstream.client.web_client import create_web_app
from cardstream.core.identify_options import IdentifyOptions


def _analyzer_factory(fake_detector, fake_embedder, fake_identify, **cfg):
    """The ``make_analyzer(on_result, on_log)`` callable create_web_app wants."""

    def make_analyzer(on_result, on_log):
        return make_smart_analyzer(
            fake_detector,
            fake_embedder,
            fake_identify,
            on_result=on_result,
            on_log=on_log,
            **cfg,
        )

    return make_analyzer


def _make_client(fake_detector, fake_embedder, fake_identify, debug=False):
    factory = _analyzer_factory(
        fake_detector, fake_embedder, fake_identify, debug=debug
    )
    return TestClient(create_web_app(factory, debug=debug))


def _drain_until_frame_and_identified(client, attempts):
    """Read the viewer WS until both a relayed frame and an identified
    snapshot arrived (bounded by ``attempts`` mixed binary/JSON messages)."""
    got_frame = got_identified = False
    with client.websocket_connect("/ws") as ws:
        for _ in range(attempts):
            raw = ws.receive()
            if raw.get("bytes"):
                got_frame = True
            elif raw.get("text"):
                msg = json.loads(raw["text"])
                if msg.get("state") == "identified" and msg.get("identification"):
                    got_identified = True
            if got_frame and got_identified:
                break
    return got_frame, got_identified


@pytest.fixture
def web_client(fake_detector, fake_embedder, fake_identify):
    return _make_client(fake_detector, fake_embedder, fake_identify)


def test_serves_the_page(web_client):
    r = web_client.get("/")  # redirects to /smart/
    assert r.status_code == 200
    assert "Smart Card Client" in r.text
    assert web_client.get("/smart/app.js").status_code == 200
    assert web_client.get("/smart/smart.css").status_code == 200
    # the shared modules the page imports must be reachable too
    assert web_client.get("/shared/overlay.js").status_code == 200
    assert web_client.get("/shared/style.css").status_code == 200


def test_ws_returns_snapshots_and_identifies_once(web_client, fake_identify):
    valid = {"empty", "moving", "settled", "identifying", "identified"}
    states = []
    with web_client.websocket_connect("/ws") as ws:
        for _ in range(6):
            ws.send_bytes(jpeg_bytes())
            msg = ws.receive_json()
            assert msg["state"] in valid
            states.append(msg["state"])
    assert fake_identify.calls == 1  # one paid call for one held card
    assert states[-1] == "identified"
    assert "bbox" in msg and msg["bbox"] == [10, 10, 50, 70]
    assert msg["identification"]["full_name"] == "Charizard"


def test_ws_ignores_text_and_bad_bytes(web_client):
    with web_client.websocket_connect("/ws") as ws:
        ws.send_text("hello")  # ignored
        ws.send_bytes(b"not-a-jpeg")  # undecodable -> skipped
        ws.send_bytes(jpeg_bytes())
        msg = ws.receive_json()
        assert msg["state"] in {
            "empty",
            "moving",
            "settled",
            "identifying",
            "identified",
        }


def test_ws_pushes_log_frames_in_debug_mode(
    fake_detector, fake_embedder, fake_identify
):
    client = _make_client(fake_detector, fake_embedder, fake_identify, debug=True)
    logs = []
    with client.websocket_connect("/ws") as ws:
        # Log frames are scheduled onto the loop and interleave with the
        # per-frame snapshots — collect everything over a fixed frame budget.
        for _ in range(8):
            ws.send_bytes(jpeg_bytes())
            msg = ws.receive_json()
            while "log" in msg:
                logs.append(msg["log"])
                msg = ws.receive_json()
    assert any("[detect] card found" in line for line in logs)
    assert any("[identify] new card" in line for line in logs)
    assert any("[gate] cos_sim" in line for line in logs)


def test_ws_no_log_frames_without_debug(web_client):
    with web_client.websocket_connect("/ws") as ws:
        for _ in range(8):
            ws.send_bytes(jpeg_bytes())
            msg = ws.receive_json()
            assert "log" not in msg


# --- analysis frame vs full-resolution identify crop --------------------------


def test_ws_snapshots_carry_the_analysed_dimensions(
    fake_detector, fake_embedder, fake_identify
):
    """The page sends full-res frames now, so bboxes (analysis space) are no
    longer in the space of what it sent — it needs these dims to map them."""
    factory = _analyzer_factory(
        fake_detector, fake_embedder, fake_identify, analysis_width=480
    )
    client = TestClient(create_web_app(factory))
    big = jpeg_bytes(make_frame(w=1920, h=1080, fill=40))
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(big)
        assert ws.receive_json()["analysis"] == [480, 270]
    assert fake_detector.frames_seen[0] == (480, 270)


def test_ws_identify_crop_comes_from_the_full_frame(
    fake_detector, fake_embedder, fake_identify
):
    factory = _analyzer_factory(
        fake_detector, fake_embedder, fake_identify, analysis_width=480
    )
    client = TestClient(create_web_app(factory))
    big = jpeg_bytes(make_frame(w=1920, h=1080, fill=40))
    with client.websocket_connect("/ws") as ws:
        for _ in range(8):
            ws.send_bytes(big)
            ws.receive_json()
    assert fake_identify.calls == 1
    assert fake_identify.crops[0].shape[:2] == (70 * 4, 50 * 4)


@pytest.mark.asyncio
async def test_latest_frame_mailbox_skips_stale_frames():
    """Analysis is synchronous on the event loop, so a queued frame is always
    staler than the one behind it. The mailbox hands over the freshest and
    counts what it skipped. (Driven directly: TestClient's WebSocket is
    lock-step, so a backlog can never build up through it.)"""
    from cardstream.client.web_camera import LatestFrame

    mailbox = LatestFrame()
    for i in range(20):
        mailbox.put(f"frame-{i}".encode())
    assert await mailbox.take() == b"frame-19"  # newest wins
    assert mailbox.skipped == 19  # ...the rest never analysed

    # A producer running while the consumer is busy: the skip-ahead loop yields,
    # letting it deliver, and only the last of that burst is analysed.
    mailbox = LatestFrame()
    mailbox.put(b"first")

    async def producer():
        for i in range(3):
            mailbox.put(f"late-{i}".encode())  # queued, then hand back control
            await asyncio.sleep(0)

    task = asyncio.create_task(producer())
    assert await mailbox.take() == b"late-2"
    await task

    # Closing wakes the consumer with nothing to do.
    mailbox = LatestFrame()
    mailbox.close()
    assert await mailbox.take() is None
    assert mailbox.done


# --- identify-call badge (counter rides on every WS payload) ------------------


def test_ws_snapshots_carry_the_identify_call_count(
    fake_detector, fake_embedder, fake_identify
):
    client = _make_client(fake_detector, fake_embedder, fake_identify)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(jpeg_bytes())
        assert ws.receive_json()["identify_calls"] == 0  # nothing identified yet

        seen = 0
        for _ in range(12):  # settle -> identify
            ws.send_bytes(jpeg_bytes())
            seen = max(seen, ws.receive_json()["identify_calls"])
        assert seen >= 1

    # A new connection is a new session: its own analyzer, counting from zero.
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(jpeg_bytes())
        assert ws.receive_json()["identify_calls"] == 0


# --- settings dialog (GET/POST /settings -> shared identify client) -----------


def test_settings_endpoints_read_and_write_the_identify_client(
    fake_detector, fake_embedder, fake_identify
):
    fake_identify.options = IdentifyOptions()  # a known starting point
    factory = _analyzer_factory(fake_detector, fake_embedder, fake_identify)
    client = TestClient(create_web_app(factory, identify_client=fake_identify))

    s = client.get("/settings").json()
    assert s["enabled"] is True
    assert s["category"] == "tcg"
    assert s["game"] == "Not Specified" and s["set_code"] == ""
    assert s["known_attrs"] is True
    # Unset by default: the record carries no Alphabet at all.
    assert s["alphabet"] == "Not Specified" and s["alphabets"][0] == "Not Specified"
    assert "japanese" in s["alphabets"]
    assert s["games"][0] == "Not Specified" and "Pokémon" in s["games"]
    assert {c["id"] for c in s["categories"]} == {"tcg", "sport", "slab", "comics"}

    s = client.post("/settings", json={"game": "Pokémon", "set_code": " PBL "}).json()
    assert s["game"] == "Pokémon" and s["set_code"] == "PBL"
    assert fake_identify.options.game == "Pokémon"
    assert fake_identify.options.set_code == "PBL"

    s = client.post("/settings", json={"known_attrs": False}).json()
    assert s["known_attrs"] is False and fake_identify.options.known_attrs is False

    # A game prefill kills the endpoint's alphabet detection, so the pair has
    # to be settable together in one Save.
    s = client.post(
        "/settings", json={"game": "Pokémon", "alphabet": "japanese"}
    ).json()
    assert s["alphabet"] == "japanese" and fake_identify.options.alphabet == "japanese"
    assert client.post("/settings", json={"alphabet": "klingon"}).status_code == 400
    assert fake_identify.options.alphabet == "japanese"  # unchanged on bad input

    # ...and back to sending nothing at all.
    s = client.post("/settings", json={"alphabet": "Not Specified"}).json()
    assert s["alphabet"] == "Not Specified" and fake_identify.options.alphabet is None
    assert (
        client.post("/settings", json={"alphabet": None}).json()["alphabet"]
        == "Not Specified"
    )

    assert client.post("/settings", json={"game": "Uno"}).status_code == 400
    assert fake_identify.options.game == "Pokémon"  # unchanged on bad input

    s = client.post("/settings", json={"game": "Not Specified"}).json()
    assert s["game"] == "Not Specified" and fake_identify.options.game is None


def test_settings_category_switch_moves_the_endpoint_and_games(
    fake_detector, fake_embedder, fake_identify
):
    """A category switch is an endpoint switch — and the game list follows it."""
    from cardstream.client.ximilar_api import DirectXimilarClient
    from cardstream.core.identify_options import IdentifyOptions

    identify = DirectXimilarClient("key", IdentifyOptions(game="Pokémon"))
    factory = _analyzer_factory(fake_detector, fake_embedder, fake_identify)
    client = TestClient(create_web_app(factory, identify_client=identify))

    s = client.post("/settings", json={"category": "sport"}).json()
    assert s["category"] == "sport" and identify.options.id_type.key == "sport"
    assert "Baseball" in s["games"] and "Pokémon" not in s["games"]
    assert s["game"] == "Not Specified"  # Pokémon means nothing to sport_id
    assert identify.options.game is None

    s = client.post("/settings", json={"game": "Baseball"}).json()
    assert s["game"] == "Baseball" and identify.options.game == "Baseball"

    assert client.post("/settings", json={"category": "nope"}).status_code == 400
    assert identify.options.id_type.key == "sport"


def test_settings_result_threshold_retunes_live_analyzers(
    fake_detector, fake_embedder, fake_identify
):
    made = []

    def factory(on_result, on_log):
        analyzer = make_smart_analyzer(
            fake_detector,
            fake_embedder,
            fake_identify,
            on_result=on_result,
            on_log=on_log,
        )
        made.append(analyzer)
        return analyzer

    client = TestClient(
        create_web_app(factory, identify_client=fake_identify, result_threshold=0.8)
    )
    assert client.get("/settings").json()["result_threshold"] == 0.8

    with client.websocket_connect("/ws"):
        assert len(made) == 1 and made[0].result_threshold == 0.8
        s = client.post("/settings", json={"result_threshold": 0.35}).json()
        assert s["result_threshold"] == 0.35
        assert made[0].result_threshold == 0.35  # the running analyzer, live

    # …and the new value is the default for analyzers created later.
    with client.websocket_connect("/ws"):
        assert made[-1].result_threshold == 0.35

    assert client.post("/settings", json={"result_threshold": 2}).status_code == 400
    assert client.post("/settings", json={"result_threshold": "x"}).status_code == 400


def test_settings_camera_width_round_trips(web_client):
    """A page/capture knob, not an identify one — it must work with no client."""
    s = web_client.get("/settings").json()
    assert s["camera_width"] == 1920 and s["analysis_width"] == 960

    s = web_client.post("/settings", json={"camera_width": 3840}).json()
    assert s["camera_width"] == 3840
    assert web_client.get("/settings").json()["camera_width"] == 3840

    for bad in (100, 99999, "wide"):
        assert (
            web_client.post("/settings", json={"camera_width": bad}).status_code == 400
        )
    assert (
        web_client.get("/settings").json()["camera_width"] == 3840
    )  # no partial apply


def test_settings_without_identify_client(web_client):
    assert web_client.get("/settings").json()["enabled"] is False
    assert web_client.post("/settings", json={"game": "Pokémon"}).status_code == 400
    # The threshold and camera width are ours either way, so a save that
    # touches only those must go through. The dialog stays usable with no key.
    assert (
        web_client.post("/settings", json={"result_threshold": 0.5}).status_code == 200
    )
    assert (
        web_client.post(
            "/settings", json={"result_threshold": 0.4, "camera_width": 1280}
        ).status_code
        == 200
    )


def test_settings_serves_the_limits_it_validates_against(web_client):
    """The page draws its controls from these, so they have to come from the
    same model that rejects out-of-range values — not a second hardcoded copy."""
    limits = web_client.get("/settings").json()["limits"]
    assert limits["camera_widths"] == [640, 1280, 1920, 2560, 3840]
    assert limits["result_threshold"] == {"min": 0.0, "max": 1.0, "step": 0.05}

    lo, hi = limits["result_threshold"]["min"], limits["result_threshold"]["max"]
    assert (
        web_client.post("/settings", json={"result_threshold": hi}).status_code == 200
    )
    assert (
        web_client.post("/settings", json={"result_threshold": lo}).status_code == 200
    )
    assert (
        web_client.post("/settings", json={"result_threshold": hi + 0.1}).status_code
        == 400
    )
    for width in limits["camera_widths"]:
        assert (
            web_client.post("/settings", json={"camera_width": width}).status_code
            == 200
        )


def test_settings_rejects_bad_input_as_400_with_an_error_string(web_client):
    """FastAPI would answer 422 {"detail": [...]}; the page reads .error off a
    400, so the validation handler has to keep that contract."""
    for payload in (
        {"result_threshold": 5},
        {"result_threshold": "abc"},
        {"camera_width": 1},
        {"send_width": 99},  # 0 or 320..7680 — 99 is neither
        {"typo": 1},  # extra="forbid": a typo is not a silent no-op
    ):
        r = web_client.post("/settings", json=payload)
        assert r.status_code == 400, payload
        assert isinstance(r.json()["error"], str) and r.json()["error"]

    # 0 is the "send as captured" escape hatch, not an out-of-range width.
    assert web_client.post("/settings", json={"send_width": 0}).status_code == 200


def test_settings_patch_leaves_unsent_fields_alone(web_client):
    before = web_client.get("/settings").json()
    after = web_client.post("/settings", json={"camera_width": 1280}).json()
    assert after["camera_width"] == 1280
    for key in ("result_threshold", "send_width", "category", "known_attrs"):
        assert after[key] == before[key]


# --- stream mode (--source <url>) ---------------------------------------------


def test_mode_endpoint_camera(web_client):
    assert web_client.get("/mode").json() == {
        "mode": "camera",
        "endpoint": None,
        "source": None,
        "show_detection": False,  # bbox overlay is opt-in
        "split_results": False,  # history merges reappearances
        "min_card_time": 1.0,  # a card must earn its history row
    }


def test_mode_endpoint_stream(fake_detector, fake_embedder, fake_identify):
    def make_analyzer(on_result, on_log):
        raise AssertionError("not needed for /mode")

    client = TestClient(
        create_web_app(
            make_analyzer,
            source=WsJpegSource("ws://example:9/feed"),
            show_detection=True,  # --show-detection
            split_results=True,  # --split-results
        )
    )
    assert client.get("/mode").json() == {
        "mode": "stream",
        "endpoint": "ws://example:9/feed",
        "source": "ws",
        "show_detection": True,
        "split_results": True,
        "min_card_time": 1.0,
    }


def test_stream_mode_pulls_analyses_and_broadcasts(
    ws_source, fake_detector, fake_embedder, fake_identify
):
    pytest.importorskip("websockets")
    endpoint = ws_source(jpeg_bytes())

    factory = _analyzer_factory(fake_detector, fake_embedder, fake_identify)
    app = create_web_app(factory, source=WsJpegSource(endpoint))
    # `with TestClient(...)` runs startup, which launches the pump task.
    with TestClient(app) as client:
        got_frame, got_identified = _drain_until_frame_and_identified(client, 60)
    assert got_frame, "viewer never received a relayed source frame"
    assert got_identified, "analysis of the source stream never identified"
    assert fake_identify.calls == 1  # identical frames -> one identify


def test_capture_mode_pulls_from_video_source(
    tmp_path, fake_detector, fake_embedder, fake_identify
):
    # cv2.VideoCapture opens files with the same code path as rtsp:// URLs —
    # a looping MJPG AVI stands in for a camera (EOF -> pump reconnects).
    video = write_mjpg_avi(tmp_path / "feed.avi")

    factory = _analyzer_factory(fake_detector, fake_embedder, fake_identify)
    app = create_web_app(factory, source=CaptureSource(video))
    with TestClient(app) as client:
        assert client.get("/mode").json()["source"] == "capture"
        got_frame, got_identified = _drain_until_frame_and_identified(client, 80)
    assert got_frame, "viewer never received a relayed rtsp frame"
    assert got_identified, "rtsp analysis never identified"
    assert fake_identify.calls == 1


def test_stream_pump_relays_the_analysed_frame_not_the_original(
    ws_source, fake_detector, fake_embedder, fake_identify
):
    """Viewers must never receive the full-resolution frame: bboxes are in
    analysis space, and a 4K source would swamp the browser socket."""
    pytest.importorskip("websockets")
    endpoint = ws_source(jpeg_bytes(make_frame(w=1600, h=900, fill=30)))
    factory = _analyzer_factory(
        fake_detector, fake_embedder, fake_identify, analysis_width=400
    )
    app = create_web_app(factory, source=WsJpegSource(endpoint), analysis_width=400)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        for _ in range(60):
            message = ws.receive()
            raw = message.get("bytes")
            if raw is None:
                continue
            relayed = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            assert relayed.shape[:2] == (225, 400)  # analysed, not 1600x900
            break
        else:
            raise AssertionError("no relayed frame arrived")


def test_stream_pump_reconnects_after_source_failure(
    monkeypatch, fake_detector, fake_embedder, fake_identify
):
    """A source that dies is reopened with backoff — the pump must not stay
    down after a SourceError."""
    import threading
    import time as time_mod

    from cardstream.client import web_stream as web_stream_mod
    from cardstream.client.sources import Backoff, FrameSource, SourceError

    # Fast backoff so the test doesn't sleep through the default 1s delay.
    monkeypatch.setattr(
        web_stream_mod, "Backoff", lambda: Backoff(initial=0.02, cap=0.05)
    )

    class FlakySource(FrameSource):
        name = "flaky"
        endpoint = "flaky://test"

        def __init__(self):
            self.opens = 0
            self.opened_twice = threading.Event()

        def frames(self):
            self.opens += 1
            if self.opens == 1:
                yield make_frame(), None
                raise SourceError("connection lost")
            self.opened_twice.set()
            while True:
                yield make_frame(), None
                time_mod.sleep(0.005)

        @property
        def is_live(self):
            return True

    source = FlakySource()

    factory = _analyzer_factory(fake_detector, fake_embedder, fake_identify)
    app = create_web_app(factory, source=source)
    with TestClient(app):
        assert source.opened_twice.wait(5), "pump never reconnected after SourceError"
    assert source.opens >= 2


def test_min_card_time_reaches_the_page():
    """The page owns the history list, so the threshold has to travel to it —
    /mode is the only channel that carries page-level settings."""
    app = create_web_app(lambda **k: None, min_card_time=2.5)
    assert TestClient(app).get("/mode").json()["min_card_time"] == 2.5


def test_min_card_time_defaults_to_one_second():
    app = create_web_app(lambda **k: None)
    assert TestClient(app).get("/mode").json()["min_card_time"] == 1.0
