"""FrameSource module: classification, JPEG splitting, and all three pullers.

FFmpeg-gated tests skip automatically when no system ffmpeg is on PATH.
"""

from __future__ import annotations

import io
import shutil

import cv2
import numpy as np
import pytest

from _helpers import jpeg_bytes, make_frame, write_mjpg_avi
from cardstream.client.sources import (
    CaptureSource,
    FFmpegSource,
    SourceError,
    WsJpegSource,
    iter_jpegs,
    make_source,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="system ffmpeg not installed"
)


# --- make_source classification ------------------------------------------------


def test_make_source_classification():
    assert make_source("camera") is None
    assert isinstance(make_source("ws://h:1/feed"), WsJpegSource)
    assert isinstance(make_source("wss://h/feed"), WsJpegSource)
    assert isinstance(make_source("rtsp://cam/s1"), CaptureSource)
    assert isinstance(make_source("rtmp://srv/live"), CaptureSource)
    assert isinstance(make_source("srt://h:9000"), CaptureSource)
    assert isinstance(make_source("0"), CaptureSource)
    assert isinstance(make_source("clip.mp4"), CaptureSource)


def test_make_source_ffmpeg_routing():
    assert isinstance(
        make_source("rtmp://0.0.0.0:1935/live", listen=True), FFmpegSource
    )
    assert isinstance(make_source("srt://:9000", listen=True), FFmpegSource)
    assert isinstance(make_source("rtsp://cam/s1", force_ffmpeg=True), FFmpegSource)
    with pytest.raises(SourceError, match="--listen supports"):
        make_source("rtsp://cam/s1", listen=True)
    with pytest.raises(SourceError):
        make_source("camera", listen=True)
    with pytest.raises(SourceError):
        make_source("ws://h/feed", force_ffmpeg=True)


def test_is_live():
    assert make_source("rtsp://cam/s1").is_live
    assert make_source("0").is_live  # webcams don't "end"
    assert make_source("srt://:9000", listen=True).is_live
    assert not make_source("clip.mp4").is_live


# --- ffmpeg argv builder (covers srt listen without needing an SRT caller) -----


def test_ffmpeg_argv_listen_variants():
    rtmp = FFmpegSource("rtmp://0.0.0.0:1935/live", listen=True).build_argv()
    assert rtmp[rtmp.index("-listen") + 1] == "1"
    assert "rtmp://0.0.0.0:1935/live" in rtmp

    srt = FFmpegSource("srt://:9000", listen=True).build_argv()
    assert "-listen" not in srt
    assert "srt://:9000?mode=listener" in srt

    srt2 = FFmpegSource(
        "srt://:9000?mode=listener&latency=200", listen=True
    ).build_argv()
    assert "srt://:9000?mode=listener&latency=200" in srt2  # not doubled

    pull = FFmpegSource("rtsp://cam/s1").build_argv()
    assert "-listen" not in pull
    assert "-re" not in pull  # network pulls pace themselves
    # No scale filter anywhere: frames arrive native so the identify crop can
    # be cut from them; the analyzer does the analysis downscale.
    for argv in (rtmp, srt, pull):
        assert "-vf" not in argv

    local = FFmpegSource("clip.mp4").build_argv()
    assert "-re" in local  # local files replay in real time


# --- iter_jpegs -----------------------------------------------------------------


def test_iter_jpegs_splits_stream():
    jpegs = [jpeg_bytes(make_frame(fill=i * 40)) for i in range(4)]
    stream = io.BytesIO(b"garbage-prefix" + b"".join(jpegs))
    out = list(iter_jpegs(stream, chunk_size=777))  # odd chunk size: split markers
    assert len(out) == 4
    for original, parsed in zip(jpegs, out, strict=False):
        assert parsed == original
        assert (
            cv2.imdecode(np.frombuffer(parsed, np.uint8), cv2.IMREAD_COLOR) is not None
        )


def test_iter_jpegs_empty_and_garbage():
    assert list(iter_jpegs(io.BytesIO(b""))) == []
    assert list(iter_jpegs(io.BytesIO(b"\x00" * 1000))) == []


def test_iter_jpegs_survives_markers_split_across_chunks():
    """chunk_size=1 is the worst case: every SOI/EOI marker spans a chunk
    boundary, exercising the split-prefix retention paths."""
    jpegs = [jpeg_bytes(make_frame(fill=i * 60)) for i in range(2)]
    stream = io.BytesIO(b"".join(jpegs))
    out = list(iter_jpegs(stream, chunk_size=1))
    assert out == jpegs


def test_iter_jpegs_drops_truncated_finaljpeg_bytes():
    jpeg = jpeg_bytes()
    stream = io.BytesIO(jpeg + jpeg[: len(jpeg) // 2])  # complete + truncated
    out = list(iter_jpegs(stream, chunk_size=64))
    assert out == [jpeg]


# --- CaptureSource ----------------------------------------------------------------


def test_capture_source_reads_file_at_native_resolution(tmp_path):
    """Sources no longer downscale — the analyzer does, and it needs the
    original to cut the identify crop from."""
    path = write_mjpg_avi(tmp_path / "clip.avi", frames=10, w=640, h=480)
    frames = list(CaptureSource(path, fps=0).frames())  # fps=0: unpaced
    assert len(frames) == 10
    frame, jpeg = frames[0]
    assert frame.shape[1] == 640 and jpeg is None
    # re-calling frames() re-opens (the --loop / reconnect contract)
    assert len(list(CaptureSource(path).frames())) == 10


def test_capture_source_open_failure(tmp_path):
    with pytest.raises(SourceError, match="could not open"):
        list(CaptureSource(str(tmp_path / "missing.avi")).frames())


def test_capture_source_paces_files_to_native_fps(tmp_path):
    import time

    path = write_mjpg_avi(tmp_path / "clip.avi", frames=5)  # written at 10 fps
    start = time.monotonic()
    frames = list(CaptureSource(path).frames())  # default: native pacing
    elapsed = time.monotonic() - start
    assert len(frames) == 5
    # 5 frames at 10 fps ≈ 0.5s of pacing (first frame is immediate).
    assert elapsed >= 0.3, f"file was not paced (took {elapsed:.3f}s)"


def test_capture_source_fps_override(tmp_path):
    import time

    path = write_mjpg_avi(tmp_path / "clip.avi", frames=5)
    start = time.monotonic()
    frames = list(CaptureSource(path, fps=100).frames())  # much faster than native
    elapsed = time.monotonic() - start
    assert len(frames) == 5
    assert elapsed < 0.3, f"fps override ignored (took {elapsed:.3f}s)"


# --- WsJpegSource ------------------------------------------------------------------


def test_ws_source_yields_frames_and_bytes(ws_source):
    jpeg = jpeg_bytes()
    url = ws_source(jpeg)
    got = []
    for frame, raw in WsJpegSource(url).frames():
        got.append((frame, raw))
        if len(got) >= 3:
            break
    assert all(raw == jpeg for _, raw in got)
    assert got[0][0].shape == (240, 320, 3)


def test_ws_source_connect_failure():
    with pytest.raises(SourceError, match="could not connect"):
        list(WsJpegSource("ws://127.0.0.1:1/nope").frames())


def test_ws_source_skips_text_and_undecodable_messages(ws_source):
    """Text frames and garbage bytes are skipped, real JPEGs still arrive."""
    jpeg = jpeg_bytes()
    url = ws_source(jpeg, prelude=("hello-text", b"\x00 not a jpeg"))
    for _frame, raw in WsJpegSource(url).frames():
        assert raw == jpeg  # the two prelude messages never surfaced
        break


# --- FFmpegSource -------------------------------------------------------------------


def test_ffmpeg_missing_binary_raises_source_error(monkeypatch, tmp_path):
    """CI without ffmpeg hits exactly this path — must be a clean SourceError."""
    from cardstream.client import sources as sources_mod

    monkeypatch.setattr(sources_mod.shutil, "which", lambda _name: None)
    with pytest.raises(SourceError, match="ffmpeg not found"):
        list(FFmpegSource(str(tmp_path / "x.avi")).frames())


@needs_ffmpeg
def test_ffmpeg_source_reads_file(tmp_path):
    path = write_mjpg_avi(tmp_path / "clip.avi", frames=12, w=640, h=480)
    frames = list(FFmpegSource(path).frames())
    assert len(frames) == 12
    frame, jpeg = frames[0]
    assert frame.shape[1] == 640  # native, not scaled by ffmpeg
    assert jpeg is not None
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) is not None


@needs_ffmpeg
def test_ffmpeg_source_bad_input_raises(tmp_path):
    bad = tmp_path / "not-a-video.bin"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(SourceError, match="no frames"):
        list(FFmpegSource(str(bad)).frames())


@needs_ffmpeg
def test_ffmpeg_listen_receives_pushed_rtmp(tmp_path):
    """Full push loop: FFmpegSource listens on rtmp://, a second ffmpeg
    process pushes the AVI to it — at least one frame must arrive."""
    import socket
    import subprocess
    import threading

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    path = write_mjpg_avi(tmp_path / "clip.avi", frames=40, w=320, h=240)
    url = f"rtmp://127.0.0.1:{port}/live"
    source = FFmpegSource(url, listen=True)

    def push():
        # small delay so the listener is up before the push starts
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-i",
                path,
                "-c:v",
                "flv",
                "-f",
                "flv",
                url,
            ],
            timeout=30,
            capture_output=True,
        )

    pusher = threading.Timer(1.0, push)
    pusher.start()
    got = 0
    try:
        for frame, jpeg in source.frames():
            assert frame.shape[0] > 0 and jpeg
            got += 1
            if got >= 3:
                break
    finally:
        pusher.join()
    assert got >= 3
