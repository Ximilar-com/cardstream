"""Frame sources for the smart client — one abstraction for every puller.

Both entrypoints (headless ``stream_client.py`` and browser-UI
``web_client.py``) consume the same synchronous generator interface:

* ``CaptureSource`` — ``cv2.VideoCapture``: rtsp://, rtmp://, srt:// (pull),
  HTTP MJPEG, a local webcam index, or a video/image file. Protocol support
  depends on the FFmpeg bundled with the opencv wheel.
* ``WsJpegSource``  — binary JPEG frames from a ``ws://``/``wss://`` endpoint.
* ``FFmpegSource``  — the SYSTEM ``ffmpeg`` binary piping MJPEG to stdout.
  Needed for ``--listen`` (receiving a stream PUSHED by an encoder: OBS via
  RTMP, an SRT caller) and as a ``--ffmpeg`` portability fallback when the
  opencv wheel's FFmpeg lacks a protocol (e.g. no libsrt).

``make_source`` classifies a ``--source`` value and builds the right one.
Sources connect lazily inside :meth:`frames`; callers own retry/backoff by
re-calling ``frames()``.
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import IO

import cv2
import numpy as np

FrameAndJpeg = tuple[np.ndarray, bytes | None]

_NETWORK_SCHEMES = (
    "ws://",
    "wss://",
    "rtsp://",
    "rtmp://",
    "rtmps://",
    "srt://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
)


class SourceError(RuntimeError):
    """A source could not be opened / died; message is user-facing."""


class Backoff:
    """Capped exponential reconnect delay — shared by every reconnect loop.

    ``delay`` is what the next wait will be; ``next()`` returns it and doubles
    up to the cap; ``reset()`` snaps back once frames flow again.
    """

    def __init__(self, initial: float = 1.0, cap: float = 10.0) -> None:
        self._initial = initial
        self._cap = cap
        self.delay = initial

    def reset(self) -> None:
        self.delay = self._initial

    def next(self) -> float:
        current = self.delay
        self.delay = min(self.delay * 2, self._cap)
        return current


class FrameSource(ABC):
    name = "base"  # "capture" | "ws" | "ffmpeg" — for /mode and logs
    endpoint: str

    @abstractmethod
    def frames(self) -> Iterator[FrameAndJpeg]:
        """Connect (raise SourceError on failure) and yield
        ``(frame_bgr, original_jpeg_or_None)`` until the source ends.
        Re-calling ``frames()`` reconnects."""

    @property
    def is_live(self) -> bool:
        """Network sources reconnect forever; files/webcams end naturally."""
        return self.endpoint.startswith(_NETWORK_SCHEMES)


class CaptureSource(FrameSource):
    """cv2.VideoCapture puller — pull URLs, webcam indices, files.

    Finite video files are PACED to real time (decoding runs far faster than
    playback, which would rush the motion gate and the viewer): frames are
    delivered at the file's native fps by default. ``fps`` overrides the rate;
    ``fps=0`` disables pacing (as fast as decoding allows). Live sources
    (webcams, network URLs) pace themselves and are never slept.
    """

    name = "capture"

    def __init__(self, endpoint: str, fps: float | None = None) -> None:
        self.endpoint = endpoint
        self._fps = fps
        # A bare digit means a local webcam index, like the headless CLI.
        self._target: object = int(endpoint) if endpoint.isdigit() else endpoint

    @property
    def is_live(self) -> bool:
        return isinstance(self._target, int) or super().is_live

    def _frame_interval(self, cap: cv2.VideoCapture) -> float | None:
        if self.is_live:
            return None
        fps = self._fps if self._fps is not None else (cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return (1.0 / fps) if fps and fps > 0 else None

    def frames(self) -> Iterator[FrameAndJpeg]:
        import time

        cap = cv2.VideoCapture(self._target)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"could not open {self.endpoint}")
        interval = self._frame_interval(cap)
        next_due = time.monotonic()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return  # source ended / read failed
                # Frames leave at their native size: the analyzer downscales
                # for detection and cuts the identify crop from the original.
                yield frame, None
                if interval:
                    next_due += interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_due = time.monotonic()  # slower than real time: no debt
        finally:
            cap.release()


class WsJpegSource(FrameSource):
    """Binary JPEG frames from a WebSocket endpoint (relayed as sent)."""

    name = "ws"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def frames(self) -> Iterator[FrameAndJpeg]:
        from websockets.sync.client import connect

        try:
            conn = connect(self.endpoint, max_size=None)
        except Exception as exc:
            raise SourceError(f"could not connect to {self.endpoint}: {exc}") from exc
        try:
            for msg in conn:
                if not isinstance(msg, (bytes, bytearray)):
                    continue  # source control/text frames
                frame = cv2.imdecode(
                    np.frombuffer(msg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                yield frame, bytes(msg)
        except Exception as exc:
            # Connection dropped -> caller reconnects; say why instead of
            # swallowing silently (a programming error would otherwise vanish).
            print(f"[ws-source] {self.endpoint} dropped: {type(exc).__name__}: {exc}")
            return
        finally:
            conn.close()


def iter_jpegs(stream: IO[bytes], chunk_size: int = 65536) -> Iterator[bytes]:
    """Split a raw MJPEG byte stream into individual JPEGs.

    Scans for SOI (FFD8FF) .. EOI (FFD9) marker pairs — the format ffmpeg's
    ``-f mjpeg`` muxer emits back-to-back. Tolerates leading garbage.
    """
    buf = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8\xff")
            if start < 0:
                buf = buf[-2:]  # keep a possible split SOI prefix
                break
            end = buf.find(b"\xff\xd9", start + 3)
            if end < 0:
                if start > 0:
                    buf = buf[start:]
                break
            yield buf[start : end + 2]
            buf = buf[end + 2 :]


class FFmpegSource(FrameSource):
    """System-ffmpeg puller: decodes ANY protocol ffmpeg supports and pipes
    MJPEG to stdout. The only way to LISTEN for a pushed stream (OBS pushing
    RTMP, an SRT caller), and a fallback when opencv's FFmpeg lacks a
    protocol."""

    name = "ffmpeg"

    def __init__(self, endpoint: str, listen: bool = False) -> None:
        if listen and not endpoint.startswith(("rtmp://", "rtmps://", "srt://")):
            raise SourceError(
                f"--listen supports rtmp:// and srt:// sources, not {endpoint!r}"
            )
        self.endpoint = endpoint
        self._listen = listen

    @property
    def is_live(self) -> bool:
        return self._listen or super().is_live

    def build_argv(self) -> list[str]:
        url = self.endpoint
        argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
        if self._listen:
            if url.startswith(("rtmp://", "rtmps://")):
                argv += ["-listen", "1"]
            elif "mode=" not in url:  # srt listener is URL-native
                url += ("&" if "?" in url else "?") + "mode=listener"
        elif not url.startswith(_NETWORK_SCHEMES):
            argv += ["-re"]  # local file: read at native rate, like a live feed
        argv += ["-i", url, "-an"]
        # No scale filter: ffmpeg hands over native-resolution frames so the
        # identify crop can be cut from them. (If a pushed stream's fps ever
        # overwhelms the analyzer, throttle here with -vf fps=N.)
        argv += ["-f", "mjpeg", "-q:v", "4", "pipe:1"]
        return argv

    def frames(self) -> Iterator[FrameAndJpeg]:
        if shutil.which("ffmpeg") is None:
            raise SourceError(
                "ffmpeg not found on PATH — brew install ffmpeg "
                "(needed only for --listen / --ffmpeg sources)"
            )
        # `with`, not a bare Popen: __exit__ CLOSES stdout/stderr before it
        # waits. Killing and reaping the process alone leaves the two pipe
        # file descriptors to a finalizer, which leaks one pair per source
        # restart — and a reconnecting --listen source restarts a lot.
        with subprocess.Popen(
            self.build_argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            got_any = False
            try:
                for jpeg in iter_jpegs(proc.stdout):
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        continue
                    got_any = True
                    yield frame, jpeg
            finally:
                proc.kill()
                stderr = (proc.stderr.read() or b"").decode(errors="replace").strip()
        # Reached only when ffmpeg's pipe ended on its own (not when the
        # consumer stopped iterating) — dying without a single frame is an
        # open failure worth surfacing with ffmpeg's own words.
        if not got_any:
            tail = "; ".join(stderr.splitlines()[-3:]) or "no output"
            raise SourceError(f"ffmpeg produced no frames from {self.endpoint}: {tail}")


def make_source(
    source: str,
    listen: bool = False,
    force_ffmpeg: bool = False,
    fps: float | None = None,
) -> FrameSource | None:
    """Build the right FrameSource for a ``--source`` value.

    ``"camera"`` returns None — browser-side capture, only meaningful for the
    web client. ``fps`` overrides the real-time pacing of finite video files
    (None = the file's native rate; 0 = unpaced). Raises SourceError on
    invalid combinations.
    """
    source = source.strip()
    if source == "camera":
        if listen or force_ffmpeg:
            raise SourceError("--listen/--ffmpeg need a URL source, not 'camera'")
        return None
    if source.startswith(("ws://", "wss://")):
        if listen or force_ffmpeg:
            raise SourceError("--listen/--ffmpeg do not apply to ws:// sources")
        return WsJpegSource(source)
    if listen or force_ffmpeg:
        return FFmpegSource(source, listen=listen)
    return CaptureSource(source, fps=fps)
