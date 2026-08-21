"""Pulled-source mode: iterate a FrameSource in a thread; browsers view.

One shared analyzer for the process; connected pages get the relayed frames
plus the same result JSON camera mode sends. Analysis runs whether or not a
browser is open.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

from cardstream.client.analyzer import DEFAULT_ANALYSIS_WIDTH
from cardstream.client.sources import Backoff, FrameSource, SourceError
from cardstream.client.web_common import (
    LogSink,
    WebSocket,
    WebSocketDisconnect,
    result_payload,
    snapshot_payload,
)
from cardstream.core.imaging import JPEG_QUALITY_STREAM, downscale, encode_jpeg


class StreamPump:
    """Runs the source in a thread and fans results out to viewer sockets.

    Starts/stops with the app's lifespan and reconnects to the source with
    capped backoff (re-calling ``source.frames()`` reconnects — see sources.py).
    """

    def __init__(
        self,
        make_analyzer,
        debug: bool,
        source: FrameSource,
        analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    ) -> None:
        self._make_analyzer = make_analyzer
        self._analysis_width = analysis_width
        self._source = source
        self._analyzer = None
        self._viewers: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._on_log = LogSink(debug, self._push)

    @asynccontextmanager
    async def lifespan(self, app):
        self._loop = asyncio.get_running_loop()
        task = asyncio.create_task(asyncio.to_thread(self._run))
        try:
            yield
        finally:
            self._stop.set()  # unblock the source thread
            task.cancel()

    async def _broadcast(self, payload: dict | bytes) -> None:
        """Send to every viewer; a dead viewer is dropped, not fatal."""
        for ws in list(self._viewers):
            try:
                if isinstance(payload, (bytes, bytearray)):
                    await ws.send_bytes(payload)
                else:
                    await ws.send_json(payload)
            except Exception:
                self._viewers.discard(ws)

    def _push(self, payload: dict | bytes) -> None:
        """Thread-safe broadcast from the pump/identify threads."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    def _calls(self) -> int:
        return self._analyzer.identify_calls if self._analyzer is not None else 0

    def _on_result(self, ident: dict) -> None:
        self._push(result_payload(ident, self._calls()))

    def _run(self) -> None:
        analyzer = self._analyzer = self._make_analyzer(self._on_result, self._on_log)
        backoff = Backoff()
        while not self._stop.is_set():
            try:
                self._on_log(
                    f"[stream] opening {self._source.endpoint} ({self._source.name})"
                )
                for frame, jpeg in self._source.frames():
                    if self._stop.is_set():
                        return
                    backoff.reset()  # frames flowing
                    snapshot = analyzer.process(frame)
                    if self._viewers and self._loop is not None:
                        # Viewers get the frame the pipeline ANALYSED, never the
                        # full-resolution original: it keeps bbox coordinates
                        # meaningful and a 4K source off the browser's socket.
                        analysed = downscale(frame, self._analysis_width)
                        if jpeg is None or analysed is not frame:
                            jpeg = encode_jpeg(analysed, JPEG_QUALITY_STREAM)
                        if jpeg:
                            self._push(jpeg)
                        self._push(snapshot_payload(snapshot, analyzer))
                self._on_log("[stream] source ended — reconnecting")
            except SourceError as exc:
                self._on_log(f"[stream] {exc} — retrying in {backoff.delay:.0f}s")
            except Exception as exc:  # never let the pump die silently
                self._on_log(
                    f"[stream] {type(exc).__name__}: {exc} — retrying in {backoff.delay:.0f}s"
                )
            if self._stop.wait(backoff.next()):
                return

    def register(self, app) -> None:
        @app.websocket("/ws")
        async def ws_viewer(websocket: WebSocket) -> None:
            await websocket.accept()
            self._viewers.add(websocket)
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    # viewers don't send frames; ignore anything else
            except WebSocketDisconnect:
                pass
            finally:
                self._viewers.discard(websocket)
