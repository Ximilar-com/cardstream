"""Camera mode: the browser pushes frames, one analyzer per connection.

Frames arrive faster than full-resolution analysis can consume them, so the
socket reader and the analysis loop are decoupled by a one-slot mailbox that
drops rather than queues.
"""

from __future__ import annotations

import asyncio
import contextlib

from cardstream.client.web_common import (
    LogSink,
    WebSocket,
    WebSocketDisconnect,
    result_payload,
    snapshot_payload,
)
from cardstream.core.imaging import decode_jpeg

# How many stale frames one analysis pass may skip past. Analysis is
# synchronous on this loop and full-resolution frames make it slower than the
# page can send; without this, latency grows with every frame.
_MAX_SKIP_AHEAD = 8


class LatestFrame:
    """One-slot mailbox between the socket reader and the analysis loop.

    A backlog is worthless here: every queued frame is staler than the one
    behind it, so analysing them in order only grows latency. The producer
    overwrites, and the consumer skips ahead to whatever arrived while it was
    busy — the same drop-don't-queue policy capture.js applies page-side.
    """

    def __init__(self) -> None:
        self._data: bytes | None = None
        self._arrived = asyncio.Event()
        self._closed = False
        self.skipped = 0

    def put(self, data: bytes) -> None:
        if self._data is not None:
            self.skipped += 1  # the frame it replaces was never analysed
        self._data = data
        self._arrived.set()

    def close(self) -> None:
        self._closed = True
        self._arrived.set()  # wake the consumer so it can notice

    @property
    def done(self) -> bool:
        return self._closed and self._data is None

    async def take(self) -> bytes | None:
        """The freshest frame, or None when the socket closed."""
        await self._arrived.wait()
        self._arrived.clear()
        data, self._data = self._data, None
        if data is None:
            return None
        # Each yield lets the reader hand over one more buffered message;
        # bounded so a fast producer can never starve analysis completely.
        for _ in range(_MAX_SKIP_AHEAD):
            await asyncio.sleep(0)
            if self._data is None:
                break
            self.skipped += 1
            data, self._data = self._data, None
        return data


def add_camera_ws(app, make_analyzer, debug: bool) -> None:
    """Register ``/ws`` for camera mode."""

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()

        async def send(payload: dict) -> None:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_json(payload)

        def push(payload: dict) -> None:
            # May fire from the event-loop thread (gate/detect logs inside
            # process()) or the identify thread — safe from both.
            asyncio.run_coroutine_threadsafe(send(payload), loop)

        def on_result(ident: dict) -> None:
            # Called from the analyzer's identify thread — push the result to
            # the page immediately instead of waiting for the next frame.
            push(result_payload(ident, analyzer.identify_calls))

        on_log = LogSink(debug, push)
        analyzer = make_analyzer(on_result, on_log)
        mailbox = LatestFrame()

        async def reader() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    data = message.get("bytes")
                    if data is None:
                        continue  # ignore text/control frames
                    mailbox.put(data)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                mailbox.close()

        reader_task = asyncio.create_task(reader())
        try:
            while not mailbox.done:
                data = await mailbox.take()
                if data is None:
                    continue  # woken by the close, nothing left to analyse
                if mailbox.skipped and debug:
                    on_log(f"[ws] behind — skipped {mailbox.skipped} frame(s)")
                    mailbox.skipped = 0
                frame = decode_jpeg(data)
                if frame is None:
                    continue
                await send(snapshot_payload(analyzer.process(frame), analyzer))
        except WebSocketDisconnect:
            pass
        finally:
            reader_task.cancel()
