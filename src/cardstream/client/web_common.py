"""Shared plumbing for the browser UI's two frame paths.

Camera mode (browser pushes frames, one analyzer per connection) and stream
mode (one shared analyzer over a pulled source, browsers view) send the page
the *same three* message shapes, and each used to build them itself. They live
here once, along with the single fastapi/uvicorn import guard whose error
string used to appear verbatim in two places.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

FASTAPI_HINT = (
    "error: the browser UI needs fastapi + uvicorn — pip install 'cardstream[client]'"
)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(FASTAPI_HINT) from exc

__all__ = [
    "FASTAPI_HINT",
    "FastAPI",
    "JSONResponse",
    "LogSink",
    "RedirectResponse",
    "RequestValidationError",
    "StaticFiles",
    "WebSocket",
    "WebSocketDisconnect",
    "result_payload",
    "snapshot_payload",
]


def result_payload(ident: dict, identify_calls: int) -> dict[str, Any]:
    """A late identification, pushed the moment the call returns.

    ``bbox`` is None on purpose: the box in the snapshot the page already has
    is fresher than whatever was true when this call started.
    """
    return {
        "state": "identified",
        "bbox": None,
        "identification": ident,
        "identify_calls": identify_calls,
    }


def snapshot_payload(snapshot, analyzer) -> dict[str, Any]:
    """One analysed frame.

    The call counter rides on every snapshot — the page badge follows one
    analyzer, which IS the session (one per connection in camera mode, one per
    process in stream mode). ``analysis`` carries the dims those bboxes are in,
    which is no longer what the page sent.
    """
    return {
        **snapshot.to_dict(),
        "identify_calls": analyzer.identify_calls,
        "analysis": list(analyzer.analysis_size or ()),
    }


class LogSink:
    """The analyzer's ``on_log``: always to the console, to the page on --debug."""

    def __init__(self, debug: bool, push: Callable[[dict], None]) -> None:
        self._debug = debug
        self._push = push

    def __call__(self, msg: str) -> None:
        print(msg)
        if self._debug:
            self._push({"log": msg})
