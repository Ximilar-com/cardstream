"""Browser UI for the smart client.

Starts a small LOCAL web app, opens your browser, and runs the whole smart
pipeline (RF-DETR / RT-DETRv2 detection + MobileNetV2 / pHash identity gate) in THIS
process. One crop per distinct card then goes to Ximilar.

One ``--source`` flag picks where frames come from (see sources.py):

* ``camera`` (default) — the page uses YOUR browser's webcam and streams
  JPEG frames into this process; each browser connection gets its own
  analyzer.
* anything else — a pulled source (ws:// JPEG feed, rtsp://rtmp://srt://
  pull, webcam index of THIS machine, video/image file) or, with
  ``--listen``, a stream PUSHED to this machine (OBS via RTMP, an SRT
  caller). One shared analyzer processes it; the browser page becomes a
  viewer showing the incoming frames, bounding box, results and (with
  ``--debug``) the log panel. Analysis runs even with no browser open.

Nothing here is meant to be exposed beyond localhost: the app serves the page
and the camera WebSocket for YOUR browser, and the analysis happens in this
process, on your machine.

Usage:
    export XIMILAR_API_KEY=...
    cardstream-web                       # opens http://127.0.0.1:8001
    cardstream-web --no-browser --port 9000
    cardstream-web --source ws://host:1234/feed
    cardstream-web --source rtsp://user:pw@cam/stream1
    cardstream-web --source rtmp://0.0.0.0:1935/live --listen   # OBS pushes here
    cardstream-web --source "srt://:9000" --listen              # SRT caller pushes
    cardstream-web --source 0     # this machine's webcam, browser views
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser

from cardstream.client.analyzer import DEFAULT_ANALYSIS_WIDTH
from cardstream.client.common import bounded_float
from cardstream.client.web_camera import add_camera_ws
from cardstream.client.web_common import (
    FASTAPI_HINT,
    FastAPI,
    JSONResponse,
    RedirectResponse,
    RequestValidationError,
    StaticFiles,
)
from cardstream.client.web_settings import AnalyzerRegistry, add_settings_routes
from cardstream.client.web_stream import StreamPump

# Durations. Negative would silently mean "never", which is what 0 already says.
_seconds = bounded_float(
    0.0, None, "expected a non-negative number of seconds (0 = no minimum)"
)


def create_web_app(
    make_analyzer,
    debug: bool = False,
    source=None,
    identify_client=None,
    show_detection: bool = False,
    result_threshold: float | None = None,
    split_results: bool = False,
    min_card_time: float = 1.0,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    camera_width: int = 1920,
    send_width: int = 1920,
):
    """Build the FastAPI app; ``make_analyzer(on_result, on_log)`` returns a
    fresh SmartAnalyzer (per browser connection in camera mode; one shared
    instance when ``source`` is given). With ``debug`` the analyzer's log
    lines are also pushed to the page as ``{"log": ...}`` frames.
    ``identify_client`` is the pipeline's SHARED identify target — the page's
    settings dialog reads and rebinds its options via GET/POST /settings, and
    ``result_threshold`` (the pipeline's startup default) is retuned on every
    live analyzer from there too."""
    # The whole webui/ tree, not just smart/ — the page imports ../shared/*.
    webui_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "webui"))

    live = AnalyzerRegistry(make_analyzer, result_threshold, camera_width, send_width)
    pump = (
        StreamPump(live.make, debug, source, analysis_width)
        if source is not None
        else None
    )
    app = FastAPI(title="Smart Card Client", lifespan=pump.lifespan if pump else None)

    @app.exception_handler(RequestValidationError)
    async def _validation_as_400(request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's default is 422 with a {"detail": [...]} body; the page has
        always read {"error": "..."} off a 400, so keep that contract."""
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"][1:])
        message = first["msg"].removeprefix("Value error, ")
        return JSONResponse(
            status_code=400,
            content={"error": f"{where}: {message}" if where else message},
        )

    @app.get("/")
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/smart/")

    add_settings_routes(
        app,
        live=live,
        identify_client=identify_client,
        source=source,
        show_detection=show_detection,
        split_results=split_results,
        min_card_time=min_card_time,
        analysis_width=analysis_width,
    )

    if pump is None:
        add_camera_ws(app, live.make, debug)
    else:
        pump.register(app)

    # After /ws so the WebSocket route wins; html=True serves index.html at
    # directory URLs (/smart/).
    app.mount("/", StaticFiles(directory=webui_dir, html=True), name="web")
    return app


def build_parser() -> argparse.ArgumentParser:
    """Every flag ``cardstream-web`` takes.

    Split out of :func:`main` so the banner, the tests and the docs contract
    can read the real parser back instead of keeping a second copy of the flag
    list. The ``common`` import stays lazy: importing this module must not drag
    the analyzer and its ML backends in behind it.
    """
    from cardstream.client.common import (
        add_pipeline_args,
        add_source_args,
        add_version_arg,
    )

    ap = argparse.ArgumentParser(
        description="Smart card client — browser UI, local analysis"
    )
    add_version_arg(ap)
    ui = ap.add_argument_group("web UI and output")
    ui.add_argument(
        "--port", type=int, default=8001, help="local port for the UI (default 8001)"
    )
    ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; keep the default — this UI is meant to stay local",
    )
    ui.add_argument(
        "--no-browser", action="store_true", help="don't auto-open the browser"
    )
    ui.add_argument(
        "--show-detection",
        action="store_true",
        help="draw the located card on the page (hidden by "
        "default): the bounding box in RED, or with "
        "--segmentor-model its four-corner outline. Under "
        "--detection-expansion a second GREEN outline shows "
        "what is actually cut and paid for. The ⚙ dialog "
        "toggles it live too",
    )
    ui.add_argument(
        "--camera-width",
        type=int,
        default=1920,
        help="resolution the browser asks its webcam for (ideal width, "
        "default 1920). Bigger frames identify better and cost more "
        "bandwidth + decode; the settings dialog changes it live",
    )
    ui.add_argument(
        "--send-width",
        type=int,
        default=1920,
        help="cap the width the browser encodes and sends (default 1920, "
        "0 = send as captured). The identify crop is cut from this, "
        "so more is sharper — but the page encodes every frame: "
        "~10 ms at 1280, ~18 ms at 1920, ~49 ms at 3840",
    )
    ui.add_argument(
        "--split-results",
        "--split_results",
        dest="split_results",
        action="store_true",
        help="log every appearance of a card as its own history row; "
        "by default a card put back in frame resumes its existing "
        "row and its on-stream time keeps totalling",
    )
    ui.add_argument(
        "--min-card-time",
        "--min_card_time",
        dest="min_card_time",
        type=_seconds,
        default=1.0,
        metavar="SECONDS",
        help="keep a card out of the history list until it has been "
        "on stream this long (default 1.0, 0 = list every card). "
        "A card glimpsed mid-swap is identified and paid for like "
        "any other — this only stops it cluttering the log. Time "
        "is TOTAL across appearances unless --split-results",
    )
    add_source_args(ap, default_source="camera")
    add_pipeline_args(ap)
    return ap


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(FASTAPI_HINT) from None

    from cardstream import __version__
    from cardstream.client.analyzer import SmartAnalyzer
    from cardstream.client.banner import print_banner
    from cardstream.client.common import build_pipeline
    from cardstream.client.sources import make_source

    ap = build_parser()
    args = ap.parse_args()
    # Before the models load and before anything can fail: what this run is,
    # and which of it you actually asked for.
    print_banner(ap, args, version=__version__, subtitle="browser UI, local analysis")

    try:
        source = make_source(
            args.source,
            listen=args.listen,
            force_ffmpeg=args.ffmpeg,
            fps=args.fps,
        )
        pipeline = build_pipeline(args)
    except (ValueError, RuntimeError) as exc:  # includes SourceError
        raise SystemExit(f"error: {exc}") from None

    def make_analyzer(on_result, on_log):
        return SmartAnalyzer(
            detector=pipeline.detector,
            embedder=pipeline.embedder,
            identify_client=pipeline.identify_client,
            config=pipeline.config,
            on_result=on_result,
            on_log=on_log,
            store=pipeline.store,
        )

    url = f"http://{args.host}:{args.port}"
    print(f"[web] cardstream {__version__} — {pipeline.description}")
    if source is not None:
        how = "listening for" if args.listen else "pulling"
        print(f"[web] frame source ({source.name}): {how} {source.endpoint}")
    print(f"[web] serving UI at {url} — Ctrl-C to stop")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()

    uvicorn.run(
        create_web_app(
            make_analyzer,
            debug=args.debug,
            source=source,
            identify_client=pipeline.identify_client,
            show_detection=args.show_detection,
            result_threshold=pipeline.config.result_threshold,
            split_results=args.split_results,
            min_card_time=args.min_card_time,
            analysis_width=pipeline.config.analysis_width,
            camera_width=args.camera_width,
            send_width=args.send_width,
        ),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
