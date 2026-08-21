"""Smart streaming client: local detection + embedding identity gate.

Runs everything except the final tcg_id lookup on this machine: an RF-DETR /
RT-DETRv2 detector finds the card, MobileNetV2 embeddings (or pHash) decide whether it's a new
card, and only then does one JPEG crop leave the machine — straight to
Ximilar's id endpoint using XIMILAR_API_KEY. There is no service in between.

``--source`` accepts the same values as the browser entrypoint (see
sources.py): a webcam index, video/image file, rtsp://, rtmp://, srt:// pull
URLs, a ws:// binary-JPEG feed, plus ``--listen`` to receive a stream pushed
to this machine (OBS via RTMP, an SRT caller) and ``--ffmpeg`` to force the
system-ffmpeg puller.

Usage:
    pip install -e '.[client,onnx]'     # ONNX Runtime backends (.onnx) — the default
    pip install -e '.[client,torch]'    # transformers RT-DETRv2 / torch MobileNetV2

    export XIMILAR_API_KEY=...
    cardstream-client --source 0                        # webcam, direct
    cardstream-client --source clip.mp4                 # video file
    cardstream-client --source card.jpg --loop          # still image
    cardstream-client --source rtsp://user:pw@cam/stream1
    cardstream-client --source rtmp://0.0.0.0:1935/live --listen  # OBS pushes
    cardstream-client --source 0 --detector rfdetr --detector-model card_rfdetr.onnx
    cardstream-client --source 0 --detector rtdetr --detector-model card_rtdetr.onnx
    cardstream-client --source 0 --gate phash           # no torch/onnx needed
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

from cardstream import __version__
from cardstream.client.analyzer import SmartAnalyzer
from cardstream.client.banner import print_banner
from cardstream.client.common import (
    add_pipeline_args,
    add_source_args,
    add_version_arg,
    build_pipeline,
    print_identification,
)
from cardstream.client.sources import Backoff, SourceError, make_source


class StatePrinter:
    """Prints '[state] ...' lines only on transitions, not per frame."""

    def __init__(self) -> None:
        self._last: str | None = None

    def __call__(self, snap) -> None:
        if snap.state.value != self._last:
            print(f"[state] {snap.state.value}")
            self._last = snap.state.value


def _run_still_image(analyzer, frame, fps: float, loop: bool, print_state) -> None:
    """A still image is a 1-frame 'video' — resend the decoded frame so the
    motion gate can settle; without --loop give it 10s then exit."""
    interval = 1.0 / fps
    start = time.monotonic()
    while True:
        print_state(analyzer.process(frame.copy()))
        time.sleep(interval)
        if not loop and time.monotonic() - start > 10:
            return


def _run_source(analyzer, source, loop: bool, print_state) -> None:
    """Iterate a FrameSource. Live sources (network URLs, webcams, --listen)
    reconnect forever with capped backoff; finite files are paced to real time
    by the source itself and exit at the end unless --loop."""
    backoff = Backoff()
    while True:
        try:
            for frame, _jpeg in source.frames():
                backoff.reset()  # frames flowing
                print_state(analyzer.process(frame))
        except SourceError as exc:
            if not source.is_live:
                raise SystemExit(f"error: {exc}") from None
            print(f"[client] {exc} — retrying in {backoff.delay:.0f}s")
        else:
            if not source.is_live:
                if not loop:
                    return
                continue  # finite file + --loop: reopen immediately
            print(f"[client] source ended — reconnecting in {backoff.delay:.0f}s")
        time.sleep(backoff.next())


def build_parser() -> argparse.ArgumentParser:
    """Every flag ``cardstream-client`` takes — see web_client.build_parser."""
    ap = argparse.ArgumentParser(
        description="Smart streaming card-id client (local detection + embedding gate)"
    )
    add_version_arg(ap)
    add_source_args(ap, default_source="0")
    play = ap.add_argument_group("playback")
    play.add_argument(
        "--loop", action="store_true", help="loop a finite source (e.g. still image)"
    )
    add_pipeline_args(ap)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    print_banner(ap, args, version=__version__, subtitle="headless, local analysis")

    try:
        pipeline = build_pipeline(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None

    analyzer = SmartAnalyzer(
        detector=pipeline.detector,
        embedder=pipeline.embedder,
        identify_client=pipeline.identify_client,
        config=pipeline.config,
        on_result=print_identification,
        store=pipeline.store,
    )

    print_state = StatePrinter()

    # A still image is a 1-frame "video" that VideoCapture can't seek/loop — so
    # detect it up front and just re-analyse the decoded frame.
    static_frame = None
    if not args.source.isdigit() and os.path.isfile(args.source):
        # Native resolution on purpose — the analyzer downscales for analysis
        # and cuts the identify crop from this original.
        static_frame = cv2.imread(args.source)

    print(f"[client] cardstream {__version__} — {pipeline.description}")
    try:
        if static_frame is not None:
            still_fps = args.fps or 10.0
            print(f"[client] analysing still image at {still_fps} fps — Ctrl-C to stop")
            _run_still_image(analyzer, static_frame, still_fps, args.loop, print_state)
        else:
            try:
                source = make_source(
                    args.source,
                    listen=args.listen,
                    force_ffmpeg=args.ffmpeg,
                    fps=args.fps,
                )
            except SourceError as exc:
                raise SystemExit(f"error: {exc}") from None
            if source is None:  # "camera" is a web_client concept
                raise SystemExit("error: --source camera is only for web_client.py")
            how = "listening for" if args.listen else "analysing"
            print(f"[client] {how} {source.endpoint} ({source.name}) — Ctrl-C to stop")
            _run_source(analyzer, source, args.loop, print_state)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[client] done")


if __name__ == "__main__":
    main()
