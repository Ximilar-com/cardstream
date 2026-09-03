"""Shared CLI plumbing for the smart-client entrypoints.

Both ``stream_client.py`` (headless capture loop) and ``web_client.py``
(browser UI) run the exact same local pipeline — detector, identity gate,
identify target. The flags and component construction live here so the two
stay in lockstep. Argparse defaults are read from ``AnalyzerConfig()`` — one
source of truth for the tuning values.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from cardstream import __version__
from cardstream.client.analyzer import AnalyzerConfig
from cardstream.client.embedders import DEFAULT_EMBED_MODEL, Embedder, make_embedder
from cardstream.client.identify_target import IdentifyTarget
from cardstream.client.ximilar_api import DirectXimilarClient
from cardstream.core.detectors import (
    DEFAULT_DETECTOR,
    DEFAULT_DETECTOR_CONF,
    DEFAULT_SEGMENTOR,
    DEFAULT_SEGMENTOR_MODEL,
    CardDetector,
    make_detector,
    make_segmentor,
)
from cardstream.core.id_types import ALPHABETS, ID_TYPES
from cardstream.core.identify_options import IdentifyOptions
from cardstream.core.image_store import FRAME, OBJECT, STORE_TYPES, ImageStore
from cardstream.core.prices import price_summary
from cardstream.core.tracking import make_tracker

# Tuning defaults shown in --help and used when a flag is omitted.
_DEFAULTS = AnalyzerConfig()


def bounded_float(lo: float, hi: float | None, hint: str, *, inclusive: bool = True):
    """An argparse ``type=`` for a float in a range, with a hint on the error.

    Shared by every numeric flag here and in web_client: parse, range-check,
    and say what the bound MEANS rather than only what it is — the three
    hand-written copies had drifted into three different error styles.
    ``hi=None`` bounds below only.
    """

    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
        over = hi is not None and (number > hi if inclusive else number >= hi)
        if number < lo or over:
            raise argparse.ArgumentTypeError(f"{number} is out of range — {hint}")
        return number

    return parse


# A fraction OF THE FRAME. Half-open at the top: 1.0 would reject every box,
# which reads as "nothing is ever detected" rather than as a bad flag value.
_fraction = bounded_float(
    0.0, 1.0, "expected 0..1 (0 = off; 1 would reject every detection)", inclusive=False
)
# Grow-the-crop. Inclusive at both ends: 1.0 pushes every edge out by a full
# card-width, which is extreme but meaningful rather than degenerate.
_expansion = bounded_float(
    0.0,
    1.0,
    "expected 0..1 (0 = send the card as located; 0.1 = a tenth more on every side)",
)


@dataclass(frozen=True)
class Pipeline:
    """Everything ``build_pipeline`` wires up, ready to feed a SmartAnalyzer."""

    detector: CardDetector
    embedder: Embedder | None
    identify_client: IdentifyTarget
    config: AnalyzerConfig
    description: str  # one-line summary for startup banners
    # --store-images, or None. The identify client keeps the crops; the
    # analyzer keeps the frames — which of the two actually writes is the
    # store's own --store-images-type.
    store: ImageStore | None = None


def add_version_arg(ap: argparse.ArgumentParser) -> None:
    """--version prints and exits inside parse_args — before any model loads."""
    ap.add_argument("--version", action="version", version=f"cardstream {__version__}")


def add_source_args(ap: argparse.ArgumentParser, default_source: str) -> None:
    """Frame-source flags shared by both entrypoints (see sources.py)."""
    frames = ap.add_argument_group("frame source")
    frames.add_argument(
        "--source",
        default=default_source,
        help="frame source: a ws://|wss:// URL = binary JPEG WebSocket "
        "feed; anything else is opened with OpenCV/FFmpeg — "
        "rtsp://, rtmp://, srt://, a webcam index, or a "
        "video/image file path"
        + (" ; 'camera' = the browser's webcam" if default_source == "camera" else ""),
    )
    frames.add_argument(
        "--listen",
        action="store_true",
        help="LISTEN for a stream pushed to this machine instead of "
        "pulling: rtmp:// (e.g. OBS pushing to "
        "rtmp://0.0.0.0:1935/live) or srt:// (caller mode "
        "encoders). Uses the system ffmpeg binary.",
    )
    frames.add_argument(
        "--ffmpeg",
        action="store_true",
        help="force the system-ffmpeg puller for the source — "
        "portability fallback when the opencv wheel's FFmpeg "
        "lacks a protocol (e.g. srt)",
    )
    # Both entrypoints hand this straight to make_source; it lived in each of
    # them separately until the two help strings had already drifted apart.
    frames.add_argument(
        "--fps",
        type=float,
        default=None,
        help="analysis rate for finite sources: video files default "
        "to their native fps (0 = as fast as decoding allows); "
        "still images resend at this rate (default 10)",
    )


def add_pipeline_args(ap: argparse.ArgumentParser) -> None:
    """Detector / gate / identify-target flags shared by both entrypoints."""
    ident = ap.add_argument_group("identification")
    ident.add_argument(
        "--api-key", default=None, help="Ximilar API key (default: XIMILAR_API_KEY env)"
    )
    ident.add_argument(
        "--type",
        default="tcg",
        choices=sorted(ID_TYPES),
        help="which id endpoint to call (default: tcg)",
    )
    ident.add_argument(
        "--game",
        default=None,
        help="prefill the game/sport sent as the record's Subcategory "
        "so the id endpoint narrows its search (faster + more "
        "precise); valid values depend on --type: "
        # Straight off the registry, so a new category's
        # games show up here without anyone remembering to.
        + "; ".join(
            f"{t.key} = {', '.join(t.subcategories)}"
            for t in ID_TYPES.values()
            if t.subcategories
        )
        + ". "
        "Default: not specified — the web UI switches it live.",
    )
    ident.add_argument(
        "--set-code",
        "--set_code",
        dest="set_code",
        default=None,
        metavar="CODE",
        help="prefill the set code sent as the record's set_code "
        "(e.g. PBL) so the id endpoint only matches that set; "
        "default: not specified",
    )
    ident.add_argument(
        "--alphabet",
        default=None,
        choices=list(ALPHABETS),
        help="writing system of the cards, sent as the record's "
        "Alphabet. Omitted by default — the endpoint classifies "
        "it. Set it whenever you also pass --game: a Subcategory "
        "prefill turns the endpoint's own alphabet detection OFF "
        "and it then assumes latin, so a japanese card matches "
        "its English print.",
    )
    ident.add_argument(
        "--known-attrs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="assert Side=front + Rotation=rotation_ok on the record "
        "(default); --no-known-attrs lets the endpoint classify "
        "side and rotation itself (backs, rotated cards)",
    )
    ident.add_argument(
        "--price-stats",
        "--price_stats",
        dest="price_stats",
        action="store_true",
        help="ask the id endpoint for market price statistics with every "
        "match — USD median, range and latest sale, shown on the page and "
        "in the terminal. tcg, sport and comics only (slab has none); off "
        "by default, since the extra data is not documented as free. The "
        "web UI's settings dialog toggles it live.",
    )
    ident.add_argument(
        "--store-images",
        "--store_images",
        dest="store_images",
        default=None,
        metavar="FOLDER",
        help="save every crop sent for identification into FOLDER "
        "as <call number>-<random>.jpg — one file per PAID "
        "call, byte-identical to the record's _base64 "
        "(default: keep nothing). The folder is created if it "
        "does not exist",
    )
    ident.add_argument(
        "--store-images-type",
        "--store_images_type",
        dest="store_images_type",
        default=OBJECT,
        choices=list(STORE_TYPES),
        help=f"what --store-images writes: {OBJECT} (default) = the "
        f"crop that was identified, exactly as sent; {FRAME} = "
        "the whole frame it was cut from, for when the question "
        "is where the card was and what else was in shot",
    )

    detect = ap.add_argument_group("detection")
    detect.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        choices=["rfdetr", "rtdetr"],
        help=f"detector family (default {DEFAULT_DETECTOR}); the "
        "--detector-model extension picks the runtime. Only "
        "used when --detector-model is given",
    )
    detect.add_argument(
        "--detector-model",
        dest="detector_model",
        default=None,
        help="locate cards by BOUNDING BOX. rfdetr: .onnx only; "
        "rtdetr: .onnx, a transformers model dir, or an HF "
        "hub id. Mutually exclusive with --segmentor-model, "
        "and one of the two is required",
    )
    detect.add_argument(
        "--segmentor",
        default=DEFAULT_SEGMENTOR,
        choices=["rfdetr"],
        help=f"segmentor family (default {DEFAULT_SEGMENTOR}); only "
        "used when --segmentor-model is given",
    )
    # default=None, NOT the shipped path: `_locator` applies that only when no
    # locator was typed at all, so "the user asked for this" stays
    # distinguishable from "nobody typed anything".
    detect.add_argument(
        "--segmentor-model",
        dest="segmentor_model",
        default=None,
        help="locate cards by INSTANCE MASK (.onnx from an RF-DETR "
        "segmentation checkpoint). The mask gives the card's "
        "four corners, so the crop sent for identification is "
        "deskewed and cut tight at the card edge instead of "
        "square with background wedges — more precise than "
        "--detector-model, and a little slower. Mutually "
        "exclusive with it. This is the DEFAULT locator; "
        "an explicit --detector-model replaces it",
    )
    detect.add_argument(
        "--detector-conf",
        dest="detector_conf",
        type=float,
        default=DEFAULT_DETECTOR_CONF,
        help="confidence floor, for whichever locator is in use",
    )
    detect.add_argument(
        "--detector-classes",
        dest="detector_classes",
        default="",
        help="comma-separated class filter for whichever locator is "
        "in use: names for a transformers model, integer ids "
        "for .onnx; empty = any",
    )
    detect.add_argument(
        "--min-card-size",
        "--min_card_size",
        dest="min_card_size",
        type=_fraction,
        default=_DEFAULTS.min_card_fraction,
        metavar="FRACTION",
        help="ignore any detection narrower than this fraction of "
        "the analysed frame in EITHER dimension — a card too "
        "far from the lens to read is a paid call for nothing "
        f"(default {_DEFAULTS.min_card_fraction:g}, 0 = accept "
        "any box). Lower it if your framing is wide and real "
        "cards are getting ignored",
    )
    detect.add_argument(
        "--min-card-aspect-ratio",
        "--min_card_aspect_ratio",
        dest="min_card_aspect_ratio",
        type=_fraction,
        default=_DEFAULTS.min_card_aspect,
        metavar="RATIO",
        help="ignore any detection whose SHORTEST side over its "
        "longest is under this — a card is ~0.71 held either "
        "way up, so a fragment (a corner clipped by the frame "
        "edge, a sleeve lip, a card caught edge-on mid-swap) "
        "is well below it. Orientation-blind by construction, "
        "and unlike --min-card-size it does not move when the "
        f"card is held nearer the lens (default "
        f"{_DEFAULTS.min_card_aspect:g}, 0 = accept any shape)",
    )
    detect.add_argument(
        "--detection-expansion",
        "--detection_expansion",
        dest="detection_expansion",
        type=_expansion,
        default=_DEFAULTS.detection_expansion,
        metavar="FRACTION",
        help="grow the located card by this fraction on EVERY "
        "side before cutting the crop that is sent for "
        "identification — 0.1 adds a tenth of the card's "
        "width and height to each edge. Works for both "
        "locators: a detector's box is expanded, a "
        "segmentor's four corners are pushed outward from "
        "their centre, and either way the crop is then cut "
        "from the ORIGINAL frame. Only the paid crop "
        "changes — the identity gate and the overlay still "
        f"see the card as located (default "
        f"{_DEFAULTS.detection_expansion:g}, = send it as located)",
    )
    detect.add_argument(
        "--width",
        type=int,
        default=_DEFAULTS.analysis_width,
        help="downscale frames to this width FOR ANALYSIS ONLY "
        "(motion, detection, tracking, identity gate); the crop "
        "sent for identification is always cut from the original "
        "frame. 0 = analyse at full resolution",
    )

    gate = ap.add_argument_group("identity gate and call policy")
    gate.add_argument(
        "--gate",
        default=_DEFAULTS.gate,
        choices=["embedding", "phash"],
        help="same-card identity gate (phash needs no torch/onnx)",
    )
    gate.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=".pt (TorchScript), .onnx or .tflite "
        f"(default {DEFAULT_EMBED_MODEL}); "
        "'' = torchvision pretrained MobileNetV2",
    )
    gate.add_argument(
        "--similarity-threshold",
        type=float,
        default=_DEFAULTS.similarity_threshold,
        help="cosine similarity below this = new card",
    )
    gate.add_argument(
        "--phash-threshold",
        type=int,
        default=_DEFAULTS.phash_threshold,
        help="pHash hamming distance above this = new card (--gate phash)",
    )
    gate.add_argument(
        "--result-threshold",
        type=float,
        default=_DEFAULTS.result_threshold,
        help="drop identifications whose best-match distance exceeds this "
        "(lower = better match; 1.0 = keep everything)",
    )
    gate.add_argument(
        "--cooldown",
        type=float,
        default=_DEFAULTS.cooldown_seconds,
        help="min seconds between identify calls",
    )
    gate.add_argument(
        "--retry-unmatched",
        "--retry_unmatched",
        dest="retry_unmatched",
        type=float,
        default=_DEFAULTS.retry_unmatched_seconds,
        metavar="SECONDS",
        help="ask again about a card whose identify came back with "
        "nothing (no match, or a match dropped by "
        "--result-threshold) after this many seconds "
        f"(default {_DEFAULTS.retry_unmatched_seconds:g}). A "
        "card that DID match still costs exactly one call, "
        "however long it is held. Raise it, or set 0 to never "
        "retry, if something that can never match — a slab "
        "back, a hand read as a card — is sitting in frame "
        "spending calls",
    )
    gate.add_argument(
        "--forget-after",
        type=float,
        default=_DEFAULTS.forget_after_seconds,
        metavar="SECONDS",
        help="a card gone longer than this stops counting as the same "
        "card: the identity gate is cleared so the next one is "
        f"identified from scratch (default {_DEFAULTS.forget_after_seconds:g}s, "
        "0 = remember forever). Short dropouts are unaffected",
    )

    motion = ap.add_argument_group("motion gate and detection throttle")
    motion.add_argument(
        "--motion-threshold", type=float, default=_DEFAULTS.motion_threshold
    )
    motion.add_argument(
        "--still-frames", type=int, default=_DEFAULTS.still_frames_required
    )
    motion.add_argument(
        "--detect-interval",
        type=float,
        default=_DEFAULTS.detect_interval_seconds,
        help="moving scene",
    )
    motion.add_argument(
        "--idle-detect-interval",
        type=float,
        default=_DEFAULTS.idle_detect_interval_seconds,
        help="static, card present",
    )
    motion.add_argument(
        "--empty-detect-interval",
        type=float,
        default=_DEFAULTS.empty_detect_interval_seconds,
        help="static, no card",
    )

    track = ap.add_argument_group("tracking")
    track.add_argument(
        "--tracker-model",
        default=_DEFAULTS.tracker_model,
        help="path to a vitTracker .onnx (OpenCV zoo); when set, a "
        "visual tracker carries the bbox between detections and "
        "detection drops to --tracking-detect-interval",
    )
    track.add_argument(
        "--tracker-score-threshold",
        type=float,
        default=_DEFAULTS.tracker_score_threshold,
        help="tracking score below this = card lost, re-detect now",
    )
    track.add_argument(
        "--tracking-detect-interval",
        type=float,
        default=_DEFAULTS.tracking_detect_interval_seconds,
        help="re-sync detection interval while the tracker is locked",
    )

    diag = ap.add_argument_group("diagnostics")
    diag.add_argument(
        "--debug", action="store_true", help="log gate similarities and state flow"
    )


def _image_store(args) -> ImageStore | None:
    """The --store-images folder, or None. Constructed BEFORE any model loads
    so an unwritable path fails at startup rather than after the detector has
    spent seconds reading weights."""
    kind = getattr(args, "store_images_type", OBJECT)
    if getattr(args, "store_images", None):
        return ImageStore(args.store_images, kind)
    if kind != OBJECT:
        # Silently writing nothing because the folder flag was forgotten is the
        # kind of thing you discover after the show, not during it.
        raise ValueError("--store-images-type needs --store-images FOLDER")
    return None


def _identify_options(args) -> IdentifyOptions:
    """The prefill bundle every identify call carries. Built once and handed to
    the client, which is what stopped two call sites spelling the same six
    fields out separately (in different orders). ValueError here is
    user-facing: a bad --type/--game/--alphabet."""
    return IdentifyOptions(
        id_type=args.type,
        game=getattr(args, "game", None),
        set_code=getattr(args, "set_code", None),
        known_attrs=args.known_attrs,
        alphabet=args.alphabet,
        price_stats=args.price_stats,
    )


def _require_model(path: str, flag: str) -> None:
    """Fail on a missing model file with a message that names the fix.

    The shipped defaults point into ``model/``, which is gitignored — so a
    fresh checkout hits this, and it must read as "put the weights here", not
    as an onnxruntime stack trace.
    """
    if not path or os.path.exists(path):
        return
    raise ValueError(
        f"{flag}: no model at {path}\n"
        "Model weights are not in git. Drop them into model/ (see "
        "model/README.md), fetch them with scripts/build-from-source.sh "
        f"--models, or point {flag} somewhere else."
    )


def resolve_locator(args) -> tuple[str | None, str | None]:
    """``(detector_model, segmentor_model)`` this run will actually use.

    Neither flag is defaulted by argparse, so this is the ONE place the shipped
    segmentor is filled in — and only when the user typed no locator at all.
    That is what keeps "you asked for the segmentor" distinguishable from "you
    asked for nothing": comparing an argparse default against its own constant
    could not tell the two apart, so typing the shipped path alongside
    --detector-model used to drop it in silence instead of raising.
    """
    detector_model, segmentor_model = args.detector_model, args.segmentor_model
    if detector_model and segmentor_model:
        raise ValueError(
            "pick one card locator: --detector-model (bounding boxes) or "
            "--segmentor-model (deskewed card corners), not both"
        )
    if not detector_model and not segmentor_model:
        segmentor_model = DEFAULT_SEGMENTOR_MODEL
    return detector_model, segmentor_model


def _locator(args) -> CardDetector:
    """The ONE card locator: a detector OR a segmentor, never both, never neither.

    The MODEL PATH is the switch, not the family flag; :func:`resolve_locator`
    owns which path wins. Detection finds a box; segmentation finds the
    boundary and so can deskew the crop.
    """
    detector_model, segmentor_model = resolve_locator(args)
    _require_model(
        segmentor_model or detector_model,
        "--segmentor-model" if segmentor_model else "--detector-model",
    )
    classes = tuple(c.strip() for c in args.detector_classes.split(",") if c.strip())
    build = make_segmentor if segmentor_model else make_detector
    family = args.segmentor if segmentor_model else args.detector
    return build(
        family,
        model=segmentor_model or detector_model,
        conf=args.detector_conf,
        classes=classes,
    )


def _analyzer_config(args) -> AnalyzerConfig:
    """Every tuning flag, in one mapping. The dataclass owns the DEFAULTS (see
    _DEFAULTS above); this owns only flag name -> field name."""
    return AnalyzerConfig(
        gate=args.gate,
        analysis_width=args.width,
        min_card_fraction=args.min_card_size,
        min_card_aspect=args.min_card_aspect_ratio,
        detection_expansion=args.detection_expansion,
        similarity_threshold=args.similarity_threshold,
        phash_threshold=args.phash_threshold,
        result_threshold=args.result_threshold,
        cooldown_seconds=args.cooldown,
        forget_after_seconds=args.forget_after,
        retry_unmatched_seconds=args.retry_unmatched,
        motion_threshold=args.motion_threshold,
        still_frames_required=args.still_frames,
        detect_interval_seconds=args.detect_interval,
        idle_detect_interval_seconds=args.idle_detect_interval,
        empty_detect_interval_seconds=args.empty_detect_interval,
        tracker_model=args.tracker_model,
        tracker_score_threshold=args.tracker_score_threshold,
        tracking_detect_interval_seconds=args.tracking_detect_interval,
        debug=args.debug,
    )


def build_pipeline(args) -> Pipeline:
    """Construct a :class:`Pipeline` from parsed args.

    Assembly only — each part is built by one helper above, in the order that
    fails cheapest first. Raises ValueError / RuntimeError with a user-facing
    message on bad flags or missing optional dependencies; entrypoints turn
    that into SystemExit.
    """
    store = _image_store(args)
    options = _identify_options(args)
    config = _analyzer_config(args)

    detector = _locator(args)
    if args.gate == "embedding":
        _require_model(args.embed_model, "--embed-model")
        embedder = make_embedder(args.embed_model)
    else:
        embedder = None
    # Analyzers build their own tracker (stateful, one per connection) — this
    # throwaway construction just fails fast on a bad model path / old cv2.
    tracker = make_tracker(args.tracker_model, args.tracker_score_threshold)

    api_key = (args.api_key or os.environ.get("XIMILAR_API_KEY", "")).strip()
    identify_client: IdentifyTarget = DirectXimilarClient(api_key, options, store=store)

    gate_desc = f"{detector.name} + {args.gate}" + (
        f"({embedder.name})" if embedder else ""
    )
    if tracker is not None:
        gate_desc += f" + {tracker.name}-track"
    target = f"ximilar {args.type}_id"
    if options.set_code:
        target += f" [set_code={options.set_code}]"
    if options.price_stats:
        target += " [price_stats]"
    return Pipeline(
        detector=detector,
        embedder=embedder,
        identify_client=identify_client,
        config=config,
        description=f"{gate_desc} -> {target}",
        store=store,
    )


def print_identification(ident: dict) -> None:
    """Human-readable one-liner for an identification dict (both CLIs)."""
    name = ident.get("full_name") or ident.get("name") or "Unknown"
    dist = ident.get("distance")
    dist_s = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
    elapsed = ident.get("elapsed_ms")
    elapsed_s = f" ({elapsed:.0f}ms)" if isinstance(elapsed, (int, float)) else ""
    print(
        f"\n[IDENTIFIED] {name} "
        f"| set={ident.get('set')} #{ident.get('card_number')} "
        f"| tier={ident.get('confidence_tier')} dist={dist_s}{elapsed_s}"
    )
    # Only when the call asked for prices AND the endpoint had sales: the
    # same one-line summary the page puts on the history row.
    summary = price_summary(ident.get("price_stats") or [])
    if summary:
        print(f"             price: {summary}")
    for link_name, url in (ident.get("links") or {}).items():
        print(f"             {link_name}: {url}")
