# CLAUDE.md

Guidance for working in this repository.

## What this is

`cardstream` — a real-time trading-card identification service backed by
Ximilar's detection + `collectibles/v2/{tcg,sport,slab,comics}_id` endpoints:
point a camera at a card and get name / set / confidence the moment it enters
the frame.

**Core idea — one paid API call per distinct card, not per frame.** A per-card
state machine driven by cheap, fully-local signals decides *when* a fresh call
is warranted:

1. **Detection** finds the card box — or, with `--segmentor-model`,
   **segmentation** finds its four corners instead — throttled in three tiers by a free motion
   gate (moving / static-with-card / empty). Optionally a cheap per-frame
   **visual tracker** (OpenCV TrackerVit, `core/tracking.py`) carries the bbox
   between detections; while it stays locked, detection drops to a slow
   re-sync tier, and a tracker score drop forces an immediate re-detect.
2. **MotionGate** (mean frame-diff) reports when the scene has *settled*.
3. **Identity gate** decides if a settled card is the SAME one (skip) or NEW
   (trigger identify). Its memory survives short dropouts but is cleared once
   the card has been gone for `--forget-after` seconds (default 2, client
   only), so a card that comes back later is analysed fresh. The signature is
   committed when a call FIRES, not when it succeeds, so a card that MATCHED
   costs exactly one call however long it is held. A card that came back with
   nothing would stick the same way, which is why `--retry-unmatched SECONDS`
   (default 0.5, `0` = the strict old policy) drops the signature again after
   the delay — one call per distinct card still holds; one call per distinct
   *failure* does not.
4. **Identify** (`tcg_id` etc.) runs off the frame loop; result pushed back.
   The crop it sends is re-cut from the ORIGINAL frame — steps 1-3 all run on
   a `--width` downscale, identification does not.

The state machine lives ONCE in `core/engine.py` (`DecisionCore` + `CallGuard`
+ `IdentityGate`). One driver schedules it: `SmartAnalyzer`
(`cardstream.client`) runs the core synchronously (identify in a background
thread) — RF-DETR/RT-DETRv2 detection, MobileNetV2-embedding (or pHash) gate —
and only one JPEG crop per distinct card leaves the machine, straight to
Ximilar. **Everything runs locally; there is no service in the path.**

## Architecture

Single installable package, src-layout, one root `pyproject.toml` with extras
`[client] [torch] [onnx] [tflite]` (dev deps are a PEP 735
`[dependency-groups]` group, not an extra) and two console scripts,
`cardstream-client` (headless) and `cardstream-web` (local browser UI). Both
build the same pipeline from `client/common.py`; they differ only in how
results are surfaced.

```
src/cardstream/
  log.py             configure_logging() / get_logger()
  core/              SHARED, dependency-light (numpy + opencv-headless + requests only)
    engine.py        DecisionCore (throttle tiers incl. tracking re-sync, snapshot ladder,
                     cooldown, no-retry + its --retry-unmatched escape hatch,
                     forget-after-absence) -> Snapshot (state/bbox/identification/quad,
                     a named shape rather than a tuple that kept growing),
                     CallGuard (in-flight + interval + watchdog),
                     IdentityGate ABC (decide/commit/reset) + PhashGate
    motion.py        MotionGate + phash + hamming
    detection_filters.py  DetectionFilter ABC + MinSizeFilter (--min-card-size,
                     a FRACTION of the analysed frame, either dimension) and
                     MinAspectFilter (--min-card-aspect-ratio, SHORT side over
                     long, so orientation-blind — a card is ~0.71 either way
                     up). reject() returns the REASON, which doubles as the
                     debug line, so each rule owns its wording and the driver
                     owns none of it. make_detection_filters() builds only the
                     ENABLED ones (a 0 threshold builds no object), so adding a
                     rule is a class plus one line and the analyzer never grows
                     a branch
    tracking.py      ObjectTracker ABC + VitTracker (cv2.TrackerVit, opt-in via model
                     path) + make_tracker — carries the bbox between detections
    quad.py          THE card-corners geometry, numpy+cv2 only (no model, so it
                     tests without one): bbox_quad (a box AS corners, the
                     inverse of quad_bbox — lets a box locator describe its
                     crop outline in the one shape the page draws), expand_quad (--detection-expansion for
                     the corners path — scales about the centroid so each EDGE
                     moves out by the fraction, matching BoundingBox.expanded,
                     because the flag must mean one thing for both locators),
                     order_quad (TL/TR/BR/BL, angle-sorted
                     around the centroid — NOT the x+y rule, which ties at 45°,
                     exactly a hand-held card's tilt), mask_to_quad (largest
                     contour → approxPolyDP sweep for a 4-gon → minAreaRect
                     fallback), quad_bbox (the axis-aligned hull every later
                     stage still speaks), warp_quad (the four-point
                     transform, cut TIGHT — the box path's _CROP_MARGIN exists
                     to survive background wedges, and a deskew has none),
                     paid_quad (THE one statement of how --detection-expansion
                     applies to each kind of locator, so the outline the page
                     draws and the crop that costs money cannot diverge) and
                     map_quad (carries corners along with a tracker's moved
                     box — as approximate as the box, which is what the page
                     would otherwise fall back to drawing).
                     MIN_CROP_SIDE is shared with detectors._crop_with_margin
    detectors.py     CardDetector ABC → DetectionResult:
                     RF-DETR .onnx (its export: ImageNet-normalized input,
                     name-matched dets/labels outputs with a trailing no-object
                     logit), RF-DETR SEGMENTATION .onnx (the same export plus a
                     masks output — per-query logits at S/4 covering the whole
                     input plane, NOT ROI-aligned, so decoding is upsample +
                     threshold at logit 0; RfDetrSegOnnxDetector subclasses the
                     detector and shares its _decode), RT-DETRv2
                     .onnx (onnxruntime) / transformers dir-or-hub-id,
                     + make_detector / make_segmentor (family picks the class,
                     model extension picks the runtime). Every backend is
                     Apache-2.0 top to bottom — a licence-constrained one does
                     not belong here.
                     _best_named_detection() owns the pick-the-best-box rule for
                     the transformers backend (the only one carrying class
                     NAMES); _chw_blob() the input tensor the ONNX ones want
                     (+ _imagenet_blob for RF-DETR's mean/std).
                     DEFAULT_DETECTOR / DEFAULT_SEGMENTOR (families),
                     DEFAULT_SEGMENTOR_MODEL (the shipped segmentor — what a
                     bare `cardstream-web` runs) and DEFAULT_DETECTOR_CONF are
                     what the CLI applies — never the factories themselves,
                     which take whatever family and path they are handed. There
                     is deliberately no default DETECTOR model: --detector-model
                     is the opt-in for boxes, and an explicit one REPLACES the
                     defaulted segmentor instead of colliding with it.
                     ML imports stay lazy inside each class, so this lives in
                     core without dragging torch into a base install.
    _onnx.py         shared onnxruntime session bootstrap (detectors + embedders)
    id_types.py      THE id-type registry: one frozen IdType per endpoint (key,
                     label, url, category_attrs = the Top Category/Category pair,
                     subcategories = display name → "Subcategory",
                     price_stats = whether the endpoint documents the
                     top-level request flag) in ID_TYPES,
                     plus resolve_id_type(key, noun=…) — the ONLY validation site,
                     noun= just picks the wording ("type" for ?type=, "category"
                     for the settings dialog). Also ALPHABETS +
                     normalize_alphabet, normalize_set_code, NOT_SPECIFIED and
                     is_unspecified (the one "leave this field out" rule).
                     A new card category is one tuple entry here and nothing else.
    identify_client.py  XimilarIdentifier — THE paid identify call (upscale →
                     b64 → record → POST → parse), wrapped by the client's
                     DirectXimilarClient; auth_headers + DEFAULT_HTTP_TIMEOUT
                     + USER_AGENT ("CardStream") live here too — auth_headers is
                     THE place request headers are built, so a new one reaches
                     every endpoint without touching a call site. Optionally
                     hands each record's _base64 to an ImageStore
                     (--store-images) before the POST
    identify_options.py  IdentifyOptions — the frozen (id_type, game, set_code,
                     known_attrs, alphabet, price_stats) bundle every identify
                     call carries,
                     normalized once in __post_init__. .with_(**patch) owns the
                     cross-field rule (switching category DROPS a game the new
                     one doesn't know, but an explicitly patched bad game
                     raises); .record() builds the Ximilar record —
                     _objects[0] carries Side front + Rotation rotation_ok (unless
                     known_attrs=False), the id type's Category pair, the game as
                     "Subcategory" and the writing system as "Alphabet", while the
                     RECORD carries "set_code". .payload(record) builds the POST
                     body — the records envelope plus "price_stats": true,
                     sent only where IdType.price_stats says the endpoint
                     takes it, so the preference survives a category switch
                     without ever reaching slab_id.
    ximilar.py       HTTP + parsing only: TierThresholds, distance_to_tier,
                     post_json (handles RequestException / non-2xx / malformed
                     JSON), full_image_card_object, parse_best_match (which
                     hands best_match.price_stats to prices.py)
    prices.py        market price statistics, stdlib only: parse_price_stats
                     (best_match.price_stats → one flat entry per stats_type,
                     unusable ones skipped), select_price_stats (ungraded,
                     graded; overall only as the fallback), format_number /
                     money (two decimals, .00 dropped, $ prefixed — USD is
                     assumed, the API names no currency) and price_summary,
                     the one line the terminal prints and the page puts on a
                     history row. overlay.js carries the JS twin, and
                     tests/core/test_prices.py + tests/webui/price-stats.test.js
                     run the same cases against both
    image_store.py   ImageStore — the --store-images folder: one file per PAID
                     call. --store-images-type picks the shape and the mode
                     lives HERE, so both call sites stay unconditional: `object`
                     (default) writes the record's own _base64 from the identify
                     path (what went on the wire, upscaling and JPEG quality
                     included, saved BEFORE the POST so an unmatched card is
                     still on disk); `frame` writes the whole ORIGINAL frame
                     from SmartAnalyzer, which is the only place that still has
                     it. save_b64/save_frame no-op in the other mode; the
                     numbering is one sequence either way. Named <call number>-<random>.jpg: the number sorts
                     the folder in card order, the suffix stops a second run
                     overwriting the first. Constructed in build_pipeline ahead
                     of the model loads (a bad folder must fail at startup);
                     save_b64 logs and returns None instead of raising, so a
                     full disk costs the archive, not the show
    imaging.py       encode_jpeg(_b64) / decode_jpeg / upscale_small / downscale;
                     FramePair (analysis frame for detect+gate, full frame for the
                     identify crop; .crop(bbox) rescales outward + owns its array,
                     .warp(quad) is its segmentation counterpart — same contract,
                     deskewed and tight);
                     JPEG quality consts
    models.py        CardState, ConfidenceTier, BoundingBox, DetectionResult
                     (+ quad: a segmentor's four corners, None from any box
                     locator — the ONE thing that separates the two kinds),
                     Identification, AnalysisResult (identification may be a
                     dict client-side; quad + crop_quad — located vs paid-for —
                     serialize to nested ints for the WS)
  client/            the smart client
    analyzer.py      SmartAnalyzer — sync driver over DecisionCore (+ AnalyzerConfig,
                     the single source of tuning defaults — argparse reads from it);
                     process() is the per-frame sequence (frame pair, motion,
                     tracker, detect tick, snapshot); _detect_tick() and
                     _identify_detection() are the two steps worth naming.
                     A detection failing any core/detection_filters rule is
                     dropped to None before anything else sees it (defaults
                     0.2 / 0.4, both ON — a fragment detects at 0.9, identifies
                     at nothing and costs a call);
                     builds the FramePair each frame: everything local runs on the
                     --width analysis frame, the identify crop is cut from the
                     original by _identify_crop() — pair.crop(det.bbox), or
                     pair.warp(det.quad) when the locator found corners, which
                     is what makes --segmentor pay off, and the ONE place
                     --detection-expansion is applied (so it changes what is
                     IDENTIFIED without touching what was LOCATED);
                     stamps elapsed_ms on each identification for the UI, counts
                     fired calls in .identify_calls (the page's "N calls" badge) and
                     takes result_threshold live from the settings dialog via
                     .tune(), which replaces the frozen config wholesale;
                     LIVE_FIELDS is the honest list of what CAN be retuned
    embedders.py     embedder backends (torch/.pt, .onnx, .tflite — extension-routed
                     via _EXTENSION_BACKENDS) + EmbeddingGate(IdentityGate)
    _tflite.py       shared LiteRT/TFLite interpreter bootstrap
    sources.py       FrameSource ABC + Backoff; capture / ws / system-ffmpeg pullers
                     (they yield NATIVE-resolution frames — the analyzer downscales)
    common.py        shared flags + build_pipeline() -> Pipeline dataclass —
                     assembly only, with _image_store() / _identify_options() /
                     _analyzer_config() doing the per-part work in the order
                     that fails cheapest first. resolve_locator() owns the
                     detector-or-segmentor rule and is the ONE place the
                     shipped segmentor default is applied; bounded_float()
                     is the shared numeric-flag validator.
                     Flags are declared in argparse GROUPS (frame source /
                     identification / detection / identity gate and call
                     policy / motion gate and detection throttle / tracking
                     / diagnostics) — banner.py and --help both read that
                     grouping back, so a new flag lands in a section without
                     a second list to update. Each entrypoint exposes
                     build_parser() so the banner, the tests and the docs
                     contract read the real parser rather than a copy
    banner.py        the startup banner both entrypoints print before any
                     model loads: the wordmark, then every knob and its
                     value, with the ones that came from a flag MARKED and
                     carrying the default they replaced (a dropped `\` in a
                     pasted multi-line command is then visible, not silent).
                     Pure formatting (parser + namespace -> str); the API key
                     is confirmed, never printed; colour only for a tty
    identify_target.py IdentifyTarget ABC — the ONE type the analyzer takes;
                     owns the mutable `options: IdentifyOptions` the settings
                     dialog rebinds (target.options = target.options.with_(...))
    ximilar_api.py   DirectXimilarClient — the only implementation, wrapping
                     core's XimilarIdentifier
    stream_client.py / web_client.py   the two entrypoints
    web_common.py    the ONE fastapi import guard + the three message shapes both
                     frame paths send (result_payload / snapshot_payload / LogSink)
    web_settings.py  GET /mode + GET/POST /settings (the ⚙ dialog's whole
                     contract). SettingsPatch (pydantic, extra=forbid) validates
                     — POST is a PATCH, so exclude_unset is what separates
                     "absent" from "cleared" — and serves its own bounds as the
                     `limits` block, so the page's controls and the process's
                     validation are the same numbers. AnalyzerRegistry retunes
                     every analyzer in flight (weakly held — a closed
                     connection stays collectable) via SmartAnalyzer.tune()
    web_camera.py    camera mode: browser pushes frames, one analyzer per
                     connection, LatestFrame keeps only the NEWEST queued frame
    web_stream.py    pulled-source mode: one shared analyzer in a thread,
                     browsers view; capped-backoff reconnect
  webui/
    shared/          constants.js (NOT_SPECIFIED — mirrored from core/id_types.py
                     and pinned by tests/core/test_webui_contract.py),
                     overlay.js (panel, colour-coded state badge, TWO outlines
                     via _strokeQuad: what was LOCATED in red — the segmentor's
                     four-corner quad when the snapshot carries one, else the
                     bbox; a tracker update drops the quad so an exact-looking
                     outline never sits in a stale place — and, only under
                     --detection-expansion, what is PAID FOR in green
                     (crop_quad). Red not the confidence tier because --high is
                     green; the tier is on its own badge and every history row.
                     With --price-stats the card panel gets a price block and
                     each history row a price line (renderPriceStats /
                     formatPriceStats — the JS twin of core/prices.py).
                     Plus history rows timing each card's stay, reappearances
                     resume the row unless --split-results. A row is BUILT on
                     identification but held out of the DOM until the card has
                     been on stream --min-card-time (_revealIfEarned, called
                     from the 500ms tick AND from _closeEntry so a card that
                     leaves between ticks still earns its row) — so a card
                     glimpsed mid-swap never appears, rather than flashing in
                     and being removed),
                     ws.js, capture.js (width 0 = send as captured),
                     theme.js (the ☀/☾ header button: the stored choice in
                     localStorage wins, else the page follows the system
                     setting live; index.html applies the same rule inline
                     in <head> so the first paint is already right, and the
                     test pins the two to one key),
                     style.css (dark is the base palette; the light one is
                     :root[data-theme="light"] overriding the SAME tokens —
                     no rule elsewhere names a colour, and overlay.js
                     caches cssVar() per theme so the canvas follows),
                     logo.svg (the wordmark, copied by hand from web/)
    smart/           the page: sends frames at CAPTURE resolution (q0.85)
                     and maps bboxes with the analysed dims the process reports;
                     /mode switch + relayed-frame viewer, debug log panel, header
                     badges (cam / analysed / N calls), history rows timed per
                     appearance, and the ⚙ dialog — whose every control is
                     GENERATED from settings-fields.js (one descriptor per knob:
                     kind, label, hint, options/limits read off the process's
                     `limits` block). Save posts ONLY the fields that differ from
                     the confirmed state, so a threshold-only edit carries no
                     identify fields. The draft mirrors the process's state verbatim,
                     "Not Specified" included, which is what makes the dirty
                     check a plain !==. Adding a knob = one entry there + one
                     SettingsPatch field.
docs/                the USER reference: cli-reference.md (GENERATED from the
                     parser by scripts/gen-cli-docs.py — CI fails if stale),
                     tuning.md, locators.md, sources.md. Lives here rather than
                     on the site so tests/test_docs_contract.py can pin every
                     documented "(default X)" against the real parser
.github/             CI (ruff + mypy + both suites on 3.11-3.13 x linux/mac/
                     windows) and the tag-triggered release (PyPI Trusted
                     Publishing, no long-lived token; actions SHA-pinned)
model/               EVERY runtime weight, one root: detection/ (boxes),
                     segmentation/ (outlines), similarity/ (the gate embedder),
                     tracking/ (the vitTracker). Gitignored except its README,
                     which documents each one's contract
```

Endpoints (all served by `cardstream-web` on localhost, for its own page):
`WS /ws` — frames in / results out in camera mode, results out only in viewer
mode; `GET /mode` — which of the two the process is in, plus the page-level
settings that are not live-tunable (`show_detection`, `split_results`,
`min_card_time`); `GET`/`POST /settings`
— the ⚙ dialog's contract (POST is a PATCH); `GET /` redirects to the page,
and the whole `webui/` tree is mounted so `../shared/` resolves.

Data flow: a frame arrives (browser JPEG, or a `FrameSource` pull) →
`SmartAnalyzer.process()` → returns an `AnalysisResult` snapshot immediately;
identification is pushed via the `on_result` callback when the background
identify thread returns. **Camera mode gives each browser connection its own
`SmartAnalyzer`; a pulled `--source` has ONE shared analyzer in a thread and
browsers are viewers.**

`--source` accepts: `camera` (web client only — browser webcam), a webcam
index, video/image files, `rtsp://` / `rtmp://` / `srt://` pull URLs
(cv2.VideoCapture), `ws://` binary-JPEG feeds, plus `--listen` (receive a
PUSHED RTMP/SRT stream via the system ffmpeg binary — the OBS case) and
`--ffmpeg` (force the system-ffmpeg puller when the opencv wheel lacks a
protocol). **Locating the card always uses a model, and there are two kinds.**
Segmentation is the DEFAULT — a bare `cardstream-web` runs
`model/segmentation/onnx/model.onnx` — and detection is the opt-in:

* `--detector-model PATH` (family `--detector rfdetr|rtdetr`) — a bounding box.
  Passing it REPLACES the defaulted segmentor; passing both model flags
  explicitly is an error (`resolve_locator` owns that rule, and compares the
  paths as TYPED — neither is an argparse default).
* `--segmentor-model PATH` (family `--segmentor rfdetr`) — an instance mask,
  from which `core.quad` recovers the card's four corners. The identify crop is
  then **deskewed and cut tight at the card edge** instead of square with
  background wedges, and so is the crop the identity gate compares — a card
  tilting in the hand keeps its embedding instead of drifting. More precise,
  and a little slower (~82 ms/frame for the shipped nano at 312², against
  ~80–150 ms for detection; the mask→corners→warp postprocess adds ~1 ms).

Built by `core.detectors.make_detector` / `make_segmentor`. The family picks the
class and the model extension picks the runtime: `rfdetr` is `.onnx` only
(the rfdetr pipeline's export_onnx.py); `rtdetr` routes `.onnx`→onnxruntime,
anything else→transformers. `--detector-conf` and `--detector-classes` tune
whichever locator is in use — there is deliberately no `--segmentor-conf` to
keep in sync — and everything downstream (`--detect-interval` and the other
throttle tiers, the shape filters, the identity gate, the call policy) is
untouched by the choice, because a segmentor returns the same `DetectionResult`
with `quad` filled in. `--embed-model` works the same way for the gate.
One locator is built per process (it holds a loaded model) and shared across
connections. Every runtime weight lives under `model/` (`detection/`,
`segmentation/`, `similarity/`, `tracking/` — see `model/README.md`); the
Apache-2.0
RF-DETR and RT-DETRv2 training pipelines (Ximilar data download → finetune →
ONNX export) are a SEPARATE REPO, `cardstream/detector/`, checked out beside
this one — see the workspace CLAUDE.md. Visual tracking
is opt-in: `--tracker-model` pointing at a vitTracker `.onnx` (OpenCV zoo);
analyzers each own a tracker instance (stateful — never share one across
connections).

Resolution is three knobs, in the order pixels travel (camera mode):
`--camera-width` (default 1920) is what the browser asks the webcam for,
`--send-width` (default 1920, `0` = as captured) caps what it encodes and
sends, `--width` (default 960) is what gets **analysed**. The identify crop is
cut from the frame as sent, so `--send-width` decides match quality — and
costs page main-thread time per frame (~10 ms at 1280, ~18 ms at 1920, ~49 ms
at 3840, measured in Chrome). Pulled sources yield native frames; the analyzer
downscales its own copy for analysis.

## Defaults

The shipped defaults ARE the tuned configuration — `make prod` passes no flags
and gets the same pipeline `make dev` spells out. `AnalyzerConfig` remains the
single source for the tuning ones; argparse reads every default from it.

| Flag | Default | |
| --- | --- | --- |
| `--segmentor-model` | `model/segmentation/onnx/model.onnx` | the default locator |
| `--embed-model` | `model/similarity/onnx/model.onnx` | `--gate embedding` is also default |
| `--detector-conf` | 0.35 | |
| `--similarity-threshold` | 0.85 | |
| `--result-threshold` | 0.9 | |
| `--min-card-size` | 0.1 | of the analysed frame, BOTH dimensions |
| `--still-frames` | 2 | |
| `--detect-interval` | 0.1 | idle/empty both 0.2 |
| `--cooldown` | 2.0 | |
| `--forget-after` | 2.0 | |
| `--min-card-time` | 1.0 | seconds on stream before a card gets a history row |

`--game` and `--alphabet` are deliberately NOT defaulted: a Subcategory
prefill suppresses the endpoint's own Alphabet classification (see the
limitations), so defaulting the pair would identify every card as if it were
one game and match a Japanese print against its latin twin. Unset, the
endpoint classifies both.

`tests/test_docs_contract.py` reads the artefacts themselves and pins them:
the `make dev` recipe is parsed out of the Makefile and every flag it passes
compared against the parser's default (only the four diagnostics may differ),
every `(default X)` claim in README.md / CLAUDE.md / model/README.md is checked
against argparse, the Defaults table above is checked row by row, and every
repo-relative path the docs name must exist. It is the reason a retuning or a
directory move cannot quietly leave the docs behind.

## Commands

Everything runs from the repo root with one venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[client,onnx]' --group dev   # extras are independent; [torch]/[onnx] for ML backends

export XIMILAR_API_KEY=...      # the one env var anything here reads
make check                      # lint + typecheck + both suites — what CI runs
pytest                          # the whole offline suite (tests/{core,client})
node --test "tests/webui/*.test.js" # the page's pure logic (stdlib runner, no deps)
pytest -q tests/core/test_engine.py     # one file

# A card locator is REQUIRED — --detector-model or --segmentor-model, never both.
cardstream-web --detector-model model/detection/model.onnx      # boxes; webcam UI on :8001
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx  # corners + deskew
cardstream-web --source rtsp://cam/stream1 --debug
cardstream-web --source rtmp://0.0.0.0:1935/live --listen   # OBS pushes
cardstream-client --source 0                    # headless webcam
cardstream-web --detector rtdetr --detector-model card_rtdetr.onnx
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx  # corners + deskewed crop
cardstream-web --detector rfdetr --detector-model model/detection/model.onnx \
    --tracker-model model/tracking/object_tracking_vittrack_2023sep.onnx   # per-frame bbox, slow re-sync
cardstream-web --game "Pokémon" --alphabet japanese --set-code M4   # record prefills
cardstream-web --camera-width 3840 --width 1280 --debug   # more pixels to identify from
cardstream-web --split-results                            # one history row per appearance
cardstream-web --price-stats                              # USD market prices with every match; toggle live in ⚙

cardstream-web --version                        # print the version and exit (also cardstream-client)

# Distribution (scripts/ + Dockerfile; artifacts go to GitHub releases)
scripts/build-release.sh                # wheel + sdist + SHA256SUMS into dist/ (--publish = gh release)
scripts/build-from-source.sh            # venv + editable install + smoke checks (--models fetches weights)
sh scripts/install.sh                   # the curl|sh installer (venv in ~/.cardstream, shims in ~/.local/bin)
docker build -t cardstream . && docker run --rm -e XIMILAR_API_KEY -p 127.0.0.1:8001:8001 -v cardstream-models:/models cardstream

# The training pipelines live in the SIBLING REPO cardstream/detector/ —
# separate venvs, own requirements.txt, nothing here depends on them.
```

## Conventions

- **One copy in core.** Decision-making and anything protocol-shaped (state
  machine decisions, motion/phash, Ximilar parsing, JPEG helpers, models)
  lives in `cardstream/core` — never re-implement it in the driver. `core`
  imports only numpy/cv2/requests; heavy ML imports stay lazy inside backend
  classes so any extras subset installs. It has no client imports: the split
  is what keeps `DecisionCore` testable without a camera or a browser.
- **The driver stays thin.** `SmartAnalyzer` owns scheduling (the identify
  thread), logging and I/O — behavior decisions belong in `DecisionCore`.
- **A locator is a locator.** A segmentor IS a `CardDetector` returning the
  same `DetectionResult`, with `quad` filled in — that is the whole extension
  point. Nothing downstream branches on which kind is running: `bbox` stays the
  quad's axis-aligned hull for the filters, the tracker and the overlay, and
  the single `det.quad is not None` check lives in `_identify_detection`.
- **Adding an identify prefill or a settings knob is a two-file change.** A
  card category is one `IdType` entry in `core/id_types.py`; an identify option
  is one field on `IdentifyOptions` (which owns normalization and the record);
  a settings knob is one descriptor in
  `webui/smart/settings-fields.js` plus one field on `SettingsPatch`. If a
  change needs more places than that, the duplication has come back. The one
  legitimate exception is an option that also changes the RESPONSE
  (`--price-stats`): that additionally touches `parse_best_match`, one
  `Identification` field and each renderer — the card panel, the history
  row, `print_identification` — which is the cost of a new output, not
  duplication.
- **The version is authored once, in `pyproject.toml`.** Everything else reads
  `cardstream.__version__` (importlib.metadata); both CLIs surface it via the
  shared `--version` flag (`add_version_arg` in `common.py`) and their startup
  banner. `scripts/install.sh` is the ONLY copy of the installer — the site
  and the docs point at its raw-main URL, and it is deliberately not a
  release asset, so a fix lands with a push.
- **Config is flag-driven, not env-driven.** argparse in `common.py`, with
  every default read from `AnalyzerConfig()` so the dataclass is the single
  source of tuning defaults. `XIMILAR_API_KEY` is the ONE environment variable
  anything here reads (`common.py`, as the fallback for `--api-key`); nothing
  loads a `.env`.
- **Output goes through `on_log`.** The analyzer never prints directly: it
  calls the `on_log` callback so the terminal and the browser's debug panel
  see the same lines — and it hands that same callback to
  `DecisionCore(log=…)`, so engine faults (a call that hung past its timeout)
  and routine events (a card away long enough to forget) arrive on ONE
  channel — one callback, not a second one for engine-level messages.
- **Analysis frame vs identify crop.** Local stages (motion, detection,
  tracking, identity gate) run on the `--width` downscale; the crop sent to the
  paid id endpoint is re-cut from the ORIGINAL frame via `FramePair.crop` —
  or `FramePair.warp` when the locator filled in `DetectionResult.quad`.
- **`--detection-expansion` touches the paid crop and nothing else.** It is
  applied in `SmartAnalyzer._identify_crop`, after every gate and filter has
  had its say, so the identity gate keeps comparing the tight crop (padding it
  would dilute the SAME-vs-NEW decision with background) and the overlay keeps
  drawing the box the model returned — and, with --show-detection, ALSO the
  expanded outline in green next to it, computed by the same primitives so the
  page cannot promise a crop the pipeline would not cut. Both geometries use
  one convention —
  each EDGE moves out by the fraction of that dimension, so 0.1 is 1.2x in
  each direction — via `BoundingBox.expanded` and `quad.expand_quad`.
  Sources and the browser therefore hand over native-resolution frames, and the
  gate deliberately keeps using the small `det.crop` so its embeddings stay
  comparable frame to frame.
- **The identify call never blocks the frame loop**: it runs in a background
  thread (tests pass `run_async=False` for deterministic snapshots). In camera
  mode the FastAPI event loop must stay free too — `LatestFrame` keeps only
  the newest queued frame, so a slow frame is dropped rather than queued.
- **Detectors return the core `DetectionResult`** (`bbox: BoundingBox, crop,
  prob, obj`) — no tuples, no driver-local Detection type.
- **The page shares `webui/shared/`** (ES modules + base CSS + the logo). Fix
  rendering bugs there once; `smart/` owns only page-specific behavior. The
  app mounts the whole `webui/` tree so `../shared/` resolves.
- **Tests are offline** and live in one suite: shared helpers/fakes in
  `tests/_helpers.py` (importable everywhere via `pythonpath = ["tests"]`),
  per-suite fixtures in `tests/client/conftest.py`. Use
  `wait_until(...)` to await background work — no fixed sleeps. System-ffmpeg
  tests skip when the binary is absent.
- New sources: subclass `FrameSource` in `client/sources.py`, add routing in
  `make_source`. New detector backends: subclass `CardDetector` in
  `core/detectors.py` (reuse `_crop_with_margin`, or `core/quad.py` for a
  mask-based one), route in `make_detector` / `make_segmentor`, and test the
  numerics with a fake ONNX session (see `tests/core/test_detectors.py`).

## Known limitations

- **DETECTOR boxes are axis-aligned**, so an angled card's crop includes
  background wedges (a small margin is added on purpose — `_CROP_MARGIN` —
  because tcg_id matches better with a little context than with a tight cut).
  `--segmentor-model` is the answer to this: a mask gives the four corners, and
  the identify crop is deskewed and cut tight. The segmentor's own limits are
  that it still picks ONE card per frame (the best-scoring query), and that a
  mask which never simplifies to a quadrilateral falls back to a rotated rect —
  and a mask with no contour at all falls back to the plain box, so a bad frame
  costs precision, not the detection.
- **The two locators expand from different baselines.** A detector's
  `det.bbox` already carries `_CROP_MARGIN` (4% a side, added by
  `_crop_with_margin`), so `--detection-expansion 0.1` there lands at roughly
  14%. A segmentor's quad is cut tight, so 0.1 is exactly 10%. Tune the value
  per locator rather than assuming it transfers.
- **The deskew keeps the card's as-seen orientation.** A card held sideways
  warps to a landscape rectangle, exactly as the box path crops one. Rotating
  it to portrait would be a coin flip on which way is up, so nothing tries.
- **The defaults point at weights that are not in git.** `model/` ships only
  its README, so a fresh checkout fails at startup on the default
  `--segmentor-model` / `--embed-model` paths. `_require_model` turns that into
  a message naming the flag and `model/README.md` rather than an onnxruntime
  traceback — but it is still a failure, and `scripts/build-from-source.sh
  --models` or a manual drop into `model/` is the fix. The offline test suite never loads
  it — the detector tests use a fake ONNX session and the analyzer tests pass
  a fake detector.
- **Camera mode ships full-resolution JPEGs over the WebSocket** (~460 KB per
  1080p frame, ~1.5 MB at 4K) and analyses them inline on the event loop, so a
  slow frame is skipped rather than queued. Fine on loopback, which is what
  `--host` defaults to; do not expose that socket to a LAN.
- **A game prefill costs alphabet detection.** Measured against the live
  endpoint: sending `Subcategory` stops it classifying `Alphabet` and it
  falls back to `latin`, so a Japanese card matches its English print. The
  Category pair and Side/Rotation do not have this effect. Hence
  `--alphabet` is therefore mandatory in practice whenever `--game` is set —
  it is omitted by default (endpoint classifies) and validated locally, since
  the API accepts an unknown value silently and returns a wrong match.
- **Watchdog clears, does not cancel.** If a detect/identify call exceeds
  `call_timeout_seconds` (default 20), `CallGuard.watchdog` only clears the in-flight flag
  so the pipeline can resume — the hung call itself is not cancelled, and a
  late result could overwrite a fresher one. Acceptable given the per-card
  cadence.
- **The tracker is presence, not identity.** TrackerVit's score answers "is a
  card-like object still where I'm looking?" — an in-place card swap keeps the
  score at ~0.88 (measured; trackers are appearance-robust by design), so
  tracker continuity must NEVER replace the embedding/pHash identity gate as
  the SAME-vs-NEW decision. Only full removal drops the score (~0.12). Also
  mind the economics: an update costs ~2.7 ms/frame CPU against ~80–150 ms for
  one detect, which is what makes CPU real-time viable with RF-DETR/RT-DETRv2.
- **The Ximilar key lives on the machine running the client** — that is the
  consequence of having no service in the path. The local app is unauthenticated
  and holds a key that spends credits, so keep it on localhost (`--host`
  defaults there) and do not expose it to a LAN.
- **`--listen` / `--ffmpeg` need the system ffmpeg binary** (brew install
  ffmpeg); plain pulls work with pip-only installs.
- **`model/` is gitignored except its README** (weights + training artifacts,
  ~250 MB for the segmentor alone). The subfolders only exist once you put
  weights in them, and `scripts/install.sh` + the Docker image do NOT use this
  layout — they fetch one separately versioned tarball PER MODEL (so a retrain
  republishes one archive) whose internal names are
  still flat (`segmentation_model/`, `similarity_model/`) into their own
  directory — the opt-in vitTracker comes straight from the OpenCV zoo,
  pinned by checksum, into `tracking_model/` — and write shims that pass the
  right paths. Those
  shims RESOLVE the locator from what unpacked (segmentor preferred, box
  detector as the fallback for an older publication) and skip it entirely when you
  pass a locator flag yourself — the two are mutually exclusive, so an
  unconditional one turned your own `--detector-model` into a "pick one" error.
- **Price statistics are USD by assumption and free by assumption.** The
  `price_stats` response names no currency, so `core/prices.py` `money()`
  prefixes `$`; and Ximilar does not document whether the flag costs credits,
  which is why `--price-stats` is opt-in. slab_id does not document the flag
  at all, so `IdentifyOptions.payload` never sends it there — the switch stays
  on in the dialog and applies again on the next category that takes it.
