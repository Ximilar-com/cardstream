# CLI reference

Every flag both commands accept, in the groups `--help` prints them in.

**This page is generated** from the argparse parser by
`scripts/gen-cli-docs.py`, and CI fails if it drifts. Edit the flag's `help=`
in `client/common.py` rather than this file.

`cardstream-web` is the superset: it carries the page flags on top of every
shared pipeline flag. Flags marked **web** are not available on
`cardstream-client`.

The defaults below are what a **bare invocation** actually uses. See
[Tuning](tuning.md) for what to change and why, and
[Locating the card](locators.md) for the two locators.


## web UI and output

| Flag | Default | What it does |
| --- | --- | --- |
| `--port` **web** | `8001` | local port for the UI (default 8001) |
| `--host` **web** | `127.0.0.1` | bind address; keep the default — this UI is meant to stay local |
| `--no-browser` **web** | off | don't auto-open the browser |
| `--show-detection` **web** | off | draw the located card on the page (hidden by default): the bounding box in RED, or with --segmentor-model its four-corner outline. Under --detection-expansion a second GREEN outline shows what is actually cut and paid for. The ⚙ dialog toggles it live too |
| `--camera-width` **web** | `1920` | resolution the browser asks its webcam for (ideal width, default 1920). Bigger frames identify better and cost more bandwidth + decode; the settings dialog changes it live |
| `--send-width` **web** | `1920` | cap the width the browser encodes and sends (default 1920, 0 = send as captured). The identify crop is cut from this, so more is sharper — but the page encodes every frame: ~10 ms at 1280, ~18 ms at 1920, ~49 ms at 3840 |
| `--split-results`, `--split_results` **web** | off | log every appearance of a card as its own history row; by default a card put back in frame resumes its existing row and its on-stream time keeps totalling |
| `--min-card-time`, `--min_card_time` **web** | `1.0` | keep a card out of the history list until it has been on stream this long (default 1.0, 0 = list every card). A card glimpsed mid-swap is identified and paid for like any other — this only stops it cluttering the log. Time is TOTAL across appearances unless --split-results |

## frame source

| Flag | Default | What it does |
| --- | --- | --- |
| `--source` | `camera` | frame source: a ws://\|wss:// URL = binary JPEG WebSocket feed; anything else is opened with OpenCV/FFmpeg — rtsp://, rtmp://, srt://, a webcam index, or a video/image file path ; 'camera' = the browser's webcam |
| `--listen` | off | LISTEN for a stream pushed to this machine instead of pulling: rtmp:// (e.g. OBS pushing to rtmp://0.0.0.0:1935/live) or srt:// (caller mode encoders). Uses the system ffmpeg binary. |
| `--ffmpeg` | off | force the system-ffmpeg puller for the source — portability fallback when the opencv wheel's FFmpeg lacks a protocol (e.g. srt) |
| `--fps` | — | analysis rate for finite sources: video files default to their native fps (0 = as fast as decoding allows); still images resend at this rate (default 10) |

## identification

| Flag | Default | What it does |
| --- | --- | --- |
| `--api-key` | — | Ximilar API key (default: XIMILAR_API_KEY env) |
| `--type` | `tcg` | which id endpoint to call (default: tcg) Choices: `comics`, `slab`, `sport`, `tcg`. |
| `--game` | — | prefill the game/sport sent as the record's Subcategory so the id endpoint narrows its search (faster + more precise); valid values depend on --type: tcg = Pokémon, Magic The Gathering, One Piece; sport = Baseball, Basketball, Football, Hockey, Soccer, MMA. Default: not specified — the web UI switches it live. |
| `--set-code`, `--set_code` | — | prefill the set code sent as the record's set_code (e.g. PBL) so the id endpoint only matches that set; default: not specified |
| `--alphabet` | — | writing system of the cards, sent as the record's Alphabet. Omitted by default — the endpoint classifies it. Set it whenever you also pass --game: a Subcategory prefill turns the endpoint's own alphabet detection OFF and it then assumes latin, so a japanese card matches its English print. Choices: `latin`, `japanese`, `chinese`, `korean`, `thai`. |
| `--known-attrs`, `--no-known-attrs` | `True` | assert Side=front + Rotation=rotation_ok on the record (default); --no-known-attrs lets the endpoint classify side and rotation itself (backs, rotated cards) |
| `--store-images`, `--store_images` | — | save every crop sent for identification into FOLDER as <call number>-<random>.jpg — one file per PAID call, byte-identical to the record's _base64 (default: keep nothing). The folder is created if it does not exist |
| `--store-images-type`, `--store_images_type` | `object` | what --store-images writes: object (default) = the crop that was identified, exactly as sent; frame = the whole frame it was cut from, for when the question is where the card was and what else was in shot Choices: `object`, `frame`. |

## detection

| Flag | Default | What it does |
| --- | --- | --- |
| `--detector` | `rfdetr` | detector family (default rfdetr); the --detector-model extension picks the runtime. Only used when --detector-model is given Choices: `rfdetr`, `rtdetr`. |
| `--detector-model` | — | locate cards by BOUNDING BOX. rfdetr: .onnx only; rtdetr: .onnx, a transformers model dir, or an HF hub id. Mutually exclusive with --segmentor-model, and one of the two is required |
| `--segmentor` | `rfdetr` | segmentor family (default rfdetr); only used when --segmentor-model is given Choices: `rfdetr`. |
| `--segmentor-model` | `model/segmentation/onnx/model.onnx` | locate cards by INSTANCE MASK (.onnx from an RF-DETR segmentation checkpoint). The mask gives the card's four corners, so the crop sent for identification is deskewed and cut tight at the card edge instead of square with background wedges — more precise than --detector-model, and a little slower. Mutually exclusive with it. This is the DEFAULT locator; an explicit --detector-model replaces it |
| `--detector-conf` | `0.35` | confidence floor, for whichever locator is in use |
| `--detector-classes` | — | comma-separated class filter for whichever locator is in use: names for a transformers model, integer ids for .onnx; empty = any |
| `--min-card-size`, `--min_card_size` | `0.1` | ignore any detection narrower than this fraction of the analysed frame in EITHER dimension — a card too far from the lens to read is a paid call for nothing (default 0.1, 0 = accept any box). Lower it if your framing is wide and real cards are getting ignored |
| `--min-card-aspect-ratio`, `--min_card_aspect_ratio` | `0.4` | ignore any detection whose SHORTEST side over its longest is under this — a card is ~0.71 held either way up, so a fragment (a corner clipped by the frame edge, a sleeve lip, a card caught edge-on mid-swap) is well below it. Orientation-blind by construction, and unlike --min-card-size it does not move when the card is held nearer the lens (default 0.4, 0 = accept any shape) |
| `--detection-expansion`, `--detection_expansion` | `0.0` | grow the located card by this fraction on EVERY side before cutting the crop that is sent for identification — 0.1 adds a tenth of the card's width and height to each edge. Works for both locators: a detector's box is expanded, a segmentor's four corners are pushed outward from their centre, and either way the crop is then cut from the ORIGINAL frame. Only the paid crop changes — the identity gate and the overlay still see the card as located (default 0, = send it as located) |
| `--width` | `960` | downscale frames to this width FOR ANALYSIS ONLY (motion, detection, tracking, identity gate); the crop sent for identification is always cut from the original frame. 0 = analyse at full resolution |

## identity gate and call policy

| Flag | Default | What it does |
| --- | --- | --- |
| `--gate` | `embedding` | same-card identity gate (phash needs no torch/onnx) Choices: `embedding`, `phash`. |
| `--embed-model` | `model/similarity/onnx/model.onnx` | .pt (TorchScript), .onnx or .tflite (default model/similarity/onnx/model.onnx); '' = torchvision pretrained MobileNetV2 |
| `--similarity-threshold` | `0.85` | cosine similarity below this = new card |
| `--phash-threshold` | `10` | pHash hamming distance above this = new card (--gate phash) |
| `--result-threshold` | `0.9` | drop identifications whose best-match distance exceeds this (lower = better match; 1.0 = keep everything) |
| `--cooldown` | `2.0` | min seconds between identify calls |
| `--retry-unmatched`, `--retry_unmatched` | `0.5` | ask again about a card whose identify came back with nothing (no match, or a match dropped by --result-threshold) after this many seconds (default 0.5). A card that DID match still costs exactly one call, however long it is held. Raise it, or set 0 to never retry, if something that can never match — a slab back, a hand read as a card — is sitting in frame spending calls |
| `--forget-after` | `2.0` | a card gone longer than this stops counting as the same card: the identity gate is cleared so the next one is identified from scratch (default 2s, 0 = remember forever). Short dropouts are unaffected |

## motion gate and detection throttle

| Flag | Default | What it does |
| --- | --- | --- |
| `--motion-threshold` | `8.0` |  |
| `--still-frames` | `2` |  |
| `--detect-interval` | `0.1` | moving scene |
| `--idle-detect-interval` | `0.2` | static, card present |
| `--empty-detect-interval` | `0.2` | static, no card |

## tracking

| Flag | Default | What it does |
| --- | --- | --- |
| `--tracker-model` | — | path to a vitTracker .onnx (OpenCV zoo); when set, a visual tracker carries the bbox between detections and detection drops to --tracking-detect-interval |
| `--tracker-score-threshold` | `0.3` | tracking score below this = card lost, re-detect now |
| `--tracking-detect-interval` | `2.0` | re-sync detection interval while the tracker is locked |

## diagnostics

| Flag | Default | What it does |
| --- | --- | --- |
| `--debug` | off | log gate similarities and state flow |

---

*Generated by `scripts/gen-cli-docs.py` — do not edit by hand.*
