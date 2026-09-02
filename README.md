<div align="center">

<img src="src/cardstream/webui/shared/logo.svg" alt="cardstream" width="320">

**Point a camera at a trading card. Get the name, set and confidence the moment
it enters the frame.**

[![PyPI](https://img.shields.io/pypi/v/cardstream)](https://pypi.org/project/cardstream/)
[![Python](https://img.shields.io/pypi/pyversions/cardstream)](https://pypi.org/project/cardstream/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/Ximilar-com/cardstream/actions/workflows/ci.yml/badge.svg)](https://github.com/Ximilar-com/cardstream/actions/workflows/ci.yml)

<!-- Replace with a ~15s screen recording of the browser UI identifying cards.
     GitHub caps media at 10 MB. Prefer a GIF: MP4 does not render on PyPI,
     which serves this same README as the project description. -->
<img src="docs/media/demo.gif" alt="cardstream identifying cards live" width="720">

</div>

---

No capture button, no per-frame API spam: **one paid API call per distinct
card, not per frame.** All the vision runs on your machine; a single cropped
JPEG leaves it each time a genuinely new card appears.

## Quick start

```bash
uv tool install "cardstream[client,onnx]"   # or: pip install "cardstream[client,onnx]"
export XIMILAR_API_KEY=...                  # the one environment variable anything here reads

cardstream-web                              # browser UI on your webcam → http://127.0.0.1:8001
```

Hold a card up. The overlay shows the state machine thinking and the match when
it lands.

You will also need the model weights — they are too large to ship in the wheel.
The [installer](#install) fetches them for you; a source checkout needs them
dropped into `model/` (see **[Model weights](#model-weights)**).

Headless, for rigs and streams:

```bash
cardstream-client --source 0                              # webcam, results to the terminal
cardstream-web --source rtsp://user:pw@cam/stream1        # pull a camera
cardstream-web --source rtmp://0.0.0.0:1935/live --listen # OBS pushes here
```

## Why it is cheap

Identification is a paid call of roughly half a second, so cardstream never
makes one per frame. A per-card **state machine** driven by free, entirely
local signals decides when a fresh call is actually warranted:

```
            ┌─────────────────────── frame ───────────────────────┐
            ▼                                                      │
  1. detection (throttled)    card bbox + deskewed crop            │
            │  no card → EMPTY                                     │
            ▼                                                      │
  2. MotionGate               mean frame-diff; low for N frames → SETTLED
            │  still moving → MOVING                               │
            ▼                                                      │
  3. identity gate            pHash Hamming / embedding cosine → NEW card?
            │  same card → IDENTIFIED (re-emit cached, 0 calls)    │
            ▼                                                      │
  4. identify (async/thread, debounced)  → IDENTIFIED + result ────┘
```

Detection itself is throttled in three tiers by the free motion gate (moving /
static-with-card / empty heartbeat). The net effect is **one identify call per
distinct card presentation, regardless of frame rate**. Hold a card still and
it costs nothing more. Swap cards and it costs exactly one call.

| Stage | Where it runs | Cost |
|---|---|---|
| Frame capture | browser webcam, or a pulled source | free |
| Motion gate | local (mean frame-diff) | free |
| Card location | local: detection (RF-DETR / RT-DETRv2) or segmentation (RF-DETR) | free |
| Same-card identity gate | local (embedding cosine, or pHash) | free |
| Identify lookup | Ximilar `collectibles/v2/*_id` | **paid, once per distinct card** |

Everything above the last row is free and local. Nothing but that one crop
leaves the process — there is no service in the path.

## Model weights

`model/` is gitignored, so a fresh checkout has no weights and the CLI says so
at startup until you supply them. The installer and the Docker image fetch them
for you.

| Weight | Flag | Required |
| --- | --- | --- |
| **segmentation** — finds the card's outline, so the crop is deskewed and cut tight | `--segmentor-model` | **yes — the default locator** |
| **similarity** — the embedding the identity gate compares crops with | `--embed-model` | **yes by default** (`--gate phash` needs none) |
| detection — finds a bounding box instead | `--detector-model` | no — opt-in, replaces the segmentor |
| tracking — carries the box between detections | `--tracker-model` | no — opt-in |

Full details, download table and each export's I/O contract:
**[`model/README.md`](model/README.md)** and the
[model zoo](https://cardstream.ai/models).

## Install

**The quick way** (Linux/macOS — a venv in `~/.cardstream`, shims in
`~/.local/bin`, model weights included):

```bash
curl -fsSL https://raw.githubusercontent.com/Ximilar-com/cardstream/main/scripts/install.sh | sh
cardstream-web --version
```

**From PyPI.** Extras are independent and ML imports are lazy, so any subset
installs — but locating a card always needs a model, so pick a backend:

```bash
pip install "cardstream[client,onnx]"    # ONNX Runtime — what the shipped weights use
pip install "cardstream[client,torch]"   # torch/transformers backends instead
```

**From a checkout:**

```bash
git clone https://github.com/Ximilar-com/cardstream && cd cardstream
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[client,onnx]' --group dev
```

or in one step with the smoke checks: `./scripts/build-from-source.sh`
(`--models` also fetches the weights).

**Docker.** Read the security header in the `Dockerfile` before publishing any
port:

```bash
docker build -t cardstream .
docker run --rm -e XIMILAR_API_KEY -p 127.0.0.1:8001:8001 -v cardstream-models:/models cardstream
```

Releases carry the wheel, sdist and `SHA256SUMS`, which `install.sh` verifies.

## Common recipes

```bash
cardstream-web --game "One Piece" --alphabet latin  # prefill: faster, more precise
cardstream-web --set-code PBL                       # restrict matching to one set
cardstream-web --camera-width 3840                  # more pixels to identify from
cardstream-web --store-images crops/                # keep every crop you paid for
cardstream-web --show-detection --debug             # see what was located, and why
cardstream-web --gate phash                         # zero-ML identity gate
cardstream-client --source card.jpg --loop          # a still image, on repeat
```

**Set `--alphabet` whenever you set `--game`.** Prefilling the game switches
off the endpoint's own writing-system detection, and it then assumes latin — so
a Japanese card comes back as its English print. Leave both unset and the
endpoint works them out itself, which is why neither is defaulted.

The shipped defaults **are** the tuned configuration: `make prod` passes no
flags at all and runs the same pipeline `make dev` spells out flag by flag.

## Documentation

| | |
| --- | --- |
| **[CLI reference](docs/cli-reference.md)** | every flag and its default, generated from the parser |
| **[Tuning](docs/tuning.md)** | call economics, detection filters, resolution, the settings dialog |
| **[Locating the card](docs/locators.md)** | detection vs segmentation, expansion, visual tracking |
| **[Frame sources](docs/sources.md)** | webcams, files, RTSP/RTMP/SRT, `--listen` |
| **[Model weights](model/README.md)** | the model zoo and each export's contract |
| **[Architecture](CLAUDE.md)** | the internals map, for contributors |

## Verify the no-spam behaviour

Run with `--debug` and watch the identify lines: they appear only on a new-card
transition. Hold one card steady → exactly one. Swap to another → one more. Run
at 10–15 fps throughout → still only those. The header's `N calls` badge counts
the same thing. That is the whole point.

## Development

```bash
make venv     # .venv with the client, onnx and dev dependencies
make check    # lint + typecheck + both test suites — what CI runs
make dev      # the tuned run plus diagnostics
```

The suite is entirely offline: no API key, camera, network or model files. See
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Limitations

- **Your Ximilar API key lives on the machine running the client**, and the
  local web app is unauthenticated. That is the direct consequence of having no
  service in the path. `--host` defaults to `127.0.0.1` — keep it there. See
  **[SECURITY.md](SECURITY.md)**.
- **Detector boxes are axis-aligned**, so an angled card's crop carries
  background wedges. `--segmentor-model` — the default — is the answer: a mask
  gives four corners and the crop is deskewed and cut tight.
- **One card per frame.** The locator picks the best-scoring candidate.
- **The tracker is presence, not identity.** An in-place card swap keeps its
  score high, so it never replaces the embedding gate's same-vs-new decision.
- **A game prefill costs alphabet detection** — see the warning above.
- `--listen` and `--ffmpeg` need the system `ffmpeg` binary.

## Licence

[Apache-2.0](LICENSE). Every model architecture and runtime weight here is
Apache-2.0 too — see [NOTICE](NOTICE). Identification is performed by the
[Ximilar](https://www.ximilar.com) collectibles API, a paid hosted service.
