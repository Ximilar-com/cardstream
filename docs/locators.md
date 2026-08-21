# Locating the card: detection or segmentation

cardstream needs exactly one card locator, and there are two kinds. This page
is the difference between them, what `--detection-expansion` does to each, and
the optional visual tracker that carries the box between detections.

See also: [Tuning](tuning.md) · [`model/README.md`](../model/README.md) for
each weight's export contract

cardstream needs **exactly one card locator**. Segmentation is the default;
detection is the opt-in, and passing `--detector-model` replaces the default
rather than colliding with it:

| | `--detector-model` | `--segmentor-model` |
| --- | --- | --- |
| finds | a bounding box | an instance mask → the card's four corners |
| identify crop | axis-aligned, plus a 4% context margin | **deskewed**, cut tight at the card edge |
| an angled card | crop carries background wedges | crop is the card square-on, nothing else |
| families | `rfdetr`, `rtdetr` | `rfdetr` |
| cost | ~80–150 ms/frame | ~82 ms/frame for the shipped nano, +~1 ms postprocess |

Segmentation is the more precise of the two, and a little slower. It also
deskews the crop the **identity gate** compares, so a card tilting in the hand
keeps its embedding instead of drifting into a second paid call.

Everything else is shared: `--detector-conf`, `--detector-classes`,
`--detect-interval` and the other throttle tiers, the shape filters, the gate
and the call policy all behave identically whichever locator is running.

**`--detection-expansion FRACTION`** (default 0, max 1) grows the located card
before the crop is cut, on every side: `0.1` pushes each edge out by a tenth of
the card, so the crop is 1.2× in each direction. A box is expanded, a
segmentor's four corners are pushed outward from their centre — so a deskewed
crop stays deskewed and simply gains a border of context. Either way the cut is
still made from the ORIGINAL frame, and the result is what
`--store-images-type object` writes.

It changes only the crop that is paid for: the identity gate keeps comparing
the tight crop, and `--show-detection` draws **both** outlines — the card as
located in red, and what actually gets cut and paid for in green. Note
the two locators start from different baselines — a detector's box already
carries a built-in 4% margin, while a segmentor's outline is cut tight, so the
same value gives a little more context on the detector path.

The family comes from `--detector` / `--segmentor`, the runtime from the model
extension. Weights live under `model/detection/` and `model/segmentation/` —
see [`model/README.md`](model/README.md), which documents every folder and its
export contract. The Apache-2.0 RF-DETR and RT-DETRv2
training pipelines that produce them are their own repo (see below):

```bash
cardstream-client --source 0 --detector-model model/detection/model.onnx     # boxes
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx          # corners + deskew
cardstream-web --detector rtdetr --detector-model card_rtdetr.onnx           # RT-DETRv2 ONNX
cardstream-web --detector rtdetr --detector-model card-rtdetr-r18/final      # transformers dir / hub id
cardstream-web --detector rfdetr --detector-model rfdetr-small.onnx          # your own RF-DETR export
cardstream-web --embed-model card_mobilenetv2.pt                             # finetuned TorchScript embedder
```

`--detector-classes` filters by class name (transformers) or integer id
(`.onnx`); leave empty for a single-class card model. `--similarity-threshold`
(default 0.85, tuned for the finetuned embedder) may need retuning for a
different embedder — `--debug` logs the similarities.

## Recommended run

**The shipped defaults are the tuned configuration.** A bare `cardstream-web`
runs the segmentation locator and the embedding gate at the tuned thresholds,
which is exactly what `make prod` does — there is no incantation to copy.

```bash
cardstream-web                                  # the tuned pipeline, no flags
cardstream-web --debug --source testvideo.mp4   # the same, against a file
```

Needs the `[onnx]` extra (`pip install -e '.[onnx]'`) and the weights under
`model/` (gitignored — see [`model/README.md`](model/README.md) for where each
one comes from). `make dev` spells the whole invocation out flag by flag as
living documentation, and a test reads that recipe back to prove it still
matches the defaults.

Swap `--source` for a webcam index or stream URL in production; add
`--tracker-model model/tracking/object_tracking_vittrack_2023sep.onnx` to carry the
bbox between detections (see below).

## Visual tracking between detections

Optionally, an OpenCV **TrackerVit** tracker (~800 KB ONNX, CPU) can carry the
bbox on every frame between detections: after each successful detection the
tracker is (re)seeded from the detector box, the overlay stays glued to the
card even while it moves, and detection drops to a slow re-sync cadence
(`--tracking-detect-interval`, default 2 s) while the tracker stays locked.
When its score falls below `--tracker-score-threshold` (card removed from the
frame) an immediate re-detection fires. Identification behavior is unchanged.

Download `object_tracking_vittrack_2023sep.onnx` from the
[OpenCV zoo](https://huggingface.co/opencv/object_tracking_vittrack) and pass
its path (the model file is gitignored, like all model weights here):

```bash
cardstream-web --tracker-model model/tracking/object_tracking_vittrack_2023sep.onnx
```

**When it pays off.** A tracker update costs ~2.7 ms/frame on CPU (flat — no
spikes), against ~80–150 ms for one RF-DETR / RT-DETRv2 detect. At that ratio
tracking is what **makes CPU real-time viable at all**: without it the
moving-scene detection load exceeds the frame budget.

**The tracker is presence, not identity.** Its score answers "is a card-like
object still where I'm looking?" — swapping a card in place keeps the score at
~0.88 (identical to the same-card control; trackers are trained to be
appearance-robust, so they cannot see swaps). The embedding/pHash identity
gate therefore remains the only thing deciding SAME vs NEW card, and must not
be replaced by tracker continuity.

## Train your own detector

The training pipelines are a **separate repo**, `cardstream/detector/`, checked
out beside this one — they have their own venvs, their own requirements and no
dependency on this package, which is why they do not live here. Both are
Apache-2.0 all the way down (model, weights, training stack), the same hard
requirement every backend here meets.

| Pipeline | Model | Produces | Use it with |
| --- | --- | --- | --- |
| `rtdetrv2/` | RT-DETRv2 | detection | `--detector rtdetr --detector-model` |
| `rfdetr/` | RF-DETR | detection **or** instance segmentation | `--detector rfdetr --detector-model` / `--segmentor-model` |

Each has the same four steps — download from a Ximilar workspace, train, export
ONNX, run it here — and its own README with the data format, device notes and
export contract. See that repo's `CLAUDE.md` for the shape and its hard rules
(separate venvs, always).

`--detector-model` consumes a detection export directly; a segmentation export
goes to `--segmentor-model` for corners and a deskewed crop, and still works as
a plain detector if handed to `--detector-model` instead (its `masks` output is
simply ignored on that path).
