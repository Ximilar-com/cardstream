# model/

Every weight cardstream loads at runtime, in one place. Nothing here is in git
except this file — model weights don't belong in git history — so a fresh
checkout has an empty `model/` and the CLI says so at startup until you drop
weights in and point a flag at them.

| Folder | What goes in it | Flag | Required |
|---|---|---|---|
| `detection/` | RF-DETR or RT-DETRv2 export — finds the card's **bounding box** | `--detector-model` | no — opt-in, replaces the segmentor |
| `segmentation/` | RF-DETR *instance segmentation* export — finds the card's **outline**, so the identify crop can be deskewed | `--segmentor-model` | **yes, the default locator** |
| `similarity/` | the embedding model the identity gate compares crops with | `--embed-model` | **yes by default** — `--gate phash` needs none |
| `tracking/` | OpenCV zoo vitTracker, carries the bbox between detections | `--tracker-model` | no — opt-in |

```bash
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx \
               --embed-model    model/similarity/onnx/model.onnx
```

---

## The model zoo

Every weight cardstream can load, what it costs, and what it is licensed
under. Downloads are published with each release; the table is the index.

> The installer and the Docker image fetch one tarball per model
> (`cardstream-segmentation-v1.tar.gz`, `cardstream-similarity-v1.tar.gz`)
> from the [v1.0.0 release](https://github.com/Ximilar-com/cardstream/releases/tag/v1.0.0),
> each versioned independently so a retrain republishes one archive without
> touching the others. The `Download` column below points at the same files;
> the `SHA256` column is the tarball's (or the tracker's `.onnx`'s) checksum,
> the one `scripts/install.sh` verifies. The opt-in tracker is fetched straight
> from the OpenCV zoo, pinned by that checksum.

| Model | Task | Architecture | Input | Params/Size | Runtime | Latency (CPU) | Flag | Weights licence | Download | SHA256 |
|---|---|---|---|---|---|---|---|---|---|---|
| `cardstream-seg-nano` | instance segmentation — the card's outline | RF-DETR Seg Nano | 312x312 | 125 MB (.onnx) | onnxruntime | ~82 ms/frame | `--segmentor-model` **(default)** | Apache-2.0 | [cardstream-segmentation-v1.tar.gz](https://github.com/Ximilar-com/cardstream/releases/download/v1.0.0/cardstream-segmentation-v1.tar.gz) | `94af531c4fea1e17fcdf393f5bf1cc257a54c8a9ab1eb402e1e8b37213249fd9` |
| `cardstream-embed-mnv2` | embedding — the same-card identity gate | MobileNetV2 | 384x384 | 9.8 MB (.onnx) | onnxruntime / torch / LiteRT | ~5 ms/crop | `--embed-model` **(default)** | Apache-2.0 | [cardstream-similarity-v1.tar.gz](https://github.com/Ximilar-com/cardstream/releases/download/v1.0.0/cardstream-similarity-v1.tar.gz) | `f45ac9756dd621f021809f3237e5c06ce82377d4d6b1f3b8f6951804cd0369e2` |
| `cardstream-det` | detection — a bounding box | RF-DETR / RT-DETRv2 | varies | — | onnxruntime / transformers | ~80-150 ms/frame | `--detector-model` | Apache-2.0 | not published — export from the detector pipelines | — |
| `vittrack` | visual tracking — carries the box between detections | TrackerVit | 128x128 | 698 KB (.onnx) | OpenCV | ~2.7 ms/frame | `--tracker-model` | Apache-2.0 | [OpenCV zoo](https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx) | `2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d` |

**Two licences, not one.** Each row states the licence of the *weights*. The
*architecture* is separately Apache-2.0 in every case — RF-DETR from Roboflow,
RT-DETRv2 from Baidu, TrackerVit from the OpenCV zoo, MobileNetV2 from Google.
The cardstream weights are trained by Ximilar on Ximilar data and released
under Apache-2.0 as well, which is a distinct grant from the architecture's.
See [`NOTICE`](../NOTICE).

**Latency** is single-frame CPU, measured on an Apple M-series laptop at the
input resolution in the table. It is what the pipeline actually spends per
frame, not a throughput figure from a batched benchmark.

**No accuracy column, deliberately.** These are single-class card locators.
Quoting a COCO mAP would describe the *architecture* on a dataset that has
nothing to do with trading cards, and would flatter or damn these weights at
random. The number that matters — whether a card is identified correctly — is
a property of the Ximilar endpoint, not of the locator.

**Choosing between the two locators** is a decision, not a benchmark:
segmentation gives four corners and a deskewed, tightly cut crop at ~82 ms;
detection gives an axis-aligned box with a 4% context margin at ~80-150 ms.
Segmentation is the default because the deskewed crop also stabilises the
identity gate. See [`docs/locators.md`](../docs/locators.md).

**Verify what you download.** Each published archive ships a `.sha256` beside
it, and `scripts/install.sh`, `docker/entrypoint.sh` and
`scripts/build-from-source.sh` all check it and refuse to unpack on a
mismatch.

---

Everything here is Apache-2.0 end to end (models, weights, training stacks); a
licence-encumbered model does not belong in this folder.

**The installed layout is not this layout.** `scripts/install.sh` and the
Docker image fetch the published per-model tarballs into their own directory
(`~/.cardstream/models`, `/models`) which still uses the flat
`segmentation_model/` / `similarity_model/` / `tracking_model/` names
(`detection_model/` from an older publication; `tracking_model/` holds the
OpenCV-zoo vitTracker, fetched from upstream and pinned by checksum). Each
tarball is versioned independently of the checkout, so the shims those scripts
write resolve the locator from what actually unpacked — the segmentor when it
is there, the box detector otherwise — and pass the paths, so you never type
them. A locator flag of your own suppresses theirs rather than colliding with
it.

---

## detection/ — bounding boxes

Not shipped and not required; build one with the RF-DETR or RT-DETRv2 pipeline
in the sibling `cardstream/detector/` repo and drop the export here.

The contract the `rfdetr` backend expects: input `input` (1, 3, S, S) — plain
resize, `/255`, then ImageNet mean/std — and outputs matched **by name**,
`dets` (1, Q, 4, normalized cxcywh) and `labels` (1, Q, C+1) whose last column
is the no-object slot. `rtdetr` instead takes `pixel_values` with a bare
`/255`, and emits `logits` + `pred_boxes`; its `.onnx` goes to onnxruntime
while a directory or hub id goes to transformers.

```bash
cardstream-web --detector-model model/detection/model.onnx
cardstream-web --detector rtdetr --detector-model model/detection/card_rtdetr.onnx
```

## segmentation/ — the card's outline

An RF-DETR instance segmentation export, single class (`Card`). Where a
detector gives a box, this gives the card's boundary, and cardstream turns that
into four corners and a **deskewed** identify crop: the card square-on and cut
tight at its edge, with none of the background wedges an axis-aligned box drags
in at an angle. More precise than detection, and a little slower.

| File | What it is |
|---|---|
| `onnx/model.onnx` | the export — what `--segmentor-model` points at |
| `model.pth` | the source checkpoint it came from (not loaded at runtime) |
| `model_info.json` | class name, input resolution, task, labels |
| `labels.txt` | one class per line (`Card`) |
| `debug/` | training config + per-epoch metrics |

The export is the detection contract plus a third output:

```
IN   input   [1, 3, S, S]        plain resize, /255, then ImageNet mean/std
OUT  dets    [1, Q, 4]           normalized cxcywh
OUT  labels  [1, Q, C+1]         raw logits; the LAST column is the no-object slot
OUT  masks   [1, Q, S/4, S/4]    per-QUERY mask logits
```

Two things about `masks` are easy to get wrong:

* they are indexed by **query**, not by detection rank — the same index that
  won the `labels` argmax selects the mask;
* they are **not** ROI-aligned. Each plane covers the whole input, so decoding
  one is a plain bilinear upsample to the frame size and a threshold at logit
  `> 0` (== sigmoid > 0.5). There is no box-crop-and-paste step.

The shipped model is `RFDETRSegNano` at S=312 (so 78×78 masks), Q=100, C=1.
`core/quad.py` does the rest: largest external contour → four corners → a
four-point perspective transform.

```bash
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx --debug   # logs "deskewed"
```

`--segmentor-model` defaults to the file above, so a bare `cardstream-web` is
the segmentation pipeline. `--detector-model` is the opt-out to boxes and
replaces it; passing both explicitly is an error. `--detector-conf`,
`--detector-classes`, `--detect-interval` and every other tuning flag apply
unchanged.

Handing this file to `--detector-model` instead also works: the detection path
matches outputs by name and simply ignores `masks`. You get boxes, not corners.

## similarity/ — the identity gate

The embedding model that answers SAME card or NEW card, so a held card costs
one paid call rather than one per frame. The extension picks the runtime:
`.pt` → torch, `.onnx` → onnxruntime, `.tflite` → LiteRT.

```bash
cardstream-web --embed-model model/similarity/onnx/model.onnx
cardstream-web --gate phash     # no embedding model at all
```

`--similarity-threshold` (default 0.85) is tuned for the finetuned embedder and
may need retuning for a different one — `--debug` logs the similarities.

## tracking/ — carrying the bbox between detections

An OpenCV zoo `vitTracker` export. Opt-in: with `--tracker-model` set, a cheap
per-frame update (~2.7 ms) carries the bbox while detection drops to a slow
re-sync tier, and a score drop forces an immediate re-detect.

```bash
cardstream-web --segmentor-model model/segmentation/onnx/model.onnx \
               --tracker-model   model/tracking/object_tracking_vittrack_2023sep.onnx
```

The tracker is **presence, not identity** — an in-place card swap keeps its
score high, so it never replaces the identity gate's SAME-vs-NEW decision.
