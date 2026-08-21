"""Local card locators — the models that answer "where is the card?".

Four backends behind one interface — a locator answers "where is the card?",
and one of them can also answer "what shape is it?":

* ``RfDetrOnnxDetector`` — an RF-DETR model exported to ``.onnx`` by the
  rfdetr pipeline's ``export_onnx.py``. Requires the ``[onnx]`` extra.
* ``RfDetrSegOnnxDetector`` — the same export from a SEGMENTATION checkpoint,
  decoding its extra ``masks`` output into the card's four corners so the
  identify crop can be deskewed. The ``--segmentor`` path.
* ``RtDetrOnnxDetector`` — an RT-DETRv2 model exported to ``.onnx`` by the
  rtdetrv2 pipeline's ``convert_to_onnx.py``. Requires the ``[onnx]`` extra.
* ``RtDetrTransformersDetector`` — an RT-DETRv2 transformers model (local dir
  or HF hub id). Requires the ``[torch]`` extra.

``make_detector`` / ``make_segmentor`` pick the family (``rfdetr`` or
``rtdetr``) and the model path picks the runtime: ``.onnx`` -> ONNX Runtime,
anything else for rtdetr -> transformers (rfdetr is ``.onnx`` only).

Which factory the CLI calls is decided by which model PATH was given, and
exactly one must be — see ``client/common.py:_locator``.

Every backend here is Apache-2.0 all the way down (model, weights, training
stack) — a hard requirement, since cardstream itself ships as open source.

All ML imports are lazy (inside ``__init__``) so any subset install works and
importing this module never drags torch into a base install. That is what lets
it live in ``core`` alongside the numpy/cv2-only helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import cv2
import numpy as np

from cardstream.core._onnx import load_onnx_session
from cardstream.core.imaging import IMAGENET_MEAN, IMAGENET_STD
from cardstream.core.models import BoundingBox, DetectionResult
from cardstream.core.quad import MIN_CROP_SIDE, mask_to_quad, quad_bbox, warp_quad

# Extra margin (fraction of box size) kept around detector boxes when cropping —
# tcg_id matches better with a little context than with a tight cut.
_CROP_MARGIN = 0.04


class CardDetector(ABC):
    """Finds the dominant card in a frame, or returns None."""

    name = "base"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None: ...


def _detection_with_prob(
    frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, prob: float
) -> DetectionResult | None:
    """The shared result tail of every box detector: crop with margin + prob."""
    det = _crop_with_margin(frame_bgr, x1, y1, x2, y2)
    if det is not None:
        det.prob = float(prob)
    return det


def _best_named_detection(
    frame_bgr: np.ndarray,
    candidates: Iterable[tuple[float, str, Any]],
    allowed: set[str],
) -> DetectionResult | None:
    """Highest-confidence candidate whose class name passes ``allowed``.

    The pick-the-best-box rule for the backend that carries class NAMES (the
    transformers one) — filter, argmax, round to ints, crop. An empty
    ``allowed`` means any class, which is the right default for a single-class
    finetuned card model.
    """
    best: tuple[float, int, int, int, int] | None = None
    for prob, name, xyxy in candidates:
        if allowed and name.lower() not in allowed:
            continue
        if best is None or prob > best[0]:
            # Unpacked rather than starred: names the four-ness the annotation
            # promises, which a generator splat leaves as "some floats".
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            best = (prob, x1, y1, x2, y2)
    if best is None:
        return None
    return _detection_with_prob(frame_bgr, *best[1:], prob=best[0])


def _chw_blob(image_bgr: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB CHW float32 batch scaled to 0..1 — what the RT-DETR
    ONNX export expects (RF-DETR normalizes it further, see ``_imagenet_blob``),
    and four transformations that deserve a name."""
    return image_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


# The shared constants, in the (1, 3, 1, 1) layout a CHW batch broadcasts over.
_IMAGENET_MEAN = IMAGENET_MEAN.reshape(1, 3, 1, 1)
_IMAGENET_STD = IMAGENET_STD.reshape(1, 3, 1, 1)


def _imagenet_blob(image_bgr: np.ndarray) -> np.ndarray:
    """``_chw_blob`` plus the ImageNet mean/std normalize RF-DETR expects."""
    return (_chw_blob(image_bgr) - _IMAGENET_MEAN) / _IMAGENET_STD


def _best_query(
    logits: np.ndarray, class_ids: tuple[int, ...], conf: float
) -> tuple[int, float] | None:
    """The winning QUERY and its score, or None if nothing clears ``conf``.

    Both DETR exports score per class with a sigmoid (focal loss, not softmax)
    and take the max over classes per query, so the index that comes back
    addresses every per-query output — boxes, and a segmentor's masks.
    ``class_ids`` empty means "any class".
    """
    scores = 1.0 / (1.0 + np.exp(-logits))
    if class_ids:
        keep = np.zeros(scores.shape[1], dtype=bool)
        keep[[i for i in class_ids if i < scores.shape[1]]] = True
        scores = np.where(keep[None, :], scores, 0.0)
    confs = scores.max(axis=1)
    q = int(confs.argmax())
    return (q, float(confs[q])) if confs[q] >= conf else None


def _cxcywh_detection(
    frame_bgr: np.ndarray, box: np.ndarray, conf: float
) -> DetectionResult | None:
    """A normalized cxcywh box denormalized to the frame, cropped with margin.

    The one shape both ONNX exports emit, so the arithmetic lives once.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy, bw, bh = box
    return _detection_with_prob(
        frame_bgr,
        int((cx - bw / 2) * w),
        int((cy - bh / 2) * h),
        int((cx + bw / 2) * w),
        int((cy + bh / 2) * h),
        conf,
    )


def _crop_with_margin(
    frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> DetectionResult | None:
    """Clamp a box to the frame, pad it by _CROP_MARGIN, and cut the crop."""
    h, w = frame_bgr.shape[:2]
    mx = int((x2 - x1) * _CROP_MARGIN)
    my = int((y2 - y1) * _CROP_MARGIN)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
        return None
    return DetectionResult(
        bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1), crop=frame_bgr[y1:y2, x1:x2]
    )


class _OnnxBoxDetector(CardDetector):
    """Shared bootstrap for the ONNX box backends.

    They differ in the input tensor they want and the size to assume when the
    export does not declare one — everything else about loading a session and
    holding the confidence/class filter was written out twice.
    """

    _FALLBACK_SIZE = 640

    def __init__(
        self, model_path: str, conf: float = 0.5, class_ids: tuple[int, ...] = ()
    ) -> None:
        self._session, self._input_name, self._size = load_onnx_session(
            model_path, fallback_size=self._FALLBACK_SIZE
        )
        self._conf = conf
        self._class_ids = tuple(class_ids)

    def _run(
        self, frame_bgr: np.ndarray, blob: Callable[[np.ndarray], np.ndarray]
    ) -> list[np.ndarray]:
        """Resize to the export's square input and run one forward pass."""
        resized = cv2.resize(frame_bgr, (self._size, self._size))
        return list(self._session.run(None, {self._input_name: blob(resized)}))


# --- RT-DETRv2 ----------------------------------------------------------------


class RtDetrOnnxDetector(_OnnxBoxDetector):
    """RT-DETRv2 exported to ONNX by the rtdetrv2 pipeline's ``convert_to_onnx.py``.

    The export contract: input ``pixel_values`` (1, 3, S, S), outputs
    ``logits`` (1, Q, C) + ``pred_boxes`` (1, Q, 4, normalized cxcywh).
    RT-DETR preprocessing is a plain resize + 1/255 (no letterbox, no
    mean/std), and DETR-style decoding needs no NMS — boxes are normalized
    to the frame, so denormalizing by the original width/height is exact.
    """

    name = "rtdetr-onnx"
    _FALLBACK_SIZE = 640

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None:
        outputs = self._run(frame_bgr, _chw_blob)

        logits, boxes = outputs[0][0], outputs[1][0]  # (Q, C), (Q, 4)
        if logits.shape[-1] == 4 and boxes.shape[-1] != 4:  # tolerate swapped order
            logits, boxes = boxes, logits
        best = _best_query(logits, self._class_ids, self._conf)
        if best is None:
            return None
        q, conf = best
        return _cxcywh_detection(frame_bgr, boxes[q], conf)


class RfDetrOnnxDetector(_OnnxBoxDetector):
    """RF-DETR exported to ONNX by the rfdetr pipeline's ``export_onnx.py``.

    The export contract: input ``input`` (1, 3, S, S) — plain resize, then
    /255 + ImageNet mean/std (unlike RT-DETR's bare /255) — and outputs
    matched BY NAME: ``dets`` (1, Q, 4) normalized cxcywh and ``labels``
    (1, Q, C+1) whose LAST column is the no-object slot (dropped before the
    per-class sigmoid). A segmentation export adds a ``masks`` output, which
    the name matching ignores, so it works as a detector too. DETR-style
    decoding needs no NMS.
    """

    name = "rfdetr-onnx"
    _FALLBACK_SIZE = 512

    def __init__(
        self, model_path: str, conf: float = 0.5, class_ids: tuple[int, ...] = ()
    ) -> None:
        super().__init__(model_path, conf=conf, class_ids=class_ids)
        # RF-DETR matches its outputs BY NAME (RT-DETR reads them positionally),
        # which is also what lets a segmentation export be told from a
        # detection one at startup rather than per frame.
        self._output_names = [o.name for o in self._session.get_outputs()]

    def _decode(
        self, frame_bgr: np.ndarray
    ) -> tuple[dict[str, np.ndarray], int, float] | None:
        """Run the model and pick the winning query: (outputs by name, q, conf).

        Shared with the segmentation subclass, which needs the same winner but
        reads its ``masks`` output instead of its box. ``q`` is a QUERY index —
        this backend takes the max over classes per query rather than
        flattening Q x C — so it indexes every per-query output directly.
        None when nothing clears ``--detector-conf``.
        """
        by_name = dict(
            zip(self._output_names, self._run(frame_bgr, _imagenet_blob), strict=False)
        )
        logits = self._output(by_name, "labels")[0][:, :-1]  # (Q, C), no-object dropped
        best = _best_query(logits, self._class_ids, self._conf)
        if best is None:
            return None
        return (by_name, *best)

    @staticmethod
    def _output(by_name: dict[str, np.ndarray], name: str) -> np.ndarray:
        """One named output, or a clear error.

        Named lookup with no positional fallback on purpose: the export
        contract above is what makes this backend work, and guessing wrong
        would feed logits in where boxes belong and locate a card nowhere near
        the card.
        """
        try:
            return by_name[name]
        except KeyError:
            raise ValueError(
                f"this ONNX export has no {name!r} output (found: "
                f"{', '.join(by_name) or 'none'}) — it is not an RF-DETR export"
            ) from None

    def _box_detection(
        self, frame_bgr: np.ndarray, by_name: dict[str, np.ndarray], q: int, conf: float
    ) -> DetectionResult | None:
        """The ``dets`` box for query ``q``, denormalized and cropped."""
        boxes = self._output(by_name, "dets")[0]  # (Q, 4) normalized cxcywh
        return _cxcywh_detection(frame_bgr, boxes[q], conf)

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None:
        decoded = self._decode(frame_bgr)
        if decoded is None:
            return None
        return self._box_detection(frame_bgr, *decoded)


class RfDetrSegOnnxDetector(RfDetrOnnxDetector):
    """RF-DETR instance SEGMENTATION exported to ONNX — the ``--segmentor`` path.

    Same export as the detector plus a third output: ``masks`` (1, Q, S/4, S/4)
    of per-query logits. They are not ROI-aligned — each covers the whole input
    plane — so decoding one is a bilinear upsample to the frame and a threshold
    at logit 0 (== sigmoid 0.5), matching the reference decoder the training
    stack ships.

    What that buys: the mask has the card's real BOUNDARY, so
    ``core.quad`` can recover four corners and the identify crop can be
    deskewed instead of cut square. The ``crop`` returned here is likewise
    deskewed, which is what the identity gate compares — a card tilting in the
    hand then keeps the same embedding instead of drifting as the background
    wedges change.

    ``bbox`` stays the quad's axis-aligned hull, so detection filters, the
    tracker and the overlay are unaffected by any of this.
    """

    name = "rfdetr-seg-onnx"

    def __init__(
        self, model_path: str, conf: float = 0.5, class_ids: tuple[int, ...] = ()
    ) -> None:
        super().__init__(model_path, conf=conf, class_ids=class_ids)
        if "masks" not in self._output_names:
            # Fail at startup, not silently per frame: a detection export loads
            # and runs perfectly well here, it just has no boundary to give.
            raise ValueError(
                f"{model_path} has no 'masks' output — it is a detection export, "
                f"not a segmentation one (outputs: {', '.join(self._output_names)}). "
                "Use --detector-model for it, or export a segmentation checkpoint"
            )

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None:
        decoded = self._decode(frame_bgr)
        if decoded is None:
            return None
        by_name, q, conf = decoded

        h, w = frame_bgr.shape[:2]
        # Index the winning query BEFORE resizing — one 78x78 plane upsampled,
        # not a hundred.
        logits = self._output(by_name, "masks")[0][q]
        mask = (
            cv2.resize(
                logits.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            > 0.0
        )

        quad = mask_to_quad(mask)
        crop = warp_quad(frame_bgr, quad) if quad is not None else None
        if quad is None or crop is None:
            # A mask that yields no usable shape is not worth losing the card
            # over — fall back to this model's own box, which is still good.
            return self._box_detection(frame_bgr, by_name, q, conf)
        return DetectionResult(bbox=quad_bbox(quad), crop=crop, prob=conf, quad=quad)


class RtDetrTransformersDetector(CardDetector):
    """RT-DETRv2 via transformers — a finetuned model dir or an HF hub id.

    The image processor handles preprocessing and
    ``post_process_object_detection`` returns absolute xyxy boxes, so this
    backend works for any checkpoint regardless of processor settings.
    """

    name = "rtdetr-transformers"

    def __init__(
        self, model_path: str, conf: float = 0.5, class_names: tuple[str, ...] = ()
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch is not installed — pip install 'cardstream[torch]', "
                "or export the model to .onnx with the rtdetrv2 training pipeline and "
                "install 'cardstream[onnx]'"
            ) from exc
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model = AutoModelForObjectDetection.from_pretrained(model_path)
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._model.to(self._device).eval()
        if self._device == "mps":
            # RT-DETRv2's sinusoidal position embedding computes in float64,
            # which MPS rejects outright — probe once and drop to CPU if so.
            try:
                size = int(self._processor.size.get("height", 640))
                with torch.no_grad():
                    self._model(
                        pixel_values=torch.zeros(1, 3, size, size, device="mps")
                    )
            except Exception:
                self._device = "cpu"
                self._model.to(self._device)
        self._conf = conf
        self._classes = {c.lower() for c in class_names}

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        result = self._processor.post_process_object_detection(
            outputs, threshold=self._conf, target_sizes=[(h, w)]
        )[0]

        id2label = self._model.config.id2label
        return _best_named_detection(
            frame_bgr,
            (
                (float(score), str(id2label.get(int(label), "")), box.tolist())
                for score, label, box in zip(
                    result["scores"], result["labels"], result["boxes"], strict=False
                )
            ),
            self._classes,
        )


# --- factory ------------------------------------------------------------------


def _int_ids(classes: tuple[str, ...]) -> tuple[int, ...]:
    """The numeric subset of a class filter (ONNX exports carry no names)."""
    return tuple(int(c) for c in classes if c.isdigit())


# The default FAMILY for each kind of locator. There is deliberately no default
# MODEL PATH: the path is what selects a detector or a segmentor, so defaulting
# one would silently pick a locator the user never asked for. The CLI applies
# these, NOT this module — the factories take whatever they are given, so a
# family/path mismatch stays the caller's to prevent.
DEFAULT_DETECTOR = "rfdetr"
DEFAULT_SEGMENTOR = "rfdetr"
DEFAULT_DETECTOR_CONF = 0.35
# The shipped segmentor is what a bare `cardstream-web` runs. There is still no
# default DETECTOR model — pointing --detector-model at a box model is the whole
# opt-in, and an explicit one beats this default rather than colliding with it
# (see client/common.py:_locator).
DEFAULT_SEGMENTOR_MODEL = "model/segmentation/onnx/model.onnx"


def make_detector(
    detector: str,
    model: str | None = None,
    conf: float = 0.5,
    classes: tuple[str, ...] = (),
) -> CardDetector:
    detector = detector.lower()
    if detector == "rtdetr":
        if not model:
            raise ValueError(
                "rtdetr detection needs a model path — --detector-model with a "
                ".onnx file, a transformers model dir, or an HF hub id"
            )
        if model.endswith(".onnx"):
            return RtDetrOnnxDetector(model, conf=conf, class_ids=_int_ids(classes))
        return RtDetrTransformersDetector(model, conf=conf, class_names=classes)
    if detector == "rfdetr":
        if not model or not model.endswith(".onnx"):
            raise ValueError(
                "rfdetr detection needs a model path — --detector-model with a "
                ".onnx file from the rfdetr training pipeline's export_onnx.py"
            )
        return RfDetrOnnxDetector(model, conf=conf, class_ids=_int_ids(classes))
    raise ValueError(f"unknown detector {detector!r}; valid: rfdetr, rtdetr")


def make_segmentor(
    segmentor: str,
    model: str | None = None,
    conf: float = 0.5,
    classes: tuple[str, ...] = (),
) -> CardDetector:
    """The segmentation locator — same interface as ``make_detector``.

    Returns a ``CardDetector`` because it is one: a segmentor is a locator that
    happens to also know the card's corners, and everything downstream consumes
    the identical ``DetectionResult``.
    """
    segmentor = segmentor.lower()
    if segmentor == "rfdetr":
        if not model or not model.endswith(".onnx"):
            raise ValueError(
                "rfdetr segmentation needs a model path — --segmentor-model with "
                "a .onnx file exported from an RF-DETR segmentation checkpoint"
            )
        return RfDetrSegOnnxDetector(model, conf=conf, class_ids=_int_ids(classes))
    raise ValueError(f"unknown segmentor {segmentor!r}; valid: rfdetr")
