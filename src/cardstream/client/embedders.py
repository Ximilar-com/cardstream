"""Card-identity embedders: one interface, pluggable inference backends.

The default identity gate (``--gate embedding``, with ``PhashGate`` as the
cheaper fallback): embed each settled card crop and compare cosine similarity
against the last identified card — below the threshold means a NEW card and
one paid /identify request.

Backends behind one interface, all returning an L2-normalized 1-D float32
vector; ``make_embedder`` picks one from the model-file extension via
``_EXTENSION_BACKENDS`` (add new backends there):

* ``TorchMobileNetV2Embedder`` — torchvision's ImageNet-pretrained MobileNetV2
  by default (1280-d pooled features); pass a ``.pt`` path to load a finetuned
  **TorchScript** model instead (``torch.jit.load``). Needs the ``[torch]``
  extra.
* ``OnnxEmbedder`` — any embedding model exported to ``.onnx``. Needs the
  ``[onnx]`` extra.
* ``TfliteEmbedder`` — any embedding model exported to ``.tflite``. Needs the
  ``[tflite]`` extra.

The ONNX/TFLite backends read the input layout from the model: torch-style
NCHW gets ImageNet-normalized input, TF/Keras-style NHWC gets raw 0-255 RGB
(those exports bake their own rescaling into the graph).

All ML imports are lazy so the base install (``--gate phash``) needs none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from cardstream.client._tflite import load_tflite_interpreter
from cardstream.core._onnx import load_onnx_session
from cardstream.core.engine import IdentityGate
from cardstream.core.imaging import IMAGENET_MEAN, IMAGENET_STD


def preprocess(crop_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """BGR crop -> (1, 3, size, size) float32, RGB, ImageNet-normalized."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None]


def preprocess_raw(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR crop -> (1, size, size, 3) float32, RGB, raw 0-255 — for TF/Keras
    exports whose graphs contain their own rescaling."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32)[None]


def _is_nhwc(shape) -> bool:
    """TF/Keras-style (N, H, W, 3) input vs torch-style (N, 3, H, W)."""
    return len(shape) == 4 and shape[-1] == 3


def _preprocess_for_layout(crop_bgr: np.ndarray, size: int, nhwc: bool) -> np.ndarray:
    return preprocess_raw(crop_bgr, size) if nhwc else preprocess(crop_bgr, size=size)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalized vectors (just the dot product)."""
    return float(np.dot(a, b))


class Embedder(ABC):
    """Embeds a card crop into an L2-normalized identity vector."""

    name = "base"

    @abstractmethod
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray: ...


class TorchMobileNetV2Embedder(Embedder):
    name = "torch-mobilenetv2"

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import torch
            import torchvision
        except ImportError as exc:
            raise RuntimeError(
                "torch/torchvision are not installed — pip install "
                "'cardstream[torch]' (or use an .onnx embed model with "
                "'cardstream[onnx]', or --gate phash for a zero-ML fallback)"
            ) from exc
        self._torch = torch
        if model_path:
            # Finetuned models are expected as TorchScript so they drop in
            # without needing the original model class on the client.
            self._model = torch.jit.load(model_path, map_location="cpu").eval()
        else:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
            backbone = torchvision.models.mobilenet_v2(weights=weights)
            self._model = torch.nn.Sequential(
                backbone.features,
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            ).eval()

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = self._torch.from_numpy(preprocess(crop_bgr))
        with self._torch.no_grad():
            out = self._model(x)
        return _normalize(out.numpy())


class OnnxEmbedder(Embedder):
    name = "onnx"

    def __init__(self, model_path: str) -> None:
        self._session, self._input_name, self._size = load_onnx_session(
            model_path, fallback_size=224
        )
        self._nhwc = _is_nhwc(self._session.get_inputs()[0].shape)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = _preprocess_for_layout(crop_bgr, self._size, self._nhwc)
        out = self._session.run(None, {self._input_name: x})[0]
        return _normalize(np.asarray(out))


class TfliteEmbedder(Embedder):
    name = "tflite"

    def __init__(self, model_path: str) -> None:
        self._interpreter, inp, out = load_tflite_interpreter(model_path)
        self._input_index = inp["index"]
        self._output_index = out["index"]
        shape = list(inp["shape"])
        self._nhwc = _is_nhwc(shape)
        self._size = int(shape[1] if self._nhwc else shape[2])

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = _preprocess_for_layout(crop_bgr, self._size, self._nhwc)
        self._interpreter.set_tensor(self._input_index, x)
        self._interpreter.invoke()
        return _normalize(np.asarray(self._interpreter.get_tensor(self._output_index)))


class EmbeddingGate(IdentityGate):
    """Embedding-cosine identity gate: below the similarity threshold = new card."""

    def __init__(
        self,
        embedder: Embedder,
        similarity_threshold: float,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._on_debug = on_debug
        self._last: np.ndarray | None = None

    def decide(self, crop_bgr: np.ndarray):
        emb = self._embedder.embed(crop_bgr)
        if self._last is None:
            return True, emb
        sim = cosine_similarity(emb, self._last)
        if self._on_debug is not None:
            self._on_debug(f"[gate] cos_sim={sim:.3f} (threshold {self._threshold})")
        return sim < self._threshold, emb

    def commit(self, token) -> None:
        self._last = token

    def reset(self) -> None:
        self._last = None


# Model-file extension -> backend class; new backends register here.
_EXTENSION_BACKENDS: dict[str, type[Embedder]] = {
    ".pt": TorchMobileNetV2Embedder,
    ".onnx": OnnxEmbedder,
    ".tflite": TfliteEmbedder,
}


# The shipped identity-gate embedder, so `--gate embedding` needs no flag.
DEFAULT_EMBED_MODEL = "model/similarity/onnx/model.onnx"


def make_embedder(embed_model: str | None = None) -> Embedder:
    """Backend by extension (see ``_EXTENSION_BACKENDS``); no path ->
    torchvision's pretrained MobileNetV2."""
    if not embed_model:
        return TorchMobileNetV2Embedder()
    backend = _EXTENSION_BACKENDS.get(Path(embed_model).suffix)
    if backend is None:
        supported = " ".join(sorted(_EXTENSION_BACKENDS))
        raise ValueError(
            f"unsupported embed model extension: {embed_model!r} (expected {supported})"
        )
    return backend(embed_model)
