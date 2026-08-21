"""Embedder math, factory selection, and the EmbeddingGate (no torch/onnx)."""

from __future__ import annotations

import numpy as np
import pytest

import cardstream.client.embedders as embedders_mod
from _helpers import FakeEmbedder, unit_vec
from cardstream.client.embedders import (
    EmbeddingGate,
    OnnxEmbedder,
    TfliteEmbedder,
    _normalize,
    cosine_similarity,
    make_embedder,
    preprocess,
)


def test_cosine_similarity_basics():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_normalize_returns_unit_vectors():
    v = _normalize(np.array([[3.0, 4.0]]))
    assert v.shape == (2,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)
    # the zero vector must not divide by zero
    z = _normalize(np.zeros(4))
    assert np.all(np.isfinite(z))


def test_preprocess_shape_and_channel_order():
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 0] = 255  # a pure-BLUE BGR crop
    x = preprocess(crop)
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
    # BGR -> RGB swap: the blue channel (index 2 in RGB) must be the hot one.
    assert float(x[0, 2].mean()) > float(x[0, 0].mean())


def test_make_embedder_rejects_bad_extension():
    with pytest.raises(ValueError, match="extension"):
        make_embedder("model.h5")


def test_make_embedder_routes_onnx(monkeypatch):
    session = _FakeOnnxSession(["batch", 3, 224, 224])
    monkeypatch.setattr(
        embedders_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "input", 224),
    )
    assert isinstance(make_embedder("model.onnx"), OnnxEmbedder)


def test_make_embedder_routes_tflite(monkeypatch):
    interp = _FakeTfliteInterpreter([1, 384, 384, 3])
    monkeypatch.setattr(
        embedders_mod,
        "load_tflite_interpreter",
        lambda path: (
            interp,
            interp.get_input_details()[0],
            interp.get_output_details()[0],
        ),
    )
    assert isinstance(make_embedder("model.tflite"), TfliteEmbedder)


# --- OnnxEmbedder layout detection (fake session) ----------------------------


class _FakeOnnxSession:
    """Stands in for an onnxruntime InferenceSession: records the input feed."""

    def __init__(self, input_shape):
        self._input_shape = input_shape
        self.received = None

    def get_inputs(self):
        class _Input:
            name = "input"
            shape = self._input_shape

        return [_Input()]

    def run(self, _outputs, feeds):
        self.received = feeds
        return [np.ones((1, 4), dtype=np.float32)]


def _onnx_embedder(monkeypatch, input_shape, size) -> OnnxEmbedder:
    session = _FakeOnnxSession(input_shape)
    monkeypatch.setattr(
        embedders_mod,
        "load_onnx_session",
        lambda path, fallback_size: (session, "input", size),
    )
    emb = OnnxEmbedder("fake.onnx")
    emb._fake_session = session
    return emb


def test_onnx_embedder_nhwc_gets_raw_pixels(monkeypatch):
    # TF/Keras export: (N, H, W, 3) input, rescaling baked into the graph.
    emb = _onnx_embedder(monkeypatch, ["batch", 384, 384, 3], size=384)
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    vec = emb.embed(crop)
    x = emb._fake_session.received["input"]
    assert x.shape == (1, 384, 384, 3)
    assert float(x.max()) == pytest.approx(200.0)  # raw 0-255, unnormalized
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_onnx_embedder_nchw_gets_imagenet_normalized(monkeypatch):
    emb = _onnx_embedder(monkeypatch, ["batch", 3, 224, 224], size=224)
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    emb.embed(crop)
    x = emb._fake_session.received["input"]
    assert x.shape == (1, 3, 224, 224)
    assert float(np.abs(x).max()) < 5.0  # ImageNet-normalized range


# --- TfliteEmbedder (fake interpreter) ---------------------------------------


class _FakeTfliteInterpreter:
    """Stands in for a LiteRT Interpreter: records the tensor it was fed."""

    def __init__(self, input_shape):
        self._input_shape = input_shape
        self.received = None

    def get_input_details(self):
        return [{"index": 0, "shape": np.array(self._input_shape)}]

    def get_output_details(self):
        return [{"index": 1}]

    def set_tensor(self, _index, value):
        self.received = value

    def invoke(self):
        pass

    def get_tensor(self, _index):
        return np.ones((1, 4), dtype=np.float32)


def _tflite_embedder(monkeypatch, input_shape) -> TfliteEmbedder:
    interp = _FakeTfliteInterpreter(input_shape)
    monkeypatch.setattr(
        embedders_mod,
        "load_tflite_interpreter",
        lambda path: (
            interp,
            interp.get_input_details()[0],
            interp.get_output_details()[0],
        ),
    )
    emb = TfliteEmbedder("fake.tflite")
    emb._fake_interpreter = interp
    return emb


def test_tflite_embedder_nhwc_gets_raw_pixels(monkeypatch):
    emb = _tflite_embedder(monkeypatch, [1, 384, 384, 3])
    crop = np.full((10, 10, 3), 200, dtype=np.uint8)
    vec = emb.embed(crop)
    x = emb._fake_interpreter.received
    assert x.shape == (1, 384, 384, 3)
    assert float(x.max()) == pytest.approx(200.0)  # raw 0-255, unnormalized
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_tflite_embedder_nchw_gets_imagenet_normalized(monkeypatch):
    emb = _tflite_embedder(monkeypatch, [1, 3, 224, 224])
    emb.embed(np.full((10, 10, 3), 200, dtype=np.uint8))
    x = emb._fake_interpreter.received
    assert x.shape == (1, 3, 224, 224)
    assert float(np.abs(x).max()) < 5.0  # ImageNet-normalized range


# --- EmbeddingGate -----------------------------------------------------------


def test_embedding_gate_same_vs_new_card():
    emb = FakeEmbedder()
    gate = EmbeddingGate(emb, similarity_threshold=0.85)
    crop = np.zeros((8, 8, 3), dtype=np.uint8)

    changed, token = gate.decide(crop)
    assert changed is True  # nothing committed yet
    gate.commit(token)

    changed, _ = gate.decide(crop)  # same embedding -> similarity 1.0
    assert changed is False

    emb.embedding = unit_vec(1)  # orthogonal -> similarity 0.0
    changed, _token2 = gate.decide(crop)
    assert changed is True


def test_embedding_gate_emits_debug_lines():
    lines = []
    gate = EmbeddingGate(
        FakeEmbedder(), similarity_threshold=0.85, on_debug=lines.append
    )
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    _, token = gate.decide(crop)
    gate.commit(token)
    gate.decide(crop)
    assert any("cos_sim" in line for line in lines)
