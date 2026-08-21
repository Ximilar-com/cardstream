"""Shared ONNX Runtime session bootstrap (detectors + embedders)."""

from __future__ import annotations

from typing import Any


def load_onnx_session(model_path: str, fallback_size: int) -> tuple[Any, str, int]:
    """Open an ONNX model on CPU; return ``(session, input_name, input_size)``.

    ``input_size`` comes from the model's static input shape when present,
    else ``fallback_size`` (dynamic-axis exports).
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed — pip install 'cardstream[onnx]'"
        ) from exc
    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    inp = session.get_inputs()[0]
    size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else fallback_size
    return session, inp.name, size
