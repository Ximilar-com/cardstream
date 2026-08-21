"""Shared TFLite/LiteRT interpreter bootstrap (embedders, future detectors)."""

from __future__ import annotations


def load_tflite_interpreter(model_path: str):
    """Open a ``.tflite`` model; return ``(interpreter, input_detail, output_detail)``.

    Tries the runtimes lightest-first: ``ai-edge-litert`` (the current LiteRT
    package), the legacy ``tflite-runtime``, then full ``tensorflow``.
    """
    interpreter_cls = None
    for importer in (
        lambda: (
            __import__(
                "ai_edge_litert.interpreter", fromlist=["Interpreter"]
            ).Interpreter
        ),
        lambda: (
            __import__(
                "tflite_runtime.interpreter", fromlist=["Interpreter"]
            ).Interpreter
        ),
        lambda: __import__("tensorflow").lite.Interpreter,
    ):
        try:
            interpreter_cls = importer()
            break
        except ImportError:
            continue
    if interpreter_cls is None:
        raise RuntimeError(
            "no TFLite runtime is installed — pip install 'cardstream[tflite]'"
        )
    interpreter = interpreter_cls(model_path=model_path)
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details()[0],
        interpreter.get_output_details()[0],
    )
