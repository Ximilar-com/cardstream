"""build_pipeline: flag -> component wiring."""

from __future__ import annotations

import argparse
import re

import pytest

from _helpers import FakeDetector
from cardstream.client import common
from cardstream.client.analyzer import AnalyzerConfig
from cardstream.client.common import (
    add_pipeline_args,
    add_source_args,
    build_pipeline,
    resolve_locator,
)
from cardstream.client.embedders import DEFAULT_EMBED_MODEL
from cardstream.client.ximilar_api import DirectXimilarClient
from cardstream.core.detectors import (
    DEFAULT_DETECTOR,
    DEFAULT_SEGMENTOR,
    DEFAULT_SEGMENTOR_MODEL,
)


def _parse(argv):
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_source="0")
    add_pipeline_args(ap)
    return ap.parse_args(argv)


@pytest.fixture(autouse=True)
def offline_locators(monkeypatch):
    """Neither factory may load a real model — the suite is offline.

    Returns the recorded calls so a test can assert WHICH factory got the
    model path without ever touching onnxruntime.
    """
    seen = {}

    def record(kind):
        def factory(family, model=None, conf=0.5, classes=()):
            seen.update(
                kind=kind, family=family, model=model, conf=conf, classes=classes
            )
            return FakeDetector()

        return factory

    monkeypatch.setattr(common, "make_detector", record("detector"))
    monkeypatch.setattr(common, "make_segmentor", record("segmentor"))
    # The shipped defaults point into the gitignored model/ tree, which a test
    # checkout does not have — the paths are what is under test, not the files.
    monkeypatch.setattr(common, "_require_model", lambda path, flag: None)
    return seen


def test_a_detector_model_builds_a_detector(offline_locators):
    build_pipeline(
        _parse(["--gate", "phash", "--api-key", "k", "--detector-model", "card.onnx"])
    )
    assert offline_locators["kind"] == "detector"
    assert offline_locators["family"] == DEFAULT_DETECTOR == "rfdetr"
    assert offline_locators["model"] == "card.onnx"


def test_a_segmentor_model_builds_a_segmentor(offline_locators):
    """The MODEL PATH is the switch — --segmentor-model alone picks the
    segmentation locator, no second flag required."""
    build_pipeline(
        _parse(["--gate", "phash", "--api-key", "k", "--segmentor-model", "seg.onnx"])
    )
    assert offline_locators["kind"] == "segmentor"
    assert offline_locators["family"] == DEFAULT_SEGMENTOR == "rfdetr"
    assert offline_locators["model"] == "seg.onnx"


def test_the_shared_detection_flags_reach_the_segmentor(offline_locators):
    """--detector-conf / --detector-classes tune whichever locator is in use;
    there is deliberately no --segmentor-conf to keep in sync."""
    build_pipeline(
        _parse(
            [
                "--gate",
                "phash",
                "--api-key",
                "k",
                "--segmentor-model",
                "seg.onnx",
                "--detector-conf",
                "0.7",
                "--detector-classes",
                "0,1",
            ]
        )
    )
    assert offline_locators["conf"] == 0.7
    assert offline_locators["classes"] == ("0", "1")


def test_no_flags_runs_the_shipped_segmentor(offline_locators):
    """A bare `cardstream-web` is the segmentation pipeline — that IS the
    product default, and `make prod` relies on it."""
    build_pipeline(_parse(["--gate", "phash", "--api-key", "k"]))
    assert offline_locators["kind"] == "segmentor"
    assert offline_locators["model"] == DEFAULT_SEGMENTOR_MODEL


def test_an_explicit_detector_replaces_the_defaulted_segmentor(offline_locators):
    """Otherwise asking for a box model would collide with a default nobody
    typed, and --detector-model alone could never work."""
    build_pipeline(
        _parse(["--gate", "phash", "--api-key", "k", "--detector-model", "box.onnx"])
    )
    assert offline_locators["kind"] == "detector"
    assert offline_locators["model"] == "box.onnx"


def test_naming_both_locators_explicitly_is_refused():
    """A defaulted segmentor yields to a detector; two typed paths is a real
    ambiguity and stays an error."""
    args = _parse(
        [
            "--gate",
            "phash",
            "--api-key",
            "k",
            "--detector-model",
            "box.onnx",
            "--segmentor-model",
            "seg.onnx",
        ]
    )
    with pytest.raises(ValueError, match="pick one card locator"):
        build_pipeline(args)


def test_a_missing_model_file_names_the_flag_and_the_fix(tmp_path, monkeypatch):
    """The defaults point into gitignored model/, so a fresh checkout lands
    here — it has to read as 'put the weights there', not as an ORT traceback."""
    monkeypatch.undo()  # the autouse fixture stubs _require_model out
    args = _parse(
        [
            "--gate",
            "phash",
            "--api-key",
            "k",
            "--segmentor-model",
            str(tmp_path / "absent.onnx"),
        ]
    )
    with pytest.raises(ValueError, match=re.escape("model/README.md")):
        build_pipeline(args)


def test_defaults_build_the_ximilar_pipeline_with_phash():
    args = _parse(["--gate", "phash", "--api-key", "k"])
    pipe = build_pipeline(args)
    assert pipe.embedder is None  # phash gate needs none
    assert isinstance(pipe.identify_client, DirectXimilarClient)
    assert "tcg_id" in pipe.description


def test_flags_reach_analyzer_config():
    args = _parse(
        [
            "--gate",
            "phash",
            "--api-key",
            "k",
            "--phash-threshold",
            "7",
            "--cooldown",
            "2.5",
            "--motion-threshold",
            "6.0",
            "--still-frames",
            "5",
            "--detect-interval",
            "0.4",
            "--idle-detect-interval",
            "0.8",
            "--empty-detect-interval",
            "1.6",
            "--debug",
        ]
    )
    cfg = build_pipeline(args).config
    assert cfg.gate == "phash"
    assert cfg.phash_threshold == 7
    assert cfg.cooldown_seconds == 2.5
    assert cfg.motion_threshold == 6.0
    assert cfg.still_frames_required == 5
    assert cfg.detect_interval_seconds == 0.4
    assert cfg.idle_detect_interval_seconds == 0.8
    assert cfg.empty_detect_interval_seconds == 1.6
    assert cfg.debug is True


def test_argparse_defaults_match_analyzer_config():
    """The argparse defaults are read FROM AnalyzerConfig — parsing no flags
    must reproduce it exactly (this pins the single source of truth)."""
    args = _parse(["--gate", "phash", "--api-key", "k"])
    cfg = build_pipeline(args).config
    assert cfg == AnalyzerConfig(gate="phash")


def test_a_missing_api_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("XIMILAR_API_KEY", raising=False)
    args = _parse(["--gate", "phash"])
    with pytest.raises(ValueError, match="API key"):
        build_pipeline(args)


def test_detector_flags_reach_the_namespace():
    """The generic --detector-* spellings are the only ones there are."""
    args = _parse(
        [
            "--detector-model",
            "m.onnx",
            "--detector-conf",
            "0.7",
            "--detector-classes",
            "0,1",
        ]
    )
    assert (args.detector_model, args.detector_conf, args.detector_classes) == (
        "m.onnx",
        0.7,
        "0,1",
    )
    # Neither model path is defaulted by argparse; resolve_locator decides.
    assert args.segmentor_model is None
    assert resolve_locator(args) == ("m.onnx", None)


def test_min_card_size_reaches_the_config():
    args = _parse(["--gate", "phash", "--api-key", "k", "--min-card-size", "0.15"])
    assert build_pipeline(args).config.min_card_fraction == 0.15


def test_the_shape_filters_are_on_by_default():
    """Both ship enabled: a fragment of a card detects at 0.9 and identifies at
    nothing, so paying for it is never what anyone wanted. Pinned because they
    are the two defaults that can drop a REAL card if set wrong — 0.1 of the
    analysed frame in BOTH dimensions is the size floor a card must clear."""
    cfg = AnalyzerConfig()
    assert (cfg.min_card_fraction, cfg.min_card_aspect) == (0.1, 0.4)
    assert _parse([]).min_card_size == 0.1
    assert _parse([]).min_card_aspect_ratio == 0.4


def test_min_card_aspect_ratio_reaches_the_config():
    args = _parse(
        ["--gate", "phash", "--api-key", "k", "--min-card-aspect-ratio", "0.55"]
    )
    assert build_pipeline(args).config.min_card_aspect == 0.55


@pytest.mark.parametrize("bad", ["1.0", "1.5", "-0.1", "half"])
def test_a_min_card_size_outside_0_to_1_is_a_usage_error(bad, capsys):
    """1.0 would reject every detection, so it is refused up front rather than
    leaving someone wondering why nothing is ever identified."""
    with pytest.raises(SystemExit):
        _parse(["--min-card-size", bad])
    assert "min-card-size" in capsys.readouterr().err


def test_retry_unmatched_reaches_the_config():
    args = _parse(["--gate", "phash", "--api-key", "k", "--retry-unmatched", "8"])
    assert build_pipeline(args).config.retry_unmatched_seconds == 8.0


def test_unmatched_cards_are_retried_out_of_the_box():
    """Shipping a non-zero default is a deliberate cost decision — pinned so it
    cannot drift back to 0 (or up) unnoticed."""
    assert AnalyzerConfig().retry_unmatched_seconds == 0.5
    assert _parse([]).retry_unmatched == 0.5


def test_store_images_type_reaches_the_store(tmp_path):
    args = _parse(
        [
            "--gate",
            "phash",
            "--api-key",
            "k",
            "--store-images",
            str(tmp_path),
            "--store-images-type",
            "frame",
        ]
    )
    assert build_pipeline(args).store.kind == "frame"


def test_a_store_type_without_a_folder_is_refused():
    """Otherwise the run writes nothing and says nothing — discovered after the
    show, not during it."""
    args = _parse(["--gate", "phash", "--api-key", "k", "--store-images-type", "frame"])
    with pytest.raises(ValueError, match="needs --store-images"):
        build_pipeline(args)


def test_no_store_flags_means_no_store():
    assert build_pipeline(_parse(["--gate", "phash", "--api-key", "k"])).store is None


def test_detection_expansion_reaches_the_config():
    args = _parse(["--gate", "phash", "--api-key", "k", "--detection-expansion", "0.1"])
    assert build_pipeline(args).config.detection_expansion == 0.1


def test_no_expansion_by_default():
    """Sending exactly what was located is the neutral default — growing the
    crop is a deliberate choice, not something that happens quietly."""
    assert AnalyzerConfig().detection_expansion == 0.0
    assert _parse([]).detection_expansion == 0.0


@pytest.mark.parametrize("bad", ["1.1", "-0.1", "lots"])
def test_a_detection_expansion_outside_0_to_1_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit):
        _parse(["--detection-expansion", bad])
    assert "detection-expansion" in capsys.readouterr().err


def test_detection_expansion_allows_the_full_1_0():
    """Unlike the rejection thresholds, 1.0 is meaningful here (a full extra
    card-width on every side), so the range is inclusive at the top."""
    assert _parse(["--detection-expansion", "1.0"]).detection_expansion == 1.0


# --- the shipped defaults ARE the tuned config -------------------------------
#
# That `make dev` and `make prod` run the same pipeline is pinned by
# tests/test_docs_contract.py, which reads the Makefile recipe itself. The
# version here used to compare argparse against a hand-copied dict of the
# flags, which could not see a Makefile edit at all.


def test_the_shipped_models_are_the_defaults():
    """A bare invocation runs the shipped segmentor and embedder."""
    args = _parse([])
    assert resolve_locator(args) == (None, DEFAULT_SEGMENTOR_MODEL)  # boxes stay opt-in
    assert args.embed_model == DEFAULT_EMBED_MODEL


def test_typing_the_shipped_segmentor_path_still_collides_with_a_detector():
    """The two paths are compared as TYPED, not against a default — spelling
    the shipped segmentor out next to --detector-model is still 'pick one',
    where an argparse default could not tell the two cases apart."""
    args = _parse(
        ["--detector-model", "b.onnx", "--segmentor-model", DEFAULT_SEGMENTOR_MODEL]
    )
    with pytest.raises(ValueError, match="pick one card locator"):
        resolve_locator(args)


def test_no_game_or_alphabet_prefill_by_default():
    """Neither ships set. A Subcategory prefill suppresses the endpoint's own
    Alphabet classification, so a defaulted pair would quietly identify every
    card as if it were the one game — and match a Japanese print against its
    latin twin. Unset lets the endpoint classify both."""
    args = _parse([])
    assert args.game is None and args.alphabet is None
    opts = build_pipeline(_parse(["--api-key", "k"])).identify_client.options
    assert opts.game is None and opts.alphabet is None
