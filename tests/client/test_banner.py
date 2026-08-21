"""The startup banner: what it marks, what it hides, what it never colours."""

from __future__ import annotations

import argparse
import io

from cardstream.client.banner import MARK, WORDMARK, print_banner, render_banner
from cardstream.client.common import add_pipeline_args, add_source_args


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_source_args(ap, default_source="0")
    add_pipeline_args(ap)
    return ap


def _line_for(text: str, flag: str) -> str:
    """The one rendered line for ``flag`` — matched on the padded column."""
    return next(
        line
        for line in text.splitlines()
        if f" {flag} " in line or line.endswith(f" {flag}")
    )


def _render(argv, **kwargs) -> str:
    ap = _parser()
    return render_banner(ap, ap.parse_args(argv), version="1.2.3", **kwargs)


def test_only_flags_that_were_passed_are_marked():
    text = _render(["--similarity-threshold", "0.97"])
    assert MARK in _line_for(text, "--similarity-threshold")
    assert MARK not in _line_for(text, "--result-threshold")


def test_an_override_carries_the_default_it_replaced():
    """The point of the mark: what the value WAS, not just what it is.

    The replaced value is read from AnalyzerConfig rather than spelled out, so
    retuning a default cannot quietly make this test a lie."""
    from cardstream.client.analyzer import AnalyzerConfig

    was = AnalyzerConfig().cooldown_seconds
    line = _line_for(_render(["--cooldown", "0.4"]), "--cooldown")
    assert "0.4" in line and f"default: {was}" in line


def test_a_flag_passed_at_its_own_default_is_not_marked():
    """Passing --gate embedding changes nothing, so it reads as a default."""
    assert MARK not in _line_for(_render(["--gate", "embedding"]), "--gate")


def test_the_api_key_is_never_printed(monkeypatch):
    monkeypatch.delenv("XIMILAR_API_KEY", raising=False)
    text = _render(["--api-key", "secret-key-value"])
    assert "secret-key-value" not in text
    assert "set on the command line" in _line_for(text, "--api-key")


def test_an_unset_key_names_the_env_var_it_would_come_from(monkeypatch):
    monkeypatch.setenv("XIMILAR_API_KEY", "from-the-environment")
    text = _render([])
    assert "from-the-environment" not in text
    assert "$XIMILAR_API_KEY" in _line_for(text, "--api-key")


def test_boolean_flags_read_as_on_off_under_their_canonical_spelling():
    text = _render(["--debug"])
    assert "on" in _line_for(text, "--debug")
    # BooleanOptionalAction offers --no-known-attrs too; the banner shows the
    # positive spelling, the same one the README and docs use.
    assert "--no-known-attrs" not in text
    assert _line_for(text, "--known-attrs")


def test_every_flag_appears_under_a_group_heading():
    """Grouping comes off the parser, so a new flag lands in a group without
    anyone editing this module — assert the coverage rather than the titles."""
    ap = _parser()
    text = render_banner(ap, ap.parse_args([]), version="1.2.3")
    for action in ap._actions:
        if action.dest in ("help", "version"):
            continue
        assert action.option_strings[0] in text
    for title in ("frame source", "identification", "detection", "tracking"):
        assert title in text


def test_the_wordmark_and_version_lead_the_banner():
    text = _render([], subtitle="headless, local analysis")
    assert WORDMARK.splitlines()[0].strip() in text
    assert "cardstream 1.2.3" in text
    assert "headless, local analysis" in text


def test_a_pipe_gets_no_ansi_escapes():
    """StringIO is not a tty; colour would corrupt a redirected log."""
    buf = io.StringIO()
    ap = _parser()
    print_banner(ap, ap.parse_args(["--debug"]), version="1.2.3", stream=buf)
    assert "\033[" not in buf.getvalue()


def test_color_is_opt_in_and_wraps_the_marked_values():
    plain = _render(["--debug"], color=False)
    colored = _render(["--debug"], color=True)
    assert "\033[" not in plain
    assert "\033[" in colored
