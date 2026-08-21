"""The version surface: cardstream.__version__ and the shared --version flag."""

import argparse
import re

import pytest

import cardstream
from cardstream.client.common import add_version_arg


def test_version_is_a_real_version_string():
    # importlib.metadata resolves the installed dist; the +uninstalled
    # fallback only fires for a bare source tree on PYTHONPATH.
    assert re.match(r"^\d+\.\d+", cardstream.__version__)


def test_version_flag_prints_and_exits_before_anything_loads(capsys):
    ap = argparse.ArgumentParser()
    add_version_arg(ap)
    with pytest.raises(SystemExit) as exc:
        ap.parse_args(["--version"])
    assert exc.value.code == 0
    assert f"cardstream {cardstream.__version__}" in capsys.readouterr().out
