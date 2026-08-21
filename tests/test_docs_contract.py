"""Pins the claims the docs make against the code that has to honour them.

``README.md`` is the product doc and ``CLAUDE.md`` is the map a reader
navigates by, so a wrong number or a path pointing at a deleted directory is a
bug in the same sense a wrong constant is — it just fails on a human instead of
in CI. These tests move that failure into CI.

The same idea as ``tests/core/test_webui_contract.py``: read the artefact, pin
it against the thing it describes. Nothing here imports a model or touches the
network.

Deliberately NOT covered: prose that lists defaults without the word "default"
next to each flag. Only the documented ``(default X)`` convention and the
CLAUDE.md defaults table are machine-checkable, which is a reason to keep
writing them that way.
"""

from __future__ import annotations

import pathlib
import re
import shlex

import pytest

from cardstream.client.common import resolve_locator
from cardstream.client.web_client import build_parser

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Every doc that states a default. The README is the front door and now
# states very few — the reference moved into docs/ so it could stay close to
# the code and be checked exactly like this. A page listed here with NO claims
# is a failure, because the usual cause is that its phrasing drifted out of
# the "(default X)" form this reads.
DOCS = [
    "CLAUDE.md",
    "model/README.md",
    "docs/cli-reference.md",
    "docs/tuning.md",
    "docs/locators.md",
    "docs/sources.md",
]

# Docs that are allowed to state no defaults at all: prose pages whose job is
# to point elsewhere. They are still checked for defaults they DO state.
DOCS_MAY_BE_SILENT = {"README.md", "docs/sources.md"}

ALL_DOCS = [*DOCS, "README.md"]


def _options() -> dict:
    """Every option string ``cardstream-web`` accepts -> its argparse action.

    The web parser is the superset: it carries the page flags on top of every
    shared pipeline flag, so one map covers both entrypoints' documentation.
    """
    return {s: a for a in build_parser()._actions for s in a.option_strings}


def _effective_default(flag: str, action) -> object:
    """What a BARE invocation actually uses for ``flag``.

    Almost always the argparse default. The locator paths are the exception:
    neither is defaulted by argparse so that "typed" stays distinguishable from
    "absent", and ``resolve_locator`` fills the shipped segmentor in — which is
    still what the docs are describing when they name it.
    """
    bare = build_parser().parse_args([])
    detector_model, segmentor_model = resolve_locator(bare)
    return {
        "--detector-model": detector_model,
        "--segmentor-model": segmentor_model,
    }.get(flag, action.default)


def _flat(doc: str) -> str:
    """A doc as one line, so a claim may wrap across lines in the source."""
    return re.sub(r"\s+", " ", (ROOT / doc).read_text(encoding="utf-8"))


# `--flag ARG` ... within 60 chars ... default 0.35 / **5** / `0.2`
_CLAIM = re.compile(
    r"`(--[a-z][a-z0-9-]*)[^`]*`.{0,60}?defaults?\s*\**\s*`?\*{0,2}([0-9]+(?:\.[0-9]+)?)"
)


def _claims(doc: str) -> list[tuple[str, str]]:
    return _CLAIM.findall(_flat(doc))


@pytest.mark.parametrize("doc", ALL_DOCS)
def test_documented_defaults_match_the_parser(doc):
    """Every "(default X)" the docs state is the value argparse actually uses.

    Written after a retuning pass left six wrong numbers across three files —
    including a README that contradicted itself about --min-card-size.
    """
    options = _options()
    claims = _claims(doc)
    if doc not in DOCS_MAY_BE_SILENT:
        assert claims, (
            f"no '(default X)' claims found in {doc} — did the phrasing change?"
        )
    wrong = []
    for flag, stated in claims:
        action = options.get(flag)
        assert action is not None, f"{doc} documents {flag}, which no parser declares"
        actual = _effective_default(flag, action)
        if actual is None or float(stated) != float(actual):
            wrong.append(f"{doc}: {flag} documented as {stated}, actually {actual}")
    assert not wrong, "stale documented defaults:\n  " + "\n  ".join(wrong)


# | `--detector-conf` | 0.35 | |
_TABLE_ROW = re.compile(
    r"^\|\s*`(--[a-z][a-z0-9-]*)`\s*\|\s*`?([^|`]+?)`?\s*\|", re.MULTILINE
)


def test_the_claude_md_defaults_table_matches_the_parser():
    """CLAUDE.md's Defaults table is the one place claiming to list them all."""
    options = _options()
    rows = _TABLE_ROW.findall((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
    assert len(rows) >= 10, "the Defaults table shrank — did its shape change?"
    for flag, stated in rows:
        action = options.get(flag)
        assert action is not None, (
            f"the Defaults table lists {flag}, which no parser declares"
        )
        actual = _effective_default(flag, action)
        try:
            same = float(stated) == float(actual)
        except (TypeError, ValueError):
            same = stated == actual
        assert same, f"{flag}: table says {stated!r}, a bare run uses {actual!r}"


# Directory names the docs use as REPO roots. Spelled out rather than scanned
# off disk, because a reference to a root that no longer exists is exactly the
# failure this catches — `detector/` is listed for that reason, the training
# pipelines having moved to their own repo beside this one.
_TOP_DIRS = ("detector", "docker", "model", "scripts", "src", "tests")
_PATH = re.compile(r"(?<![\w/.])((?:{})/[\w./-]*[\w])".format("|".join(_TOP_DIRS)))


@pytest.mark.parametrize("doc", DOCS)
def test_documented_paths_exist(doc):
    """A path named in the docs is a path a reader will try to open.

    ``model/`` is skipped: those are the weights, gitignored by design and
    documented as such in model/README.md.
    """
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = sorted(
        {
            ref
            for ref in _PATH.findall(text)
            if not ref.startswith("model/") and not (ROOT / ref).exists()
        }
    )
    assert not missing, f"{doc} names paths that do not exist:\n  " + "\n  ".join(
        missing
    )


# --- `make dev` and `make prod` must run the same pipeline --------------------

# The only flags `make dev` may add: they turn diagnostics ON and change no
# analysis behaviour. Anything else differing means the two targets have
# drifted into running different pipelines.
_DIAGNOSTIC_DESTS = {"debug", "show_detection", "store_images", "store_images_type"}

# `make dev` types the shipped segmentor path out; `make prod` lets
# resolve_locator supply it. Compared through that function instead, below.
_RESOLVED_DESTS = {"detector_model", "segmentor_model"}


def _make_recipe(target: str) -> list[str]:
    """The flags a Makefile target passes, as argv."""
    lines, collecting = [], False
    for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if re.match(rf"^{target}:", line):
            collecting = True
            continue
        if collecting:
            if not line.startswith("\t"):
                break
            lines.append(line[1:])
    assert lines, f"no recipe found for `make {target}`"
    tokens = shlex.split(" ".join(lines).replace("\\", " "))
    first_flag = next((i for i, t in enumerate(tokens) if t.startswith("--")), None)
    return tokens[first_flag:] if first_flag is not None else []


def test_make_dev_and_make_prod_run_the_same_pipeline():
    """`make dev` spells out the full invocation; `make prod` passes nothing.

    This reads the Makefile rather than a transcription of it — the previous
    version of this pin compared argparse against a hand-copied dict, so
    editing the Makefile alone could silently split the two targets and leave
    the test green.
    """
    parser = build_parser()
    dev = parser.parse_args(_make_recipe("dev"))
    prod = parser.parse_args(_make_recipe("prod"))
    drifted = {
        dest: (getattr(dev, dest), getattr(prod, dest))
        for dest in vars(prod)
        if dest not in _DIAGNOSTIC_DESTS | _RESOLVED_DESTS
        and getattr(dev, dest) != getattr(prod, dest)
    }
    assert not drifted, (
        "`make dev` differs from `make prod` in more than diagnostics:\n  "
        + "\n  ".join(
            f"{d}: dev={v[0]!r} prod={v[1]!r}" for d, v in sorted(drifted.items())
        )
    )
    assert resolve_locator(dev) == resolve_locator(prod), (
        "`make dev` names a different card locator than `make prod` resolves to"
    )


def test_make_dev_actually_turns_the_diagnostics_on():
    """The other half: dev must still differ from prod where it is meant to."""
    parser = build_parser()
    dev = parser.parse_args(_make_recipe("dev"))
    prod = parser.parse_args(_make_recipe("prod"))
    for dest in _DIAGNOSTIC_DESTS:
        assert getattr(dev, dest) != getattr(prod, dest), (
            f"`make dev` no longer sets {dest}"
        )
