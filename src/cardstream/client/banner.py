"""The startup banner: the wordmark, then every knob and what it is set to.

Both entrypoints print this before any model loads, so a run's whole
configuration is on screen — and in the log someone pastes into an issue —
instead of having to be reconstructed from the command line. Values that came
from a flag are MARKED and carry the default they replaced: the interesting
lines are the ones you changed, and a threshold that silently stayed at its
default (a dropped ``\\`` in a pasted multi-line command, say) is visible
rather than something you find out about mid-show.

Pure formatting. It takes the parser and the parsed namespace and returns a
string, so it is testable without running either CLI.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from typing import IO, Any

# CARDSTREAM at 59 columns — inside 80 with the two-space indent below.
WORDMARK = r"""
 ███   ███  ████  ████   ████ █████ ████  █████  ███  █   █
█   █ █   █ █   █ █   █ █       █   █   █ █     █   █ ██ ██
█     █████ ████  █   █  ███    █   ████  ████  █████ █ █ █
█   █ █   █ █  █  █   █     █   █   █  █  █     █   █ █   █
 ███  █   █ █   █ ████  ████    █   █   █ █████ █   █ █   █
""".strip("\n")

MARK = "●"  # flag-supplied value; a default gets a blank of the same width

# Never print a credential. The value is confirmed, not shown.
SECRET_DESTS = frozenset({"api_key"})

_BOLD = "\033[1m"
_DIM = "\033[2m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _wants_color(stream: IO[str]) -> bool:
    """Colour only for a real terminal that has not opted out."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # a stream that doesn't implement isatty
        return False


def _format_value(dest: str, value: Any) -> str:
    """One knob's value as a human reads it — never the raw repr."""
    if dest in SECRET_DESTS:
        if value:
            return "set on the command line"
        return (
            "from $XIMILAR_API_KEY" if os.environ.get("XIMILAR_API_KEY") else "not set"
        )
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "on" if value else "off"
    if value == "":
        return "not set"
    return str(value)


def _groups(
    parser: argparse.ArgumentParser,
) -> Iterator[tuple[str, list[argparse.Action]]]:
    """(title, actions) per argparse group, in declaration order.

    Reads ``_action_groups`` — private, but the only way to recover the
    grouping the parser already has, and the alternative (a second list of
    dests kept in this module) is exactly the duplication that goes stale when
    someone adds a flag. Groups the entrypoints never populate are skipped, so
    a parser that uses no groups at all still renders as one flat list.
    """
    for group in getattr(parser, "_action_groups", []):
        actions = [
            a
            for a in group._group_actions
            if a.dest not in (argparse.SUPPRESS, "help", "version")
        ]
        if actions:
            yield group.title or "options", actions


def render_banner(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    version: str,
    subtitle: str = "",
    color: bool = False,
) -> str:
    """The full banner for one run, as a string."""
    bold, dim, blue, reset = (_BOLD, _DIM, _BLUE, _RESET) if color else ("", "", "", "")
    values = vars(args)

    rows: list[tuple[str, list[tuple[bool, str, str, str]]]] = []
    width = 0
    for title, actions in _groups(parser):
        entries = []
        for action in actions:
            if action.dest not in values:
                continue
            # The FIRST spelling, which is the canonical one everywhere here:
            # --store-images-type before its --store_images_type alias,
            # --known-attrs before the --no-known-attrs half of
            # BooleanOptionalAction.
            flag = (
                action.option_strings[0]
                if action.option_strings
                else f"<{action.dest}>"
            )
            value = values[action.dest]
            overridden = value != parser.get_default(action.dest)
            was = (
                f"default: {_format_value(action.dest, parser.get_default(action.dest))}"
                if overridden
                else ""
            )
            entries.append((overridden, flag, _format_value(action.dest, value), was))
            width = max(width, len(flag))
        if entries:
            rows.append((title, entries))

    out: list[str] = [
        "",
        *(f"  {blue}{bold}{line}{reset}" for line in WORDMARK.splitlines()),
        "",
    ]
    head = f"cardstream {version}"
    out.append(
        f"  {bold}{head}{reset}" + (f" {dim}—{reset} {subtitle}" if subtitle else "")
    )
    changed = sum(1 for _, entries in rows for over, *_ in entries if over)
    out.append(
        f"  {dim}{MARK} = set by you; everything else is the built-in default"
        f" ({changed} of {sum(len(e) for _, e in rows)} overridden){reset}"
    )

    for title, entries in rows:
        out.append("")
        out.append(f"  {bold}{title}{reset}")
        for overridden, flag, value, was in entries:
            mark = f"{blue}{MARK}{reset}" if overridden else " "
            shown = f"{bold}{value}{reset}" if overridden else f"{dim}{value}{reset}"
            line = f"  {mark} {flag.ljust(width)}  {shown}"
            if was:
                line += f"  {dim}({was}){reset}"
            out.append(line)
    out.append("")
    return "\n".join(out)


def print_banner(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    version: str,
    subtitle: str = "",
    stream: IO[str] | None = None,
) -> None:
    """Render and write the banner, colouring it only for a terminal."""
    stream = stream if stream is not None else sys.stdout
    print(
        render_banner(
            parser, args, version=version, subtitle=subtitle, color=_wants_color(stream)
        ),
        file=stream,
        # Piped stdout is block-buffered: without this the banner appears only
        # once something else fills the buffer, which for `… | tee show.log`
        # means minutes after the run started (or never, if it crashes first).
        flush=True,
    )
