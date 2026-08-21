"""The settings + mode endpoints, and the live tuning behind them.

``GET /mode`` tells the page which frame path it is on; ``GET/POST /settings``
is the ⚙ dialog's whole contract: the identify options (which live on the
shared identify target) plus the knobs owned by this process or the browser.

The POST is a PATCH — the page sends only what changed — so every field is
optional and ``exclude_unset`` is what distinguishes "absent" from "cleared".
Validation is the model's job; this module only applies what survives it.
"""

from __future__ import annotations

import weakref
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cardstream.client.analyzer import (
    DEFAULT_ANALYSIS_WIDTH,
    LIVE_FIELDS,
    AnalyzerConfig,
)
from cardstream.client.web_common import JSONResponse
from cardstream.core.id_types import ALPHABETS, ID_TYPES, NOT_SPECIFIED
from cardstream.core.identify_options import IdentifyOptions

# Sane camera widths to offer. Kept here rather than in the page so the list
# the dialog shows and the list the process accepts are the same list.
CAMERA_WIDTH_CHOICES = (640, 1280, 1920, 2560, 3840)
MIN_WIDTH, MAX_WIDTH = 320, 7680


class SettingsPatch(BaseModel):
    """Any subset of the dialog's knobs. Unset fields are left alone.

    ``extra="forbid"`` turns a typo into a 400 instead of a silently ignored
    edit that the page then reports as saved.
    """

    model_config = ConfigDict(extra="forbid")

    # Dialog key -> IdentifyOptions field. "category" is the user-facing word
    # for what the code calls the id type.
    IDENTIFY_FIELDS: ClassVar[dict[str, str]] = {
        "category": "id_type",
        "game": "game",
        "set_code": "set_code",
        "known_attrs": "known_attrs",
        "alphabet": "alphabet",
    }
    # Knobs the browser owns; the process only stores them so they survive a
    # page reload and are readable without an identify client.
    BROWSER_FIELDS: ClassVar[frozenset[str]] = frozenset({"camera_width", "send_width"})

    # Identify options — value-checked by IdentifyOptions.with_(), which knows
    # which games each category accepts.
    category: str | None = None
    game: str | None = None
    set_code: str | None = None
    known_attrs: bool | None = None
    alphabet: str | None = None

    # Analyzer + browser knobs, range-checked here.
    result_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_width: int | None = Field(default=None, ge=MIN_WIDTH, le=MAX_WIDTH)
    send_width: int | None = None

    @field_validator("send_width")
    @classmethod
    def _width_or_as_captured(cls, value: int | None) -> int | None:
        # 0 is the "send whatever the camera gives us" escape hatch, so it sits
        # outside the range rather than inside it.
        if value is not None and value != 0 and not MIN_WIDTH <= value <= MAX_WIDTH:
            raise ValueError(f"must be 0 or {MIN_WIDTH}..{MAX_WIDTH}")
        return value

    def _set(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)

    def identify_patch(self) -> dict[str, Any]:
        """The sent identify fields, keyed for :meth:`IdentifyOptions.with_`."""
        sent = self._set()
        return {
            fld: sent[key] for key, fld in self.IDENTIFY_FIELDS.items() if key in sent
        }

    def tuning_patch(self) -> dict[str, Any]:
        """The sent knobs this process owns (analyzer + browser)."""
        sent = self._set()
        return {
            key: value
            for key, value in sent.items()
            if key in LIVE_FIELDS or key in self.BROWSER_FIELDS
        }


def limits() -> dict[str, Any]:
    """What the dialog is allowed to send — read off the model, so the page's
    controls and the process's validation can never disagree."""
    threshold = SettingsPatch.model_fields["result_threshold"]
    bounds = {m.__class__.__name__: m for m in threshold.metadata}
    return {
        "camera_widths": list(CAMERA_WIDTH_CHOICES),
        "result_threshold": {
            "min": bounds["Ge"].ge,
            "max": bounds["Le"].le,
            "step": 0.05,
        },
    }


class AnalyzerRegistry:
    """Every analyzer in flight, plus the tuning they inherit.

    Camera mode builds one analyzer per browser connection and stream mode one
    per process, so a changed knob has to become both the new default for
    analyzers created later AND an edit to the ones already running. Weak refs:
    a closed connection's analyzer must stay collectable.
    """

    def __init__(
        self,
        make_analyzer,
        result_threshold: float | None = None,
        camera_width: int = 1920,
        send_width: int = 1920,
    ) -> None:
        self._make = make_analyzer
        self._analyzers: weakref.WeakSet = weakref.WeakSet()
        self.result_threshold = (
            AnalyzerConfig().result_threshold
            if result_threshold is None
            else result_threshold
        )
        # Page-side capture knob: what the browser asks getUserMedia for.
        self.camera_width = camera_width
        # Upper bound on what the browser encodes per frame. Encoding is main-
        # thread work in the page (~18 ms at 1080p, ~49 ms at 4K measured), so
        # an uncapped 4K camera makes the UI crawl at 10 fps.
        self.send_width = send_width

    def make(self, on_result, on_log):
        analyzer = self._make(on_result, on_log)
        analyzer.tune(**self._analyzer_fields())
        self._analyzers.add(analyzer)
        return analyzer

    def _analyzer_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in LIVE_FIELDS}

    def apply(self, patch: dict[str, Any]) -> None:
        """Store the patch, then push the analyzer part to everything running."""
        for key, value in patch.items():
            setattr(self, key, value)
        live = {k: v for k, v in patch.items() if k in LIVE_FIELDS}
        if not live:
            return
        for analyzer in list(self._analyzers):
            analyzer.tune(**live)


def add_settings_routes(
    app,
    *,
    live: AnalyzerRegistry,
    identify_client=None,
    source=None,
    show_detection: bool = False,
    split_results: bool = False,
    min_card_time: float = 1.0,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
) -> None:
    """Register ``/mode`` and ``/settings`` on the app."""

    @app.get("/mode")
    async def mode() -> dict:
        # `endpoint` and `source` are for a human curling this route (and for
        # the tests) — the page reads only `mode` and the three page settings.
        return {
            "mode": "stream" if source else "camera",
            "endpoint": source.endpoint if source else None,
            "source": source.name if source else None,
            "show_detection": show_detection,
            "split_results": split_results,
            "min_card_time": min_card_time,
        }

    def settings_state() -> dict:
        """Everything the settings dialog renders — the identify options read
        back off the shared client, plus the live knobs and their limits."""
        # No client (no API key, no server) still renders the dialog: camera
        # width and the result threshold are ours either way.
        opts = identify_client.options if identify_client else IdentifyOptions()
        return {
            "enabled": identify_client is not None,
            "category": opts.id_type.key,
            "categories": [
                {"id": t.key, "label": t.label, "games": t.game_choices}
                for t in ID_TYPES.values()
            ],
            # None = not sent; the page renders that as "Not Specified", the
            # same convention every optional field uses.
            "game": opts.game or NOT_SPECIFIED,
            "games": opts.id_type.game_choices,
            "set_code": opts.set_code or "",
            "alphabet": opts.alphabet or NOT_SPECIFIED,
            "alphabets": [NOT_SPECIFIED, *ALPHABETS],
            "known_attrs": opts.known_attrs,
            "result_threshold": live.result_threshold,
            "camera_width": live.camera_width,
            "send_width": live.send_width,
            "analysis_width": analysis_width,
            "limits": limits(),
        }

    @app.get("/settings")
    async def get_settings() -> dict:
        return settings_state()

    @app.post("/settings")
    async def set_settings(patch: SettingsPatch) -> JSONResponse:
        """Patch any subset of the dialog's knobs; returns the full new state.

        Validate-then-apply, never halfway: the options swap either produces a
        whole new bundle or raises, so a bad game cannot leave a switched
        category behind for the page to show as confirmed.
        """
        identify_patch = patch.identify_patch()
        if identify_patch:
            if identify_client is None:
                return JSONResponse(
                    status_code=400, content={"error": "no identify client"}
                )
            try:
                identify_client.options = identify_client.options.with_(
                    **identify_patch
                )
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        live.apply(patch.tuning_patch())
        return JSONResponse(content=settings_state())
