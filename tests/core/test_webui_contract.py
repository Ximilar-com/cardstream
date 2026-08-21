"""Pins the few values the browser and the process must both know.

These cross a language boundary, so nothing but a test can hold them together.
Each one is cheap and catches a class of bug that is otherwise invisible until
a dropdown silently sends an unrecognised string.
"""

from __future__ import annotations

import re
from pathlib import Path

from cardstream.client.web_settings import SettingsPatch
from cardstream.core.id_types import NOT_SPECIFIED

WEBUI = Path(__file__).resolve().parents[2] / "src" / "cardstream" / "webui"


def test_not_specified_matches_the_js_copy():
    src = (WEBUI / "shared" / "constants.js").read_text(encoding="utf-8")
    assert f'export const NOT_SPECIFIED = "{NOT_SPECIFIED}";' in src


def test_every_dialog_field_is_a_settings_patch_field():
    """The dialog builds its controls from FIELDS and posts them by key, so a
    key the model does not know would be rejected by extra="forbid" — as a 400
    the user only sees after hitting Save."""
    src = (WEBUI / "smart" / "settings-fields.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'^\s*key: "([a-z_]+)",', src, re.MULTILINE))
    assert keys, "no field descriptors found — did the schema move?"
    assert keys <= set(SettingsPatch.model_fields)


def test_the_dialog_body_is_generated_not_hand_written():
    """index.html holds one mount point; the eight hand-written .setting blocks
    it used to carry are what the schema replaced."""
    html = (WEBUI / "smart" / "index.html").read_text(encoding="utf-8")
    assert '<div id="settings-fields"></div>' in html
    # Only the two page-local knobs (send rate, detection box) keep
    # hand-written markup — they never leave the browser.
    assert len(re.findall(r'class="setting(?: switch)?"', html)) == 2
