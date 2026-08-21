// Pure logic behind the settings dialog — no DOM, no bundler, no dependencies.
// Run with:  node --test tests/webui/
//
// Only the parts that can be wrong without anyone noticing are covered: the
// draft/dirty round-trip (whose bug made every save carry the identify fields)
// and the category/game reconciliation.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  FIELDS,
  draftFrom,
  dirtyPatch,
  gamesFor,
  reconcile,
} from "../../src/cardstream/webui/smart/settings-fields.js";
import { NOT_SPECIFIED } from "../../src/cardstream/webui/shared/constants.js";

/** A GET /settings response, as the process sends it. */
function state(overrides = {}) {
  return {
    enabled: true,
    category: "tcg",
    categories: [
      { id: "tcg", label: "Trading Card Game", games: [NOT_SPECIFIED, "Pokémon", "One Piece"] },
      { id: "sport", label: "Sport Card", games: [NOT_SPECIFIED, "Baseball"] },
      { id: "comics", label: "Comics", games: [NOT_SPECIFIED] },
    ],
    game: NOT_SPECIFIED,
    games: [NOT_SPECIFIED, "Pokémon", "One Piece"],
    alphabet: NOT_SPECIFIED,
    alphabets: [NOT_SPECIFIED, "latin", "japanese"],
    set_code: "",
    known_attrs: true,
    result_threshold: 0.8,
    camera_width: 1920,
    send_width: 1920,
    limits: {
      camera_widths: [640, 1280, 1920, 2560, 3840],
      result_threshold: { min: 0, max: 1, step: 0.05 },
    },
    ...overrides,
  };
}

test("every field descriptor is complete", () => {
  for (const f of FIELDS) {
    assert.ok(f.key, "missing key");
    assert.ok(["select", "text", "switch", "range"].includes(f.kind), `bad kind: ${f.kind}`);
    assert.ok(f.label, `${f.key}: missing label`);
    assert.ok(f.hint, `${f.key}: missing hint`);
    // A select without options would render empty and silently post undefined.
    if (f.kind === "select") assert.equal(typeof f.options, "function", f.key);
    if (f.kind === "range") assert.equal(typeof f.limits, "function", f.key);
  }
});

test("the draft mirrors the confirmed state exactly", () => {
  const s = state();
  const draft = draftFrom(s);
  for (const f of FIELDS) assert.deepEqual(draft[f.key], s[f.key]);
  // Mirroring is what makes the dirty check a plain !== — including for the
  // literal "Not Specified", which the server reads as "leave it out".
  assert.deepEqual(dirtyPatch(draft, s), {});
});

test("an untouched dialog saves nothing", () => {
  const s = state();
  assert.deepEqual(dirtyPatch(draftFrom(s), s), {});
});

test("a threshold-only edit carries no identify fields", () => {
  // The bug this fixes: the draft always included category/game/set_code/
  // known_attrs/alphabet, so a process with no identify client rejected a save
  // that moved nothing but the slider.
  const s = state();
  const draft = draftFrom(s);
  draft.result_threshold = 0.5;
  assert.deepEqual(dirtyPatch(draft, s), { result_threshold: 0.5 });
});

test("a camera-width-only edit carries no identify fields", () => {
  const s = state();
  const draft = draftFrom(s);
  draft.camera_width = 1280;
  assert.deepEqual(dirtyPatch(draft, s), { camera_width: 1280 });
});

test("several edits are sent together", () => {
  const s = state();
  const draft = draftFrom(s);
  draft.game = "Pokémon";
  draft.set_code = "PBL";
  draft.known_attrs = false;
  assert.deepEqual(dirtyPatch(draft, s), {
    game: "Pokémon",
    set_code: "PBL",
    known_attrs: false,
  });
});

test("clearing a field is a change, not an absence", () => {
  const s = state({ game: "Pokémon", set_code: "PBL" });
  const draft = draftFrom(s);
  draft.game = NOT_SPECIFIED;
  draft.set_code = "";
  assert.deepEqual(dirtyPatch(draft, s), { game: NOT_SPECIFIED, set_code: "" });
});

test("gamesFor follows the category", () => {
  const s = state();
  assert.ok(gamesFor(s, "tcg").includes("Pokémon"));
  assert.ok(!gamesFor(s, "sport").includes("Pokémon"));
  assert.deepEqual(gamesFor(s, "comics"), [NOT_SPECIFIED]);
  // Unknown category degrades to the safe option rather than throwing.
  assert.deepEqual(gamesFor(s, "nope"), [NOT_SPECIFIED]);
});

test("switching category drops a game the new one does not know", () => {
  const s = state({ game: "Pokémon" });
  const draft = draftFrom(s);
  draft.category = "sport";
  reconcile(s, draft);
  assert.equal(draft.game, NOT_SPECIFIED);
  // The process would drop it anyway; showing it now beats a surprise on Save.
  assert.deepEqual(dirtyPatch(draft, s), { category: "sport", game: NOT_SPECIFIED });
});

test("switching category keeps a game the new one knows", () => {
  const s = state({ game: "Pokémon" });
  const draft = draftFrom(s);
  draft.category = "tcg";
  reconcile(s, draft);
  assert.equal(draft.game, "Pokémon");
});

test("camera width options always include the current value", () => {
  const field = FIELDS.find((f) => f.key === "camera_width");
  const s = state({ camera_width: 1600 });      // not one of the offered widths
  const values = field.options(s, draftFrom(s)).map((o) => o.value);
  assert.ok(values.includes(1600));
  assert.deepEqual(values, [...values].sort((a, b) => a - b));
});

test("the range field takes its bounds from the process, not the page", () => {
  const field = FIELDS.find((f) => f.key === "result_threshold");
  assert.deepEqual(field.limits(state()), { min: 0, max: 1, step: 0.05 });
  assert.equal(field.format(0.8), "0.80");
});
