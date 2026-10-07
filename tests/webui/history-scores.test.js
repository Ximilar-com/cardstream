// The match line on a history row: the endpoint's distance and, when the
// process sent it, the locator's confidence in the card (object_confidence).
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeEl, installFakeDom } from "./_fake-dom.js";

installFakeDom();
const { Overlay, formatScores } = await import(
  "../../src/cardstream/webui/shared/overlay.js"
);

function makeOverlay() {
  const history = new FakeEl("ul");
  const els = {
    history,
    historyWrap: new FakeEl("section"),
    overlay: { getContext: () => ({}), clientWidth: 0, clientHeight: 0 },
    state: new FakeEl("span"),
    panel: new FakeEl("div"),
  };
  const overlay = new Overlay(els, () => ({ el: null, fw: 0, fh: 0 }));
  overlay.minCardTimeMs = 0;
  return { overlay, history };
}

const card = (extra = {}) => ({
  full_name: "Charizard", name: "Charizard", set: "Base", card_number: "4",
  distance: 0.254, confidence_tier: "medium", ...extra,
});

const meta = (li) => li.children.find((el) => el.className === "h-meta");

test("the distance alone when the process sent no object confidence", () => {
  assert.equal(formatScores(card()), "dist 0.254");
});

test("the object confidence follows the distance", () => {
  assert.equal(
    formatScores(card({ object_confidence: 0.934 })),
    "dist 0.254 · oconf 0.93",
  );
});

test("a confidence of zero is still a confidence", () => {
  assert.equal(formatScores(card({ object_confidence: 0 })), "dist 0.254 · oconf 0.00");
});

test("a history row carries both on its meta line", () => {
  const { overlay, history } = makeOverlay();
  overlay._addHistory(card({ object_confidence: 0.934 }));
  assert.equal(
    meta(history.children[0]).textContent,
    "Base #4 · dist 0.254 · oconf 0.93",
  );
});
