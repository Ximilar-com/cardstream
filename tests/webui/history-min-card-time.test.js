// --min-card-time: a card reaches the history list only once it has been on
// stream long enough. The Overlay takes its elements by injection, so this
// needs a handful of DOM stubs rather than a browser.
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeEl, installFakeDom } from "./_fake-dom.js";

installFakeDom();
const { Overlay } = await import("../../src/cardstream/webui/shared/overlay.js");

function makeOverlay(minCardTimeMs) {
  const history = new FakeEl("ul");
  const els = {
    history,
    historyWrap: new FakeEl("section"),
    overlay: { getContext: () => ({}), clientWidth: 0, clientHeight: 0 },
    state: new FakeEl("span"),
    panel: new FakeEl("div"),
  };
  const overlay = new Overlay(els, () => ({ el: null, fw: 0, fh: 0 }));
  overlay.minCardTimeMs = minCardTimeMs;
  return { overlay, history };
}

const card = (name) => ({
  full_name: name, name, set: "Base", card_number: "4",
  distance: 0.1, confidence_tier: "high",
});

// Date.now is the only clock the duration logic reads.
let now = 0;
const realNow = Date.now;
Date.now = () => now;
test.after(() => { Date.now = realNow; });

function seen(history) {
  return history.children.length;
}

// --- the rule -----------------------------------------------------------------

test("a card briefer than the minimum never reaches the history", () => {
  const { overlay, history } = makeOverlay(1000);
  now = 0;
  overlay._addHistory(card("Flicker"));
  assert.equal(seen(history), 0, "held out of the list on arrival");

  now = 400;
  overlay._closeEntry();
  assert.equal(seen(history), 0, "gone before it earned a row");
});

test("a card that stays long enough gets its row", () => {
  const { overlay, history } = makeOverlay(1000);
  now = 0;
  overlay._addHistory(card("Charizard"));
  assert.equal(seen(history), 0);

  now = 1200;
  overlay._tickDuration();               // the 500ms tick crosses the threshold
  assert.equal(seen(history), 1, "revealed while still in frame");

  now = 3000;
  overlay._closeEntry();
  assert.equal(seen(history), 1, "and not duplicated when it leaves");
});

test("leaving between ticks still earns the row", () => {
  // The tick only fires every 500ms, so the close has to settle it too or a
  // card that made the time by 1.1s would be dropped for missing a tick.
  const { overlay, history } = makeOverlay(1000);
  now = 0;
  overlay._addHistory(card("Blastoise"));
  now = 1100;
  overlay._closeEntry();
  assert.equal(seen(history), 1);
});

test("time accumulates across appearances", () => {
  // Merge mode: the same card coming back resumes its clock, so two short
  // visits that together clear the minimum do earn a row.
  const { overlay, history } = makeOverlay(1000);
  now = 0;
  overlay._addHistory(card("Venusaur"));
  now = 600;
  overlay._closeEntry();
  assert.equal(seen(history), 0, "first visit alone is not enough");

  now = 5000;
  overlay._addHistory(card("Venusaur"));  // same key -> resumes
  now = 5600;                             // 600 + 600 = 1200ms total
  overlay._closeEntry();
  assert.equal(seen(history), 1, "the two visits together cleared it");
});

test("zero lists every card", () => {
  const { overlay, history } = makeOverlay(0);
  now = 0;
  overlay._addHistory(card("Instant"));
  assert.equal(seen(history), 1, "no minimum -> in the list immediately");
});

test("a held row does not reveal the history section", () => {
  // Otherwise an empty panel appears for a card that never qualifies.
  const { overlay } = makeOverlay(1000);
  now = 0;
  overlay._addHistory(card("Flicker"));
  assert.equal(overlay.els.historyWrap.hidden, true);
  now = 1500;
  overlay._tickDuration();
  assert.equal(overlay.els.historyWrap.hidden, false);
});
