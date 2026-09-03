// --price-stats on the page: the JS twin of core/prices.py (which entries to
// show, how a dollar amount reads), the card panel's price block and the
// history row's one-line summary. tests/core/test_prices.py runs the same
// cases against the Python; a rule that changes there changes here.
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeEl, installFakeDom } from "./_fake-dom.js";

installFakeDom();

const {
  Overlay,
  formatNumber,
  money,
  selectPriceStats,
  formatPriceStats,
  renderPriceStats,
} = await import("../../src/cardstream/webui/shared/overlay.js");

// One entry as the process flattens it: every key present, unknowns null.
const entry = (stats_type, fields) => ({
  stats_type, interval: "overall",
  min: null, max: null, mean: null, median: null, q1: null, q3: null,
  latest: null, oldest: null, latest_date: null, oldest_date: null,
  ...fields,
});
const GRADED = entry("graded", { min: 15, max: 60, median: 24.99, latest: 55, latest_date: "2026-02-22" });
const UNGRADED = entry("ungraded", { min: 3, max: 9.5, median: 4.5 });
const OVERALL = entry("overall", { median: 10 });

const DASH = "–";
const DOT = "·";

// --- the twins of core/prices.py -------------------------------------------

test("money: two decimals, a whole-dollar .00 dropped", () => {
  assert.deepEqual([15, 24.99, 32.5, 0.5, 1234].map(formatNumber), ["15", "24.99", "32.50", "0.50", "1234"]);
  assert.equal(money(15), "$15");
});

test("selection: ungraded then graded; overall only when alone", () => {
  const types = (entries) => selectPriceStats(entries).map((e) => e.stats_type);
  assert.deepEqual(types([GRADED, OVERALL, UNGRADED]), ["ungraded", "graded"]);
  assert.deepEqual(types([OVERALL]), ["overall"]);
  assert.deepEqual(types(undefined), []);
  assert.deepEqual(types([entry("graded", {}), null, 3, { stats_type: "" }]), []); // no median -> nothing
});

test("the summary matches the terminal's line for line", () => {
  assert.equal(
    formatPriceStats([GRADED, UNGRADED]),
    `ungraded $4.50 (3${DASH}9.50) ${DOT} graded $24.99 (15${DASH}60)`,
  );
  assert.equal(formatPriceStats([OVERALL]), "overall $10");
  assert.equal(formatPriceStats([entry("graded", { median: 24.99, min: 15 })]), "graded $24.99");
  assert.equal(formatPriceStats(undefined), "");
});

// --- the card panel ---------------------------------------------------------

test("the card panel lists a row per shown type and hides when empty", () => {
  const el = new FakeEl("ul");
  renderPriceStats(el, [GRADED, UNGRADED]);
  assert.equal(el.hidden, false);
  assert.deepEqual(
    el.children.map((li) => li.children.map((s) => s.textContent)),
    [
      ["ungraded", "$4.50", `(3${DASH}9.50)`],
      ["graded", "$24.99", `(15${DASH}60)`, `latest $55 ${DOT} 2026-02-22`],
    ],
  );
  assert.deepEqual(el.children[1].children.map((s) => s.className), ["p-type", "p-median", "p-range", "p-latest"]);

  renderPriceStats(el, []);
  assert.equal(el.hidden, true);
  assert.equal(el.children.length, 0);
});

test("a latest sale without a date shows the amount alone", () => {
  const el = new FakeEl("ul");
  renderPriceStats(el, [entry("ungraded", { median: 4.5, latest: 5 })]);
  assert.deepEqual(el.children[0].children.map((s) => s.textContent), ["ungraded", "$4.50", "latest $5"]);
});

// --- the history row --------------------------------------------------------

function makeOverlay() {
  const history = new FakeEl("ol");
  const els = {
    history,
    historyWrap: new FakeEl("section"),
    overlay: { getContext: () => ({}), clientWidth: 0, clientHeight: 0 },
    state: new FakeEl("span"),
    panel: new FakeEl("div"),
  };
  const overlay = new Overlay(els, () => ({ el: null, fw: 0, fh: 0 }));
  overlay.minCardTimeMs = 0; // every card lands in the list at once
  return { overlay, history };
}

const card = (name, extra = {}) => ({
  full_name: name, name, set: "Base", card_number: "4",
  distance: 0.1, confidence_tier: "high", ...extra,
});

test("a history row carries the summary only when the match has prices", () => {
  const { overlay, history } = makeOverlay();
  overlay._addHistory(card("Charizard", { price_stats: [GRADED, UNGRADED] }));
  const [row] = history.children;
  assert.equal(row.lastChild.className, "h-price");
  assert.equal(row.lastChild.textContent, `ungraded $4.50 (3${DASH}9.50) ${DOT} graded $24.99 (15${DASH}60)`);

  overlay._addHistory(card("Blastoise")); // a different card, no prices
  assert.equal(history.children.length, 2);
  assert.ok(!history.children[0].children.some((c) => c.className === "h-price"));
});
