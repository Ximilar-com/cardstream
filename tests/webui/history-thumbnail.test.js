// History thumbnails: a row shows the crop that was cut from the frame and
// identified, when the process sent one (identification.thumbnail).
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeEl, installFakeDom } from "./_fake-dom.js";

installFakeDom();
const { Overlay } = await import("../../src/cardstream/webui/shared/overlay.js");

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

const THUMB = "data:image/jpeg;base64,/9j/4AAQ";

const card = (name, extra = {}) => ({
  full_name: name, name, set: "Base", card_number: "4",
  distance: 0.1, confidence_tier: "high", ...extra,
});

const thumbs = (li) => li.children.filter((el) => el.tag === "img");

test("a row shows the identified crop as its thumbnail", () => {
  const { overlay, history } = makeOverlay();
  overlay._addHistory(card("Charizard", { thumbnail: THUMB }));
  const [row] = history.children;
  const [img] = thumbs(row);
  assert.equal(img.src, THUMB);
  assert.equal(img.className, "h-thumb");
  assert.equal(row.children[0], img, "first, so it sits in the first column");
});

test("a row without a thumbnail has none", () => {
  const { overlay, history } = makeOverlay();
  overlay._addHistory(card("Pikachu"));
  assert.deepEqual(thumbs(history.children[0]), []);
});

test("the same card identified again keeps its row and its first thumbnail", () => {
  const { overlay, history } = makeOverlay();
  overlay._addHistory(card("Charizard", { thumbnail: THUMB }));
  overlay._addHistory(card("Charizard", { thumbnail: "data:image/jpeg;base64,other" }));
  assert.equal(history.children.length, 1);
  assert.equal(thumbs(history.children[0])[0].src, THUMB);
});
