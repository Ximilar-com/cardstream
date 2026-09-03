// The ☀/☾ theme button: which theme wins, what a click does, what the system
// may still change, and that the no-flash script in index.html agrees with
// theme.js on the storage key. No DOM: everything is injected.
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { THEME_KEY, initTheme, resolveTheme } from "../../src/cardstream/webui/shared/theme.js";

// --- the smallest browser the module touches ---------------------------------

function fakeStorage(initial = {}, { broken = false } = {}) {
  const data = { ...initial };
  return {
    getItem: (k) => { if (broken) throw new Error("blocked"); return k in data ? data[k] : null; },
    setItem: (k, v) => { if (broken) throw new Error("blocked"); data[k] = v; },
    data,
  };
}

function fakeMedia(matches) {
  const listeners = [];
  return {
    matches,
    addEventListener: (_type, fn) => listeners.push(fn),
    change(matches) { this.matches = matches; for (const fn of listeners) fn({ matches }); },
  };
}

function fakeButton() {
  const listeners = [];
  return {
    textContent: "", title: "", attrs: {},
    addEventListener: (_type, fn) => listeners.push(fn),
    setAttribute(name, value) { this.attrs[name] = value; },
    click() { for (const fn of listeners) fn(); },
  };
}

function setup({ stored = {}, systemLight = false, broken = false } = {}) {
  const root = { dataset: {} };
  const storage = fakeStorage(stored, { broken });
  const media = fakeMedia(systemLight);
  const button = fakeButton();
  const theme = initTheme(button, { root, storage, media });
  return { root, storage, media, button, theme };
}

// --- the rule ----------------------------------------------------------------

test("a stored choice wins; otherwise the system decides", () => {
  assert.equal(resolveTheme("light", false), "light");
  assert.equal(resolveTheme("dark", true), "dark");
  assert.equal(resolveTheme(null, true), "light");
  assert.equal(resolveTheme(null, false), "dark");
  assert.equal(resolveTheme("sepia", true), "light"); // junk in storage -> system
});

test("first paint follows the system and the button shows the other way", () => {
  const light = setup({ systemLight: true });
  assert.equal(light.root.dataset.theme, "light");
  assert.equal(light.button.textContent, "☾");
  assert.equal(light.button.attrs["aria-label"], "Switch to dark theme");

  const dark = setup({ systemLight: false });
  assert.equal(dark.root.dataset.theme, "dark");
  assert.equal(dark.button.textContent, "☀");
  assert.equal(dark.button.title, "Switch to light theme");
});

test("a click flips the theme and remembers it", () => {
  const { root, storage, button, theme } = setup({ systemLight: false });
  button.click();
  assert.equal(root.dataset.theme, "light");
  assert.equal(storage.data[THEME_KEY], "light");
  assert.equal(theme.current(), "light");
  theme.toggle();
  assert.equal(root.dataset.theme, "dark");
  assert.equal(storage.data[THEME_KEY], "dark");
});

test("the system setting is followed only until the user picks", () => {
  const free = setup({ systemLight: false });
  free.media.change(true);
  assert.equal(free.root.dataset.theme, "light", "no choice stored -> follows");

  const picked = setup({ stored: { [THEME_KEY]: "dark" }, systemLight: false });
  picked.media.change(true);
  assert.equal(picked.root.dataset.theme, "dark", "a stored choice is not overridden");
});

test("blocked storage still gives a working switch", () => {
  const { root, button } = setup({ systemLight: true, broken: true });
  assert.equal(root.dataset.theme, "light");
  button.click();
  assert.equal(root.dataset.theme, "dark");
});

// --- the inline no-flash script and the palette ------------------------------

const WEBUI = new URL("../../src/cardstream/webui/", import.meta.url);

test("index.html applies the theme before first paint with the same key", () => {
  const html = readFileSync(new URL("smart/index.html", WEBUI), "utf8");
  const head = html.slice(0, html.indexOf("</head>"));
  assert.ok(head.includes(`localStorage.getItem("${THEME_KEY}")`), "inline script uses THEME_KEY");
  assert.ok(head.includes("document.documentElement.dataset.theme"), "inline script sets data-theme");
  assert.ok(html.includes('id="theme-toggle"'), "the header carries the button");
});

test("style.css carries the light palette as an override of the same tokens", () => {
  const css = readFileSync(new URL("shared/style.css", WEBUI), "utf8");
  const tokens = (block) => [...block.matchAll(/--([a-z-]+):/g)].map((m) => m[1]).sort();
  const dark = css.match(/:root \{([^}]*)\}/)[1];
  const light = css.match(/:root\[data-theme="light"\] \{([^}]*)\}/)[1];
  assert.deepEqual(tokens(light), tokens(dark), "every token has a light value");
});
