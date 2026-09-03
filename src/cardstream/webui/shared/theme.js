// Light / dark theme: the ☀ / ☾ button in the header.
//
// style.css is dark by default and carries the light palette under
// :root[data-theme="light"]; all this module does is decide which one is on
// and keep the button honest. The user's pick is remembered in localStorage;
// with no pick the page follows the system setting, live. index.html applies
// the same rule inline in <head> so the first paint is already the right
// colour — the storage key there is pinned to THEME_KEY by
// tests/webui/theme.test.js.

export const THEME_KEY = "cardstream-theme";

/** Which theme to show: the stored choice wins; otherwise the system's. */
export function resolveTheme(stored, systemPrefersLight) {
  if (stored === "light" || stored === "dark") return stored;
  return systemPrefersLight ? "light" : "dark";
}

// The button shows where a click takes you, not where you are.
const LABELS = {
  dark: { icon: "☀", label: "Switch to light theme" },
  light: { icon: "☾", label: "Switch to dark theme" },
};

/**
 * Wire the toggle button. Everything the browser provides is injectable so
 * the logic runs under node's test runner without a DOM.
 */
export function initTheme(
  button,
  {
    root = document.documentElement,
    storage = globalThis.localStorage,
    media = globalThis.matchMedia?.("(prefers-color-scheme: light)"),
  } = {},
) {
  // localStorage throws in some contexts (private mode, blocked site data);
  // a theme that cannot be remembered is still a theme.
  const stored = () => {
    try {
      return storage?.getItem(THEME_KEY) ?? null;
    } catch {
      return null;
    }
  };
  const remember = (theme) => {
    try {
      storage?.setItem(THEME_KEY, theme);
    } catch {
      /* not remembered, still applied */
    }
  };

  const apply = (theme) => {
    root.dataset.theme = theme;
    if (button) {
      button.textContent = LABELS[theme].icon;
      button.title = LABELS[theme].label;
      button.setAttribute("aria-label", LABELS[theme].label);
    }
  };
  const current = () => root.dataset.theme;
  const toggle = () => {
    const next = current() === "light" ? "dark" : "light";
    remember(next);
    apply(next);
  };

  apply(resolveTheme(stored(), !!media?.matches));
  button?.addEventListener("click", toggle);
  // Follow the system only while the user has not picked.
  const onSystemChange = (event) => {
    if (stored() === null) apply(resolveTheme(null, event.matches));
  };
  if (media?.addEventListener) media.addEventListener("change", onSystemChange);
  else media?.addListener?.(onSystemChange); // older Safari

  return { current, toggle };
}
