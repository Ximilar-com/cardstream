// The settings dialog, as data.
//
// One descriptor per knob: the dialog's markup, its draft entry, its change
// listener and its save payload are all derived from this list. Adding a knob
// used to mean seven edits across HTML, JS and Python; it is now one entry
// here and one pydantic field in web_settings.py.
//
// Bounds and choices deliberately are NOT here — they come from the process's
// `limits` block, which is read off the model that validates them, so the
// control can never offer a value the server rejects.

import { NOT_SPECIFIED } from "../shared/constants.js";

/** Options a category offers, from the state the process confirmed. */
export function gamesFor(state, category) {
  const entry = (state.categories || []).find((c) => c.id === category);
  return entry ? entry.games : [NOT_SPECIFIED];
}

const named = (names) => names.map((name) => ({ value: name, label: name }));

export const FIELDS = [
  {
    key: "category",
    kind: "select",
    label: "Category",
    hint: "Picks the id endpoint and the record's Category.",
    options: (state) =>
      (state.categories || []).map((c) => ({ value: c.id, label: c.label })),
  },
  {
    key: "game",
    kind: "select",
    label: "Game / Sport",
    hint: "Sent as the record's <code>Subcategory</code>.",
    options: (state, draft) => named(gamesFor(state, draft.category)),
  },
  {
    key: "alphabet",
    kind: "select",
    label: "Alphabet",
    hint:
      "The card's writing system. Left unset the endpoint classifies it — but " +
      "picking a Game switches that classifier off and it then assumes " +
      "<code>latin</code>, so set this whenever a Game is selected.",
    options: (state) => named(state.alphabets || [NOT_SPECIFIED]),
  },
  {
    key: "set_code",
    kind: "text",
    label: "Set code",
    placeholder: "e.g. PBL",
    hint: "Limits matching to one set. Empty = any set.",
  },
  {
    key: "known_attrs",
    kind: "switch",
    label: "Assume front side, upright",
    hint:
      "Sends <code>Side: front</code> + <code>Rotation: rotation_ok</code>. " +
      "Turn off for backs or rotated cards.",
  },
  {
    key: "price_stats",
    kind: "switch",
    label: "Market price statistics",
    hint:
      "Asks for USD price statistics (median, range, latest sale) with every " +
      "match — tcg, sport and comics only; slab has none. Shown on the card " +
      "and in History.",
  },
  {
    key: "result_threshold",
    kind: "range",
    label: "Result threshold",
    hint: "Matches with a larger distance are dropped. 1.00 = keep everything.",
    limits: (state) => state.limits.result_threshold,
    format: (value) => Number(value).toFixed(2),
  },
  {
    key: "camera_width",
    kind: "select",
    label: "Camera width",
    // Meaningless for a pulled source, which the process reads at its own
    // resolution.
    cameraOnly: true,
    hint:
      "Resolution asked of the webcam. Identification crops are cut from " +
      "these pixels, so more is sharper — and costs bandwidth and decode time.",
    options: (state, draft) => {
      const widths = state.limits.camera_widths;
      const values = widths.includes(draft.camera_width)
        ? widths
        : [...widths, draft.camera_width].sort((a, b) => a - b);
      return values.map((w) => ({ value: w, label: `${w} px` }));
    },
  },
];

/** A draft mirroring exactly what the process confirmed.
 *
 * The draft holds the SAME representation the state does — including the
 * literal "Not Specified", which every optional field on the server reads as
 * "leave it out". That is what lets the dirty check below be a plain !==.
 */
export function draftFrom(state) {
  return Object.fromEntries(FIELDS.map((f) => [f.key, state[f.key]]));
}

/** Only what the user actually changed.
 *
 * Sending the whole draft meant every save carried the identify fields, so a
 * process with no identify client rejected a save that only moved the
 * threshold slider.
 */
export function dirtyPatch(draft, state) {
  return Object.fromEntries(
    FIELDS.filter((f) => draft[f.key] !== state[f.key]).map((f) => [f.key, draft[f.key]])
  );
}

/** Switching category can invalidate the selected game — show that before
 *  Save rather than as a surprise after it. */
export function reconcile(state, draft) {
  if (!gamesFor(state, draft.category).includes(draft.game)) {
    draft.game = NOT_SPECIFIED;
  }
  return draft;
}
