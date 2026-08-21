// Settings dialog for the smart page.
//
// Two kinds of knobs live here:
//   * process knobs — everything in settings-fields.js. They belong to the
//     LOCAL process, so they round-trip through GET/POST /settings; the
//     response is always the full new state, because one change can move
//     another (switching category drops a game the new endpoint doesn't know).
//   * page-local knobs — the detection-box overlay and the send rate. They
//     never leave the browser, so they apply on the spot; nothing to save.
//
// The dialog edits a DRAFT: controls change the draft, Save posts ONLY the
// fields that differ from the confirmed state, Cancel (or ✕ / Esc) throws it
// away. The panel's Game select is the quick path for the common change — it
// is outside the dialog, so it applies immediately and always shows the SAVED
// game.
//
// Every control in the dialog is generated from FIELDS. The markup, the draft,
// the listeners and the payload all come off that one list.

import { NOT_SPECIFIED } from "../shared/constants.js";
import { FIELDS, draftFrom, dirtyPatch, reconcile } from "./settings-fields.js";

/** Replace a select's options; disable it when there is nothing to choose. */
function fillOptions(select, options, selected) {
  select.replaceChildren();
  for (const { value, label } of options) {
    const opt = document.createElement("option");
    opt.value = String(value);
    opt.textContent = label;
    select.appendChild(opt);
  }
  select.value = String(selected);
  // comics/slab take no subcategory — nothing to choose but "Not Specified".
  select.disabled = options.length <= 1;
}

/** Build one `.setting` block; returns its input element. */
function buildField(field) {
  const wrap = document.createElement("div");
  wrap.className = field.kind === "switch" ? "setting switch" : "setting";
  wrap.id = `setting-${field.key}`;

  const label = document.createElement("label");
  label.htmlFor = `set-${field.key}`;
  label.textContent = field.label;

  let input;
  if (field.kind === "select") {
    input = document.createElement("select");
  } else {
    input = document.createElement("input");
    input.type = { text: "text", switch: "checkbox", range: "range" }[field.kind];
    if (field.placeholder) input.placeholder = field.placeholder;
    if (field.kind === "text") input.autocomplete = "off";
  }
  input.id = `set-${field.key}`;

  if (field.format) {
    // The live read-out sits in the label, e.g. "Result threshold 0.80".
    const value = document.createElement("span");
    value.className = "value";
    value.id = `set-${field.key}-val`;
    label.append(" ", value);
  }

  const hint = document.createElement("p");
  hint.className = "hint";
  hint.innerHTML = field.hint;    // authored copy, not user input

  wrap.append(label, input, hint);
  return { wrap, input, value: wrap.querySelector(".value") };
}

export function initSettings({ overlay, capture = null }) {
  const els = {
    row: document.getElementById("game-row"),
    quick: document.getElementById("game-quick"),
    game: document.getElementById("game"),
    open: document.getElementById("settings-open"),
    close: document.getElementById("settings-close"),
    cancel: document.getElementById("settings-cancel"),
    save: document.getElementById("settings-save"),
    dialog: document.getElementById("settings-dialog"),
    fields: document.getElementById("settings-fields"),
    showBox: document.getElementById("set-show-box"),
    error: document.getElementById("settings-error"),
  };

  let state = null; // last state the process confirmed
  let draft = null; // the dialog's unsaved edits

  // --- build the dialog body once, from the schema
  const controls = new Map();
  for (const field of FIELDS) {
    const built = buildField(field);
    controls.set(field.key, built);
    els.fields.appendChild(built.wrap);

    const commit = () => {
      draft[field.key] = readField(field, built.input);
      // A change can invalidate another field (category -> game), so re-render
      // the whole dialog rather than guessing which siblings moved.
      renderDialog();
    };
    built.input.addEventListener(field.kind === "text" ? "input" : "change", commit);
    if (field.kind === "range") {
      // Ranges also update the read-out as they are dragged.
      built.input.addEventListener("input", () => {
        draft[field.key] = readField(field, built.input);
        if (built.value) built.value.textContent = field.format(draft[field.key]);
      });
    }
  }

  function readField(field, input) {
    if (field.kind === "switch") return input.checked;
    if (field.kind === "range") return parseFloat(input.value);
    if (field.kind === "select" && typeof state[field.key] === "number") {
      return parseInt(input.value, 10);
    }
    return input.value;
  }

  function writeField(field, input, valueEl) {
    const value = draft[field.key];
    if (field.kind === "switch") {
      input.checked = !!value;
    } else if (field.kind === "select") {
      fillOptions(input, field.options(state, draft), value);
    } else {
      if (field.limits) Object.assign(input, field.limits(state));
      input.value = value;
    }
    if (valueEl && field.format) valueEl.textContent = field.format(value);
  }

  // The panel row always mirrors what the process actually has.
  function renderPanel() {
    fillOptions(
      els.game,
      state.games.map((g) => ({ value: g, label: g })),
      state.game
    );
  }

  function renderDialog() {
    reconcile(state, draft);
    for (const field of FIELDS) {
      const { input, value } = controls.get(field.key);
      writeField(field, input, value);
    }
  }

  function showError(message) {
    els.error.textContent = message;
    els.error.hidden = !message;
  }

  async function post(body) {
    const r = await fetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }

  fetch("/settings")
    .then((r) => (r.ok ? r.json() : Promise.reject()))
    .then((s) => {
      state = s;
      // Apply --camera-width before the user ever opens the dialog, so the
      // first getUserMedia already asks for the configured resolution.
      if (capture && s.camera_width) capture.cameraWidth = s.camera_width;
      if (capture && s.send_width !== undefined) capture.width = s.send_width;
      els.row.hidden = false;
      // The Game select is an identify knob — gone without a client. The ⚙ next
      // to it stays: camera width and the result threshold are ours either way.
      els.quick.hidden = !s.enabled;
      if (s.enabled) renderPanel();
    })
    .catch(() => {});

  // --- panel shortcut: one field, applied on the spot
  els.game.addEventListener("change", async () => {
    try {
      state = await post({ game: els.game.value });
    } catch (err) {
      showError(String(err.message || err));
    }
    renderPanel();
  });

  els.save.addEventListener("click", async () => {
    showError("");
    els.save.disabled = true;
    try {
      const before = state.camera_width;
      // Only what changed: a save that touches nothing but the threshold must
      // not carry the identify fields, or a process with no identify client
      // rejects the whole request.
      state = await post(dirtyPatch(draft, state));
      renderPanel();
      // Retune the live camera track so the new resolution takes effect now
      // rather than at the next page load.
      if (capture && state.camera_width !== before) {
        capture.applyConstraints(state.camera_width);
      }
      els.dialog.close();
    } catch (err) {
      showError(String(err.message || err)); // stay open so the value can be fixed
    } finally {
      els.save.disabled = false;
    }
  });

  // Page-local: no draft, no save — the overlay reacts as you toggle it.
  els.showBox.addEventListener("change", () => {
    overlay.showBox = els.showBox.checked;
  });

  els.open.addEventListener("click", () => {
    if (!state) return;
    showError("");
    draft = draftFrom(state);
    renderDialog();
    els.dialog.showModal();
  });

  // Cancel / ✕ / Esc all discard the draft — the panel still shows the saved
  // game, so there is nothing to roll back but the dialog's own controls.
  els.cancel.addEventListener("click", () => els.dialog.close());
  els.close.addEventListener("click", () => els.dialog.close());

  return {
    // /mode decides the initial box visibility (--show-detection) — mirror it
    // into the checkbox once it arrives.
    setShowBox(value) {
      els.showBox.checked = !!value;
    },
    // Camera-only knobs mean nothing to a pulled source: the send rate, plus
    // whichever schema fields declare themselves camera-only.
    hideRate() {
      document.getElementById("rate-setting").hidden = true;
      for (const field of FIELDS.filter((f) => f.cameraOnly)) {
        controls.get(field.key).wrap.hidden = true;
      }
    },
  };
}

export { NOT_SPECIFIED };
