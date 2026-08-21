// Shared UI pieces for both pages: standard element lookups, the card info
// panel, the state badges, and the bbox overlay. One copy for both pages so
// rendering fixes land everywhere at once.

// The DOM ids both pages share (page-specific elements are added by the page).
export function panelEls() {
  return {
    video: document.getElementById("video"),
    overlay: document.getElementById("overlay"),
    conn: document.getElementById("conn"),
    state: document.getElementById("state"),
    fps: document.getElementById("fps"),
    toggle: document.getElementById("toggle"),
    rate: document.getElementById("rate"),
    rateVal: document.getElementById("rate-val"),
    panel: document.getElementById("panel"),
    cardInfo: document.querySelector(".card-info"),
    tier: document.getElementById("tier"),
    distance: document.getElementById("distance"),
    name: document.getElementById("card-name"),
    set: document.getElementById("card-set"),
    number: document.getElementById("card-number"),
    series: document.getElementById("card-series"),
    year: document.getElementById("card-year"),
    links: document.getElementById("card-links"),
    altWrap: document.getElementById("alt-wrap"),
    alternatives: document.getElementById("alternatives"),
    historyWrap: document.getElementById("history-wrap"),
    history: document.getElementById("history"),
  };
}

// "843 ms" below a second, "1.24 s" above — the identify call's wall time.
function formatElapsed(ms) {
  return ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(2) + " s";
}

// How long a card stayed in frame: "12.3 s" under a minute, "2m 05s" above.
function formatDuration(ms) {
  if (ms < 60000) return (ms / 1000).toFixed(1) + " s";
  const total = Math.round(ms / 1000);
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
}

// Renders results: the state badge, the card panel, and the bounding-box
// overlay mapped onto the displayed (object-fit: contain) media.
//
// `getMedia()` is page-specific: it returns { el, fw, fh } — the element
// currently displaying frames and the pixel size of the ANALYSED frame,
// which is the coordinate space bboxes arrive in.
/** A CSS custom property's value, cached — style.css is the palette's only copy. */
const _cssVars = new Map();
function cssVar(name) {
  if (!_cssVars.has(name)) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name);
    _cssVars.set(name, value.trim());
  }
  return _cssVars.get(name);
}

export class Overlay {
  constructor(els, getMedia) {
    this.els = els;
    this.getMedia = getMedia;
    this.ctx = els.overlay.getContext("2d");
    this.lastBox = null;
    this.showBox = true; // pages may disable (smart page: --show-detection)
    this.lastQuad = null; // segmentor corners, when the process sends them
    this.lastCropQuad = null; // where the PAID crop is cut (--detection-expansion)
    // false: a card put back in frame resumes its history row and its clock
    // keeps totalling. true (--split-results): every appearance is its own row.
    this.splitResults = false;
    // Cards briefer than this never reach the history list (--min-card-time).
    // 0 lists everything. Measured as TOTAL time on stream, so a card that
    // comes and goes accumulates towards it in merge mode.
    this.minCardTimeMs = 0;
    this.running = false;
    this._lastHistoryKey = null;
    // The history row of the card currently in frame: its duration keeps
    // counting until the card is lost, then freezes.
    this._openEntry = null;
    // The frozen top row — kept so the same card returning can resume it.
    this._lastEntry = null;
    // When the card now in frame first appeared — the duration counts from
    // here, not from the identification, which lands a beat later.
    this._presentSince = null;
    setInterval(() => this._tickDuration(), 500);
  }

  // The state badge is colour-coded: the point is to read the pipeline at a
  // glance without parsing the word. Unknown values (the "—" of a stopped
  // stream) fall back to the plain badge.
  setState(s) {
    this.els.state.textContent = s;
    const known = ["empty", "moving", "settled", "identifying", "identified"];
    this.els.state.className = known.includes(s) ? `badge state-${s}` : "badge";
  }

  handleResult(res) {
    this.setState(res.state);
    // The identified-push carries no bbox — keep the overlay where it was.
    if (res.bbox !== null || res.state === "empty") {
      this.lastBox = res.bbox;
      // Moves with the box, so it is never left describing a stale position.
      // Null from a box locator, and from a tracker update (which knows the
      // box moved but not where the corners went) — both fall back to a rect.
      this.lastQuad = res.quad ?? null;
      this.lastCropQuad = res.crop_quad ?? null;
    }
    if (res.state !== "empty" && this._presentSince === null) {
      this._presentSince = Date.now();
    }
    if (res.identification) this.renderCard(res.identification);
    if (res.state === "empty") {
      this.els.panel.classList.add("empty");
      // Card lost: freeze its duration. With --split-results the next
      // appearance opens its own row (so each duration describes exactly one
      // appearance); otherwise it resumes this one and keeps totalling.
      this._closeEntry();
      if (this.splitResults) {
        this._lastHistoryKey = null;
        this._lastEntry = null;
      }
      this._presentSince = null;
    }
  }

  renderCard(id) {
    const els = this.els;
    els.panel.classList.remove("empty");
    els.cardInfo.hidden = false;
    els.tier.textContent = id.confidence_tier;
    els.tier.className = "tier " + id.confidence_tier;
    let distText = "dist " + Number(id.distance).toFixed(3);
    if (id.elapsed_ms != null) distText += " · " + formatElapsed(id.elapsed_ms);
    els.distance.textContent = distText;
    els.name.textContent = id.full_name || id.name || "Unknown";
    els.set.textContent = id.set || "—";
    els.number.textContent = id.card_number || "—";
    els.series.textContent = id.series || "—";
    els.year.textContent = id.year || "—";

    els.links.innerHTML = "";
    for (const [name, url] of Object.entries(id.links || {})) {
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = name;
      els.links.appendChild(a);
    }

    const alts = id.alternatives || [];
    els.altWrap.hidden = alts.length === 0;
    els.alternatives.innerHTML = "";
    for (const alt of alts) {
      const li = document.createElement("li");
      li.textContent = alt.full_name || alt.set || "—";
      els.alternatives.appendChild(li);
    }

    this._addHistory(id);
  }

  // Prepend an identification to the history list (newest on top), deduped by
  // CARD identity (not by result): re-identifying the same card — even with a
  // slightly different distance — keeps the existing top entry instead of
  // stacking duplicates. A new row appears only when a different card arrives
  // (or the same one comes back after leaving the frame).
  _addHistory(id) {
    const els = this.els;
    if (!els.history) return; // page has no history section
    const key = [id.full_name, id.name, id.set, id.card_number].join("|");
    if (key === this._lastHistoryKey) {
      // Merge mode: the card that left is back — restart its clock where it
      // stopped instead of adding a row.
      if (!this._openEntry && this._lastEntry) this._resumeEntry();
      return;
    }
    this._lastHistoryKey = key;
    // A card swapped in place (no empty state between the two) starts its own
    // clock now; one arriving into an empty frame counts from when it appeared.
    const startedAt = this._openEntry ? Date.now() : this._presentSince || Date.now();
    this._closeEntry();

    const li = document.createElement("li");
    const time = document.createElement("span");
    time.className = "h-time";
    time.textContent = new Date().toLocaleTimeString([], { hour12: false });
    const tier = document.createElement("span");
    tier.className = "h-tier tier " + id.confidence_tier;
    tier.textContent = id.confidence_tier;
    const name = document.createElement("span");
    name.className = "h-name";
    name.textContent = id.full_name || id.name || "Unknown";
    const meta = document.createElement("span");
    meta.className = "h-meta";
    meta.textContent =
      `${id.set || "—"} #${id.card_number || "—"} · dist ${Number(id.distance).toFixed(3)}`;
    const dur = document.createElement("span");
    dur.className = "h-dur live";
    dur.title = "time the card stayed in frame";
    li.append(time, tier, name, dur, meta);
    // The row is built now but held OUT of the DOM until the card has been on
    // stream for --min-card-time: a card glimpsed mid-swap should never appear
    // at all, rather than flash into the list and be taken back out.
    this._openEntry = { el: dur, startedAt, accumulatedMs: 0, li, shown: false };
    this._tickDuration();  // renders the duration and reveals if already earned
  }

  // Put a held row into the list once its card has earned the time. Idempotent
  // — the 500ms tick, the close and a resume all call it.
  _revealIfEarned(entry) {
    if (!entry || entry.shown) return;
    const total =
      entry.accumulatedMs + (this._openEntry === entry ? Date.now() - entry.startedAt : 0);
    if (total < this.minCardTimeMs) return;
    entry.shown = true;
    this.els.historyWrap.hidden = false;
    this.els.history.prepend(entry.li);
    while (this.els.history.children.length > 50) {
      this.els.history.lastChild.remove();
    }
  }

  // Live count for the card in frame; a no-op once its row is closed. Time
  // spent out of frame is not counted — only the visits are summed.
  _tickDuration() {
    const entry = this._openEntry;
    if (!entry) return;
    entry.el.textContent = formatDuration(
      entry.accumulatedMs + (Date.now() - entry.startedAt)
    );
    this._revealIfEarned(entry);
  }

  // Freeze the open row at the moment the card was lost.
  _closeEntry() {
    const entry = this._openEntry;
    if (!entry) return;
    entry.accumulatedMs += Date.now() - entry.startedAt;
    this._openEntry = null;
    entry.el.textContent = formatDuration(entry.accumulatedMs);
    entry.el.classList.remove("live");
    // The 500ms tick can miss the threshold by up to half a second, so settle
    // it here too — a card that made the time gets its row even if it left
    // between ticks. One that did not simply stays unbuilt.
    this._revealIfEarned(entry);
    this._lastEntry = entry;
  }

  // Merge mode: pick the frozen row back up where its clock stopped.
  _resumeEntry() {
    this._openEntry = { ...this._lastEntry, startedAt: Date.now() };
    this._openEntry.el.classList.add("live");
    this._tickDuration();  // also reveals, if this visit crossed the threshold
  }

  start() {
    this.running = true;
    requestAnimationFrame(() => this._draw());
  }

  stop() {
    this.running = false;
    this.lastBox = null;
    this.lastQuad = null;
    this.lastCropQuad = null;
    this.setState("—");
    // Stopping the stream loses the card as surely as removing it.
    this._closeEntry();
    this._lastHistoryKey = null;
    this._lastEntry = null;
    this._presentSince = null;
  }

  // Map the analysed-frame bbox onto the displayed (object-fit: contain) media.
  _contentRect(m) {
    const cw = m.el.clientWidth;
    const ch = m.el.clientHeight;
    if (!m.fw || !m.fh) return { x: 0, y: 0, w: cw, h: ch };
    const scale = Math.min(cw / m.fw, ch / m.fh);
    const w = m.fw * scale;
    const h = m.fh * scale;
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  }

  // One closed polygon in analysed-frame coords, mapped onto the displayed media.
  _strokeQuad(quad, rect, sx, sy) {
    this.ctx.beginPath();
    quad.forEach(([x, y], i) => {
      const px = rect.x + x * sx;
      const py = rect.y + y * sy;
      if (i === 0) this.ctx.moveTo(px, py);
      else this.ctx.lineTo(px, py);
    });
    this.ctx.closePath();
    this.ctx.stroke();
  }

  _draw() {
    if (!this.running) return;
    const overlay = this.els.overlay;
    const cw = overlay.clientWidth;
    const ch = overlay.clientHeight;
    if (overlay.width !== cw) overlay.width = cw;
    if (overlay.height !== ch) overlay.height = ch;
    this.ctx.clearRect(0, 0, cw, ch);

    const m = this.getMedia();
    if (this.showBox && this.lastBox && m.fw && m.fh) {
      const rect = this._contentRect(m);
      const sx = rect.w / m.fw;
      const sy = rect.h / m.fh;
      this.ctx.lineWidth = 3;
      // What the model LOCATED. Red rather than the confidence tier: --high is
      // green, which would be indistinguishable from the crop outline below.
      // The tier is still on its own badge and on every history row.
      this.ctx.strokeStyle = cssVar("--danger");
      // Corners when a segmentor found them: that outline IS the shape the crop
      // is cut from, so drawing the hull instead would look looser than the
      // pipeline actually is.
      if (this.lastQuad && this.lastQuad.length === 4) {
        this._strokeQuad(this.lastQuad, rect, sx, sy);
      } else {
        this.ctx.strokeRect(
          rect.x + this.lastBox[0] * sx,
          rect.y + this.lastBox[1] * sy,
          this.lastBox[2] * sx,
          this.lastBox[3] * sy
        );
      }
      // What actually gets PAID FOR, when --detection-expansion makes the two
      // differ. Absent otherwise, so the usual case stays one outline.
      if (this.lastCropQuad && this.lastCropQuad.length === 4) {
        this.ctx.strokeStyle = cssVar("--high");
        this._strokeQuad(this.lastCropQuad, rect, sx, sy);
      }
    }
    requestAnimationFrame(() => this._draw());
  }
}

// The "N fps" badge: tick() per frame, displays and resets once a second.
export class RateBadge {
  constructor(el) {
    this.el = el;
    this.count = 0;
    setInterval(() => {
      this.el.textContent = this.count + " fps";
      this.count = 0;
    }, 1000);
  }

  tick() {
    this.count++;
  }
}
