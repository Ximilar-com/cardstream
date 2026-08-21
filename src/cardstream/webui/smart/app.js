// Browser UI for the smart client. The page only does capture + display:
// frames go over a WebSocket to the LOCAL web_client.py process on the same
// origin, which runs detection + the identity gate and pushes back
// {state, bbox, identification} — same message shape as the streaming server.
//
// Two modes, told apart via GET /mode:
//   camera (default) — this page captures your webcam and pushes frames up.
//   stream           — the process pulls frames from an external source
//                      (--source <url>); this page is a passive viewer:
//                      binary messages are the source frames, JSON messages
//                      are results/logs.
//
// All shared behavior lives in ../shared/ — this file owns only the /mode
// switch, the relayed-frame viewer and the debug log panel.

import { Overlay, RateBadge, panelEls } from "../shared/overlay.js";
import { ReconnectingSocket } from "../shared/ws.js";
import { CameraCapture } from "../shared/capture.js";
import { Session } from "../shared/session.js";
import { initSettings } from "./settings.js";

const els = panelEls();
els.remote = document.getElementById("remote");
els.logWrap = document.getElementById("log-wrap");
els.log = document.getElementById("log");
els.logClear = document.getElementById("log-clear");
els.calls = document.getElementById("calls");
els.camRes = document.getElementById("cam-res");
els.sendRes = document.getElementById("send-res");

let mode = "camera"; // switched to "stream" by GET /mode when --source is a URL

// This JPEG is what the card crop is ultimately cut out of, so it goes at a
// quality worth identifying from and is NOT downscaled to the analysis size.
// It is still capped (--send-width, applied from /settings once it loads):
// encoding runs on the page's main thread, and an uncapped 4K camera spends
// ~49 ms per frame there. The local process downscales its own copy to detect.
const capture = new CameraCapture(els.video, { width: 1920, quality: 0.8 });
const badge = new RateBadge(els.fps);

// The element showing frames and the pixel size of the ANALYSED frame (which
// is what bbox coordinates are in): the capture canvas in camera mode, the
// relayed JPEG's natural size in stream mode.
// Bboxes arrive in the pixel space of the frame the process ANALYSED, which is
// its own downscale of what we sent — so prefer the dimensions it reports and
// fall back to the sent size only until the first snapshot lands.
let analysisSize = null;

function activeMedia() {
  if (mode === "stream") {
    const el = els.remote;
    return {
      el,
      fw: analysisSize ? analysisSize[0] : el.naturalWidth,
      fh: analysisSize ? analysisSize[1] : el.naturalHeight,
    };
  }
  return {
    el: els.video,
    fw: analysisSize ? analysisSize[0] : capture.canvas.width,
    fh: analysisSize ? analysisSize[1] : capture.canvas.height,
  };
}
const overlay = new Overlay(els, activeMedia);
overlay.showBox = false; // hidden by default; /mode enables it (--show-detection)

// Served by web_client.py itself — always connect back to our own origin.
function serverUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

const socket = new ReconnectingSocket({
  urlFn: serverUrl,
  connEl: els.conn,
  isActive: () => session.running,
  onOpen: () => {
    if (mode === "camera") session.scheduleSends();
  },
  onClose: () => capture.clearTimer(),
  onJson: (res) => {
    if (res.log !== undefined) {
      appendLog(res.log);
      return;
    }
    // Paid identify calls, counted by the analyzer this connection owns — so
    // it resets with the session, not with the card.
    if (res.analysis && res.analysis.length === 2) {
      analysisSize = res.analysis;
      setBadge(els.sendRes, "analysed", res.analysis[0], res.analysis[1]);
    }
    if (res.identify_calls !== undefined) {
      els.calls.textContent = `${res.identify_calls} call${res.identify_calls === 1 ? "" : "s"}`;
      els.calls.classList.toggle("badge-on", res.identify_calls > 0);
    }
    overlay.handleResult(res);
  },
  onBinary: (buf) => renderRemoteFrame(buf),
});

// --- settings: the panel's Game select + the ⚙ dialog (category, game, set
// code, known attributes, result threshold, send rate, detection box). The
// identify knobs round-trip through /settings on the local process; the row
// stays hidden when the backend has no identify client (tests).
const settings = initSettings({ overlay, capture });

const session = new Session({
  els,
  capture,
  overlay,
  // Stream mode is a viewer: the process reads the source itself.
  needsCamera: () => mode === "camera",
  // The camera size is known as soon as the stream plays.
  onStart: () => showResolutions(),
  onSend: () => {
    badge.tick();
    showResolutions();
  },
}).attach(socket);

fetch("/mode")
  .then((r) => r.json())
  .then((m) => {
    overlay.showBox = !!m.show_detection;
    overlay.splitResults = !!m.split_results;
    overlay.minCardTimeMs = (m.min_card_time ?? 0) * 1000;
    settings.setShowBox(m.show_detection);
    if (m.mode !== "stream") return;
    mode = "stream";
    els.video.hidden = true;
    els.remote.hidden = false;
    settings.hideRate(); // send rate + camera width are camera-mode knobs
    els.camRes.title = "resolution of the frames arriving from the source";
    session.start(); // viewers auto-connect; the toggle still stops/starts viewing
  })
  .catch(() => {});

// Two different numbers worth seeing side by side: what the camera hands us
// (negotiated from getUserMedia's ideal constraints, NOT the sensor's native
// mode) and what actually reaches the analyzer after CameraCapture downscales
// to its 960 px target. Written only on change — this runs per sent frame.
// "cam" is what the camera negotiated (and what we now send untouched); the
// "analysed" badge is filled in from the process's own snapshots.
function showResolutions() {
  setBadge(els.camRes, "cam", els.video.videoWidth, els.video.videoHeight);
}

function setBadge(el, label, w, h) {
  const text = w && h ? `${label} ${w}×${h}` : `${label} —`;
  if (el.textContent !== text) el.textContent = text;
}

// stream mode: show the latest relayed JPEG in the <img>, recycle blob URLs
let remoteUrl = null;
function renderRemoteFrame(buf) {
  const url = URL.createObjectURL(new Blob([buf], { type: "image/jpeg" }));
  els.remote.onload = () => {
    if (remoteUrl) URL.revokeObjectURL(remoteUrl);
    remoteUrl = url;
  };
  els.remote.src = url;
  // Stream mode: the process already downscaled (--width), so the relayed
  // frame IS the analysed frame — one resolution, not two.
  setBadge(els.camRes, "in", els.remote.naturalWidth, els.remote.naturalHeight);
  badge.tick(); // in stream mode the fps badge counts received frames
}

// --- debug log panel (only receives lines when the process runs with --debug)
const LOG_MAX_LINES = 300;
els.logClear.addEventListener("click", () => (els.log.textContent = ""));

function appendLog(msg) {
  els.logWrap.hidden = false;
  const atBottom =
    els.log.scrollHeight - els.log.scrollTop - els.log.clientHeight < 20;
  const ts = new Date().toLocaleTimeString([], { hour12: false });
  els.log.textContent += `${ts}  ${msg}\n`;
  const lines = els.log.textContent.split("\n");
  if (lines.length > LOG_MAX_LINES) {
    els.log.textContent = lines.slice(lines.length - LOG_MAX_LINES).join("\n");
  }
  if (atBottom) els.log.scrollTop = els.log.scrollHeight; // follow the tail
}
