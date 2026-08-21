// A camera that rejects a live constraint change must not go silent.
//
// The settings dialog calls applyConstraints() on Save. When the track refuses
// it, CameraCapture restarts the stream — and a restart that forgot the send
// schedule left the page connected, the video playing, and nothing at all
// going to the process until the user hit Stop/Start.
//
// Run with:  node --test tests/webui/

import assert from "node:assert/strict";
import { test } from "node:test";

// --- the smallest browser CameraCapture touches ------------------------------

class FakeTrack {
  constructor(accepts) {
    this.accepts = accepts;
    this.stopped = false;
  }
  async applyConstraints() {
    if (!this.accepts) throw new Error("OverconstrainedError");
  }
  stop() {
    this.stopped = true;
  }
}

function install({ accepts }) {
  const tracks = [];
  globalThis.document = {
    createElement: () => ({ getContext: () => ({}) }),
  };
  // navigator is getter-only on globalThis in Node — defineProperty, not assign.
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      mediaDevices: {
        getUserMedia: async () => {
          const track = new FakeTrack(accepts);
          tracks.push(track);
          return { getVideoTracks: () => [track], getTracks: () => [track] };
        },
      },
    },
  });
  const timers = { set: 0, cleared: 0 };
  globalThis.setInterval = () => {
    timers.set += 1;
    return timers.set;
  };
  globalThis.clearInterval = () => {
    timers.cleared += 1;
  };
  return { tracks, timers };
}

const { CameraCapture } = await import("../../src/cardstream/webui/shared/capture.js");

function video() {
  return { srcObject: null, play: async () => {}, videoWidth: 1280, videoHeight: 720 };
}

async function started({ accepts }) {
  const env = install({ accepts });
  const capture = new CameraCapture(video(), { cameraWidth: 1280 });
  await capture.start();
  capture.scheduleSends({ ws: null }, 10);
  return { capture, ...env };
}

test("a camera that accepts the change keeps its stream and its timer", async () => {
  const { capture, tracks, timers } = await started({ accepts: true });
  await capture.applyConstraints(1920);

  assert.equal(tracks.length, 1, "the stream should not have been torn down");
  assert.equal(tracks[0].stopped, false);
  assert.ok(capture.sendTimer, "still sending");
  assert.equal(timers.set, 1, "no reschedule was needed");
});

test("a camera that rejects the change is restarted AND keeps sending", async () => {
  const { capture, tracks, timers } = await started({ accepts: false });
  await capture.applyConstraints(1920);

  assert.equal(tracks.length, 2, "the stream should have been restarted");
  assert.equal(tracks[0].stopped, true, "the old track should be released");
  assert.ok(capture.sendTimer, "the send timer must survive the restart");
  assert.equal(timers.set, 2, "the schedule should have been re-armed exactly once");
});

test("the new resolution is what the restarted stream asks for", async () => {
  const { capture } = await started({ accepts: false });
  await capture.applyConstraints(1920);
  assert.equal(capture.cameraWidth, 1920);
});

test("restarting before any schedule exists arms nothing", async () => {
  const env = install({ accepts: false });
  const capture = new CameraCapture(video(), { cameraWidth: 1280 });
  await capture.start();
  await capture.restart();

  assert.equal(capture.sendTimer, null, "nothing was scheduled, nothing to re-arm");
  assert.equal(env.timers.set, 0);
});
