// Browser webcam capture + throttled JPEG push over a WebSocket.

export class CameraCapture {
  // width: downscale target for the frames we send; 0 = send the camera's own
  // resolution (the smart page does this, so identification crops keep their
  // detail — the process downscales for analysis at its end).
  // cameraWidth: what getUserMedia is ASKED for; the camera negotiates the
  // nearest mode it has, which is what video.videoWidth then reports.
  constructor(videoEl, { width = 960, quality = 0.7, cameraWidth = 1280 } = {}) {
    this.video = videoEl;
    this.width = width;
    this.quality = quality;
    this.cameraWidth = cameraWidth;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.stream = null;
    this.sendTimer = null;
    this.schedule = null;    // last scheduleSends() args, so restart() can re-arm
    this.encoding = false;   // one toBlob in flight — see sendFrame
    this.lastBytes = 0;      // size of the last frame sent, for backpressure
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: this.cameraWidth }, facingMode: "environment" },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  // Retune the live track instead of tearing the stream down — the settings
  // dialog changes resolution without a page reload.
  async applyConstraints(cameraWidth) {
    this.cameraWidth = cameraWidth;
    const track = this.stream && this.stream.getVideoTracks()[0];
    if (!track) return;
    try {
      await track.applyConstraints({ width: { ideal: cameraWidth } });
    } catch (err) {
      // Some cameras reject a live change — restart the stream instead.
      await this.restart();
    }
  }

  // Tear the stream down and bring it back with the SAME send schedule.
  // stop() clears the timer, and start() knows nothing about sending, so a
  // plain stop/start pair left the page alive but silently sending nothing.
  async restart() {
    const schedule = this.schedule;
    this.stop();
    await this.start();
    if (schedule) {
      this.scheduleSends(schedule.socket, schedule.fps, schedule.onSent);
    }
  }

  stop() {
    this.clearTimer();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  scheduleSends(socket, fps, onSent = null) {
    this.clearTimer();
    // Remembered, not just used: restart() has to put the same schedule back.
    this.schedule = { socket, fps, onSent };
    this.sendTimer = setInterval(() => this.sendFrame(socket, onSent), 1000 / fps);
  }

  clearTimer() {
    if (this.sendTimer) clearInterval(this.sendTimer);
    this.sendTimer = null;
  }

  sendFrame(socket, onSent = null) {
    const ws = socket.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Backpressure: at most one frame in flight. A fixed byte budget breaks at
    // high resolutions, where a single JPEG can exceed it and every frame is
    // dropped forever.
    if (ws.bufferedAmount > Math.max(this.lastBytes, 1_000_000)) return;
    // Encoding a big frame takes tens of ms; without this guard the interval
    // starts a second encode and both get sent.
    if (this.encoding) return;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    if (!vw || !vh) return;
    // width 0 (or a frame already smaller) = send as captured; never upscale.
    const scale = this.width > 0 ? Math.min(1, this.width / vw) : 1;
    this.canvas.width = Math.round(vw * scale);
    this.canvas.height = Math.round(vh * scale);
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    this.encoding = true;
    this.canvas.toBlob(
      (blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) {
          this.encoding = false;
          return;
        }
        this.lastBytes = blob.size;
        blob.arrayBuffer().then((buf) => {
          this.encoding = false;
          ws.send(buf);
          if (onSent) onSent();
        });
      },
      "image/jpeg",
      this.quality
    );
  }
}
