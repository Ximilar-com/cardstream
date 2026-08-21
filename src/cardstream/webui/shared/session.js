// The start/stop loop the page runs.
//
// Camera mode and viewer mode differ in where frames come from and what they
// render, but the session around that is the same either way: a `running`
// flag, the rate slider, the Start/Stop button, and a stop() that tears down
// capture, socket and overlay together.
//
// The socket needs `isActive: () => session.running` and the session needs the
// socket, so build the session first and hand it the socket with attach().

export class Session {
  /**
   * @param els          the shared panel elements (needs .rate, .rateVal, .toggle)
   * @param capture      CameraCapture
   * @param overlay      Overlay
   * @param onSend       called per sent frame (fps badge, resolution read-outs)
   * @param onStart      called once the camera is live, before connecting
   * @param needsCamera  false in viewer modes, where frames arrive from the process
   */
  constructor({ els, capture, overlay, onSend = null, onStart = null, needsCamera = null }) {
    this.running = false;
    this._els = els;
    this._capture = capture;
    this._overlay = overlay;
    this._onSend = onSend;
    this._onStart = onStart;
    this._needsCamera = needsCamera || (() => true);
    this._socket = null;
  }

  /** Wire the rate slider and the Start/Stop button to this session. */
  attach(socket) {
    this._socket = socket;
    this._els.rate.addEventListener("input", () => {
      this._els.rateVal.textContent = this._els.rate.value;
      if (this.running) this.scheduleSends();
    });
    this._els.toggle.addEventListener("click", () =>
      this.running ? this.stop() : this.start()
    );
    return this;
  }

  scheduleSends() {
    const fps = parseInt(this._els.rate.value, 10);
    this._capture.scheduleSends(this._socket, fps, () => this._onSend?.());
  }

  async start() {
    if (this._needsCamera()) {
      try {
        await this._capture.start();
      } catch (err) {
        alert("Could not access camera: " + err.message);
        return;
      }
    }
    this._onStart?.();
    this._socket.connect();
    this.running = true;
    this._els.toggle.textContent = "Stop";
    this._els.toggle.classList.add("stop");
    this._overlay.start();
  }

  stop() {
    this.running = false;
    this._els.toggle.textContent = "Start";
    this._els.toggle.classList.remove("stop");
    this._capture.stop();
    this._socket.close();
    this._overlay.stop();
  }
}
