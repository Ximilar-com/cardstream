// Reconnecting WebSocket with capped exponential backoff + the connection
// badge. Binary messages go to onBinary (relayed frames), JSON to onJson.

export class ReconnectingSocket {
  constructor({ urlFn, connEl, isActive, onJson, onBinary = null, onOpen = null, onClose = null }) {
    this.urlFn = urlFn;
    this.connEl = connEl;
    this.isActive = isActive;   // () => bool: reconnect only while the page streams
    this.onJson = onJson;
    this.onBinary = onBinary;
    this.onOpen = onOpen;
    this.onClose = onClose;
    this.ws = null;
    this.reconnectTimer = null;
    this.reconnectDelay = 500;  // ms; backs off to RECONNECT_MAX while down
    this.RECONNECT_MAX = 5000;
  }

  connect() {
    const ws = new WebSocket(this.urlFn());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.reconnectDelay = 500; // reset backoff on a successful connection
      this.connEl.textContent = "connected";
      this.connEl.className = "badge badge-on";
      if (this.onOpen) this.onOpen();
    };
    ws.onclose = () => {
      if (this.onClose) this.onClose();
      // Auto-reconnect while the user is still streaming — covers a server
      // restart or a flaky network without a reload.
      if (this.isActive()) {
        this.connEl.textContent = "reconnecting…";
        this.connEl.className = "badge badge-off";
        this.reconnectTimer = setTimeout(() => this.connect(), this.reconnectDelay);
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.RECONNECT_MAX);
      } else {
        this.connEl.textContent = "disconnected";
        this.connEl.className = "badge badge-off";
      }
    };
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        if (this.onBinary) this.onBinary(ev.data);
        return;
      }
      let res;
      try {
        res = JSON.parse(ev.data);
      } catch (err) {
        return; // ignore a malformed frame rather than killing the handler
      }
      this.onJson(res);
    };
    this.ws = ws;
  }

  get open() {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  close() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) this.ws.close();
  }
}
