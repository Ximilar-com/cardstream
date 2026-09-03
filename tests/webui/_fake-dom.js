// The smallest DOM the Overlay's pure logic touches, shared by the history
// tests. The Overlay takes its elements by injection, so a handful of stubs
// stands in for a browser. Not matched by the *.test.js glob, so it never
// runs as a test itself.

export class FakeEl {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.textContent = "";
    this.className = "";
    this.title = "";
    this.hidden = true;
    this.parent = null;
    this.classList = { add() {}, remove() {} };
  }
  append(...kids) {
    for (const k of kids) {
      k.parent = this;
      this.children.push(k);
    }
  }
  prepend(kid) {
    kid.parent = this;
    this.children.unshift(kid);
  }
  replaceChildren() {
    for (const k of this.children) k.parent = null;
    this.children = [];
  }
  remove() {
    if (!this.parent) return;
    this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
  get lastChild() {
    return this.children[this.children.length - 1];
  }
}

// Call BEFORE importing overlay.js: its constructor starts a 500ms duration
// tick, and a live timer would keep the test runner alive. Ticking is driven
// explicitly by the tests instead.
export function installFakeDom() {
  globalThis.document = { createElement: (tag) => new FakeEl(tag) };
  globalThis.setInterval = () => 0;
}
