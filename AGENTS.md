# AGENTS.md

Guidance for AI coding agents working in this repository.

The full map — architecture, the `DecisionCore` state machine, the core/client
split, every convention and the known limitations — lives in
**[`CLAUDE.md`](CLAUDE.md)**. Read that first; it is the authoritative
document and this file only exists so that tools which look for `AGENTS.md`
find their way there.

Human contributors should start with
**[`CONTRIBUTING.md`](CONTRIBUTING.md)**.

Quick facts:

- Set up and verify with `make venv` then `make check` (lint + typecheck +
  both test suites). No API key, camera or model weights are needed — the
  suite is entirely offline.
- Two test runners: `pytest` for Python, `node --test` for the browser page's
  logic. Both must pass.
- Documentation is contract-tested. `tests/test_docs_contract.py` checks every
  documented `(default X)` against the real argparse parser, so changing a
  default without updating the docs fails CI.
- `src/cardstream/core/` must not import from `client/`, and heavy ML imports
  stay lazy inside backend classes.
