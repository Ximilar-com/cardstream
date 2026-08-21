# Contributing

Thanks for looking. This is a small, opinionated codebase — the notes below are
the parts that are not obvious from reading it.

## Getting set up

```bash
make venv                 # .venv + the client, onnx and dev dependencies
source .venv/bin/activate
make check                # lint + typecheck + the whole suite — what CI runs
```

`make help` lists every target. You do **not** need a Ximilar API key, a
camera, or the model weights to develop or to run the tests: the suite is
entirely offline and uses fakes.

## Two test runners, both required

```bash
make test-py    # pytest — tests/core and tests/client
make test-js    # node --test — the browser page's pure logic
```

The page's logic is tested with the **Node standard-library runner**: no
bundler, no dependencies, no `package.json`. Keep it that way — the browser
stack here is deliberately vanilla ES modules.

## The rules that matter

**One copy in core.** Decision-making and anything protocol-shaped — the state
machine, motion and pHash, Ximilar parsing, JPEG helpers, the models — lives in
`src/cardstream/core/`. Never re-implement it in a driver. `core` imports only
numpy, OpenCV and requests; heavy ML imports stay lazy *inside* backend classes
so that any subset of extras installs. `core` must not import from `client`.

**The driver stays thin.** `SmartAnalyzer` owns scheduling, logging and I/O.
Behaviour decisions belong in `DecisionCore`.

**Tests are offline.** No network, no API key, no model files. Use the shared
fakes in `tests/_helpers.py`, and `wait_until(...)` rather than a fixed sleep
when waiting on the background identify thread.

**Docs are contract-tested.** `tests/test_docs_contract.py` reads `README.md`,
`CLAUDE.md`, `docs/` and `model/README.md` and checks every documented
`(default X)` against the real argparse parser, and that documented paths
exist. If you change a default, the docs fail in CI until they agree — that is
the point. Keep writing defaults in the `(default X)` form so they stay
checkable.

**A locator is a locator.** A segmentor *is* a `CardDetector` returning the
same `DetectionResult` with `quad` filled in. Nothing downstream should branch
on which kind is running.

**The version is authored once**, in `pyproject.toml`. Everything else reads
`cardstream.__version__`.

## Adding things

| You want to add | Where it goes |
| --- | --- |
| a card category | one `IdType` entry in `core/id_types.py` |
| an identify prefill | one field on `IdentifyOptions` |
| a settings knob | one descriptor in `webui/smart/settings-fields.js` + one field on `SettingsPatch` |
| a frame source | a `FrameSource` subclass in `client/sources.py` + a branch in `make_source` |
| a detector backend | a `CardDetector` subclass in `core/detectors.py` + routing in `make_detector` / `make_segmentor`, tested with a fake ONNX session |

If a change needs more places than that, the duplication has come back — say
so in the pull request rather than working around it.

## Style

`ruff format` and `ruff check` are the whole style guide; `make format` applies
both. `mypy` runs over `core/` and must stay clean. Optionally:

```bash
pre-commit install    # or: prek install
```

Prose in comments and docstrings is part of this codebase's character —
explain *why*, not *what*. Match the density of the code around you.

## Pull requests

Small and focused. Say what changed and why; if it changes behaviour, say what
a user would notice. New behaviour needs a test — the suite is the reason it is
safe to refactor this aggressively.

By contributing you agree your contributions are licensed under the
[Apache License 2.0](LICENSE).
