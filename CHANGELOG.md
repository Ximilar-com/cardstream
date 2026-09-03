# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A light/dark theme switch (☀/☾, top right of the web client). Follows the
  system setting until you pick; the pick is remembered by the browser.
- `--price-stats` (off unless passed; also a switch in the ⚙ dialog): every
  identification asks the id endpoint for aggregated USD market prices and
  shows them on the card panel, in each history row and on the terminal.
- Apache-2.0 `LICENSE` and a `NOTICE` attributing the Apache-2.0 upstreams
  (RF-DETR, RT-DETRv2, OpenCV zoo vitTracker, MobileNetV2).
- Full packaging metadata: readme, licence expression, authors, keywords,
  classifiers and project URLs, so the PyPI page renders.
- `py.typed` — the package's type hints are now visible to consumers (PEP 561).
- Continuous integration: lint, typecheck and the full suite on Python
  3.11–3.13 across Linux, macOS and Windows; a release workflow publishing to
  PyPI via Trusted Publishing with build attestations.
- `ruff`, `mypy`, and a `pre-commit` configuration; `make check`, `make lint`,
  `make format`, `make typecheck`.
- `uv.lock`, so CI and contributor environments are reproducible.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog, and issue forms that
  require the startup banner.

### Changed
- `pydantic` is now a declared dependency of the `client` extra. It was always
  imported directly by the settings endpoint but arrived only via FastAPI.
- Development dependencies moved from the `[dev]` extra to a PEP 735
  `[dependency-groups]` group: `pip install -e '.[client,onnx]' --group dev`.
- The test suite now treats warnings as errors, so upstream deprecations
  surface at the commit that introduces them instead of silently accruing.

### Fixed
- `FFmpegSource` leaked the ffmpeg subprocess's stdout and stderr pipes on
  every restart. A reconnecting `--listen` source restarts often, so the
  descriptors accumulated.

## [0.2.0]

The first release covered by this changelog. Earlier history predates the
public repository.
