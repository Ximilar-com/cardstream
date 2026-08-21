# cardstream — common tasks. Run from the repo root with the venv in .venv/
# (make venv creates it). Everything here is a thin wrapper over the commands
# in README.md / CLAUDE.md — no build logic lives only in this file.

VENV ?= .venv
PY   := $(VENV)/bin/python
WEB  := $(VENV)/bin/cardstream-web

.DEFAULT_GOAL := help
.PHONY: help venv dev prod check lint format typecheck test test-py test-js clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv:  ## create the venv and install the client + dev extras
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e '.[client,onnx]' --group dev

## Development: the same pipeline `make prod` runs, plus the diagnostics —
## the debug log, the located/paid outlines on the page, and every PAID frame
## archived to images/. The tuning flags are all built-in defaults now; they
## stay spelled out because this target doubles as the reference for what a
## full invocation looks like.
dev:  ## run the browser UI with the full development config
	XIMILAR_API_KEY=$${XIMILAR_API_KEY:?set XIMILAR_API_KEY first} $(WEB) \
	    --gate embedding --embed-model model/similarity/onnx/model.onnx \
	    --segmentor rfdetr --segmentor-model model/segmentation/onnx/model.onnx \
	    --detector-conf 0.35 \
	    --similarity-threshold 0.85 \
	    --result-threshold 0.9 \
	    --motion-threshold 8 --still-frames 2 \
	    --detect-interval 0.1 --idle-detect-interval 0.2 \
	    --empty-detect-interval 0.2 \
	    --cooldown 2 \
	    --forget-after 2 \
	    --min-card-size 0.1 \
	    --debug \
	    --store-images images \
	    --store-images-type frame \
	    --show-detection

## Production: no flags at all. Every value `make dev` spells out is the
## built-in default, so this runs the identical pipeline — minus the debug log,
## the on-page outlines and the image archive.
prod:  ## run the browser UI on the defaults alone (no diagnostics)
	XIMILAR_API_KEY=$${XIMILAR_API_KEY:?set XIMILAR_API_KEY first} $(WEB)

check: lint typecheck test  ## everything CI runs, in the order it fails fastest

lint:  ## ruff format --check + ruff check + the generated docs are current
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/ruff check src tests
	$(PY) scripts/gen-cli-docs.py --check

format:  ## rewrite src + tests to the formatted form, regenerate docs
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests
	$(PY) scripts/gen-cli-docs.py

typecheck:  ## mypy (scoped to core/ — see pyproject)
	$(VENV)/bin/mypy

test: test-py test-js  ## the whole offline suite

test-py:  ## pytest (tests/core + tests/client)
	$(PY) -m pytest -q

test-js:  ## the page's pure logic (node stdlib runner, no deps)
	# Quoted: node does the globbing itself, so this neither depends on the
	# shell nor fails when the pattern matches nothing.
	node --test "tests/webui/*.test.js"

clean:  ## drop caches and build artifacts
	# `find`, not `**/`: make runs recipes under /bin/sh, where `**/` is a
	# plain single-level glob and every nested cache survives.
	rm -rf dist build .pytest_cache *.egg-info
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
