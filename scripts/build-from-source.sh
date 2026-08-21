#!/usr/bin/env bash
# Build cardstream from a source checkout: venv + editable install + smoke check.
#
#   ./scripts/build-from-source.sh [--models]
#
# --models (or a set CARDSTREAM_MODELS_URL) also fetches the model weights into
# model/ (detection/, similarity/, tracking/ — the gitignored layout a checkout
# uses; see model/README.md).
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS_URL_DEFAULT="https://cardstream.ai/models/cardstream-models-v1.tar.gz"
FETCH_MODELS=""
for arg in "$@"; do
  case "$arg" in
    --models) FETCH_MODELS=1 ;;
    *) echo "usage: scripts/build-from-source.sh [--models]" >&2; exit 2 ;;
  esac
done
[ -n "${CARDSTREAM_MODELS_URL:-}" ] && FETCH_MODELS=1

# python >= 3.11 (uv fast path if present)
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 \
     && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done

if command -v uv >/dev/null 2>&1; then
  uv venv --python '>=3.11' .venv
  uv pip install --python .venv/bin/python -e '.[client,onnx]' --group dev
elif [ -n "$PY" ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e '.[client,onnx]' --group dev
else
  echo "error: python 3.11+ not found (or install uv: https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# The DEFAULT locator is the segmentor, so that — not the box detector — is
# what "models are already here" means.
if [ -n "$FETCH_MODELS" ] && [ ! -f model/segmentation/onnx/model.onnx ]; then
  echo "fetching model weights into model/..."
  # The published tarball is versioned independently of this checkout and still
  # uses flat detection_model/ / similarity_model/ / models/ names inside, so
  # unpack aside and reshape into the model/ root. Drop the reshaping once a
  # tarball ships the model/ layout itself.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  url="${CARDSTREAM_MODELS_URL:-$MODELS_URL_DEFAULT}"
  # To a file, then verify, then unpack -- a pipe into tar cannot be checked.
  curl -fsSL -o "$tmp/models.tar.gz" "$url"
  if curl -fsSL -o "$tmp/models.sha256" "$url.sha256" 2>/dev/null; then
    want="$(awk '{print $1}' "$tmp/models.sha256")"
    got="$(shasum -a 256 "$tmp/models.tar.gz" 2>/dev/null | awk '{print $1}')"
    [ -n "$got" ] || got="$(sha256sum "$tmp/models.tar.gz" | awk '{print $1}')"
    if [ "$want" != "$got" ]; then
      echo "error: models checksum mismatch — expected $want, got $got" >&2
      exit 1
    fi
    echo "model weights verified"
  else
    echo "warning: no checksum published for $url — cannot verify the weights" >&2
  fi
  tar -xzf "$tmp/models.tar.gz" -C "$tmp"
  rm -f "$tmp/models.tar.gz" "$tmp/models.sha256"
  mkdir -p model
  if [ -d "$tmp/detection_model" ];  then mv "$tmp/detection_model"  model/detection;  fi
  if [ -d "$tmp/segmentation_model" ]; then mv "$tmp/segmentation_model" model/segmentation; fi
  if [ -d "$tmp/similarity_model" ]; then mv "$tmp/similarity_model" model/similarity; fi
  if [ -d "$tmp/models" ];           then mv "$tmp/models"           model/tracking;   fi
fi

# Smoke: the offline engine suite + the CLI itself.
.venv/bin/pytest -q tests/core/test_engine.py
.venv/bin/cardstream-web --version

echo
echo "Built. Next steps:"
echo "  source .venv/bin/activate"
echo "  export XIMILAR_API_KEY=your-key"
echo "  cardstream-web                                               # :8001"
if [ ! -f model/detection/model.onnx ] && [ ! -f model/segmentation/onnx/model.onnx ]; then
  echo
  echo "NOTE: no card locator present. A bare cardstream-web runs the shipped"
  echo "segmentor, so re-run with --models or drop weights into model/segmentation/."
  echo "A box detector in model/detection/ works too, via --detector-model."
  echo "See model/README.md."
fi
