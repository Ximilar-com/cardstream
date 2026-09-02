#!/usr/bin/env bash
# Build cardstream from a source checkout: venv + editable install + smoke check.
#
#   ./scripts/build-from-source.sh [--models]
#
# --models (or a set CARDSTREAM_MODELS_BASE_URL) also fetches the model weights
# into model/ (segmentation/, similarity/, tracking/ — the gitignored layout a
# checkout uses; see model/README.md).
set -euo pipefail

cd "$(dirname "$0")/.."

# One tarball per model, versioned independently — a retrain republishes one
# archive; bump the one that changed here. The opt-in tracker comes straight
# from the OpenCV zoo — its URL floats on their main branch, so the pinned
# checksum holds the version still.
MODELS_BASE_URL="${CARDSTREAM_MODELS_BASE_URL:-https://github.com/Ximilar-com/cardstream/releases/download/v1.0.0}"
MODEL_ARCHIVES="${CARDSTREAM_MODEL_ARCHIVES:-cardstream-segmentation-v1.tar.gz cardstream-similarity-v1.tar.gz}"
TRACKER_URL="${CARDSTREAM_TRACKER_URL:-https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx}"
TRACKER_SHA256="2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d"
FETCH_MODELS=""
for arg in "$@"; do
  case "$arg" in
    --models) FETCH_MODELS=1 ;;
    *) echo "usage: scripts/build-from-source.sh [--models]" >&2; exit 2 ;;
  esac
done
[ -n "${CARDSTREAM_MODELS_BASE_URL:-}" ] && FETCH_MODELS=1

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
  # The published tarballs are versioned independently of this checkout and
  # still use flat segmentation_model/ / similarity_model/ / tracking_model/
  # names inside, so unpack aside and reshape into the model/ root. Drop the
  # reshaping once the tarballs ship the model/ layout themselves.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  for archive in $MODEL_ARCHIVES; do
    url="$MODELS_BASE_URL/$archive"
    # To a file, then verify, then unpack -- a pipe into tar cannot be checked.
    curl -fsSL -o "$tmp/$archive" "$url"
    if curl -fsSL -o "$tmp/$archive.sha256" "$url.sha256" 2>/dev/null; then
      want="$(awk '{print $1}' "$tmp/$archive.sha256")"
      got="$(shasum -a 256 "$tmp/$archive" 2>/dev/null | awk '{print $1}')"
      [ -n "$got" ] || got="$(sha256sum "$tmp/$archive" | awk '{print $1}')"
      if [ "$want" != "$got" ]; then
        echo "error: $archive checksum mismatch — expected $want, got $got" >&2
        exit 1
      fi
      echo "$archive verified"
    else
      echo "warning: no checksum published for $url — cannot verify the weights" >&2
    fi
    tar -xzf "$tmp/$archive" -C "$tmp"
    rm -f "$tmp/$archive" "$tmp/$archive.sha256"
  done
  mkdir -p model
  if [ -d "$tmp/detection_model" ];  then mv "$tmp/detection_model"  model/detection;  fi
  if [ -d "$tmp/segmentation_model" ]; then mv "$tmp/segmentation_model" model/segmentation; fi
  if [ -d "$tmp/similarity_model" ]; then mv "$tmp/similarity_model" model/similarity; fi

  # The opt-in tracker, straight from the OpenCV zoo. A failed download is a
  # warning (nothing passes --tracker-model by default); a checksum mismatch
  # is fatal.
  if [ ! -f model/tracking/object_tracking_vittrack_2023sep.onnx ]; then
    if curl -fsSL -o "$tmp/vittrack.onnx" "$TRACKER_URL" 2>/dev/null; then
      got="$(shasum -a 256 "$tmp/vittrack.onnx" 2>/dev/null | awk '{print $1}')"
      [ -n "$got" ] || got="$(sha256sum "$tmp/vittrack.onnx" | awk '{print $1}')"
      if [ "$got" != "$TRACKER_SHA256" ]; then
        echo "error: tracker checksum mismatch — expected $TRACKER_SHA256, got $got" >&2
        exit 1
      fi
      mkdir -p model/tracking
      mv "$tmp/vittrack.onnx" model/tracking/object_tracking_vittrack_2023sep.onnx
      echo "tracker fetched (OpenCV zoo vitTracker)"
    else
      echo "warning: could not fetch the optional tracker from $TRACKER_URL" >&2
    fi
  fi
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
