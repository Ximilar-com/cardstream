#!/bin/sh
# cardstream container entrypoint: fetch model weights on first run, then
# start the web client. Arguments are appended after the defaults, so any flag
# you pass to `docker run ... cardstream <flags>` wins (argparse last-one-wins)
# — except the card locator, which is mutually exclusive rather than
# overridable, so we only supply one when you named none.
set -eu

MODELS="/models"
SEG="$MODELS/segmentation_model/onnx/model.onnx"
BOX="$MODELS/detection_model/model.onnx"

if [ ! -f "$SEG" ] && [ ! -f "$BOX" ]; then
  echo "[cardstream] fetching model weights into $MODELS..."
  TARBALL="$(mktemp)"
  trap 'rm -f "$TARBALL" "$TARBALL.sha256"' EXIT
  # Downloaded to a file, not piped into tar, so it can be verified first.
  curl -fsSL -o "$TARBALL" "$CARDSTREAM_MODELS_URL" || {
    echo "[cardstream] error: could not fetch models from $CARDSTREAM_MODELS_URL" >&2
    echo "[cardstream] set CARDSTREAM_MODELS_URL or mount pre-fetched weights: -v ./models:/models" >&2
    exit 1
  }
  # A published checksum that disagrees is fatal; one that is missing is a
  # warning, so a self-hosted CARDSTREAM_MODELS_URL keeps working.
  if curl -fsSL -o "$TARBALL.sha256" "$CARDSTREAM_MODELS_URL.sha256" 2>/dev/null; then
    want="$(awk '{print $1}' "$TARBALL.sha256")"
    got="$(sha256sum "$TARBALL" | awk '{print $1}')"
    if [ "$want" != "$got" ]; then
      echo "[cardstream] error: models checksum mismatch — expected $want, got $got" >&2
      exit 1
    fi
    echo "[cardstream] model weights verified"
  else
    echo "[cardstream] warning: no checksum published — cannot verify the weights" >&2
  fi
  tar -xzf "$TARBALL" -C "$MODELS" || {
    echo "[cardstream] error: could not unpack the models tarball" >&2
    exit 1
  }
  rm -f "$TARBALL" "$TARBALL.sha256"
  trap - EXIT
fi

# The locator we supply, unless the caller named one of their own: the two
# model flags are mutually exclusive, so adding ours unconditionally turned
# `docker run ... cardstream --detector-model mine.onnx` into a "pick one"
# error. Prefer the segmentor — deskewed, tight crops, and what a source
# checkout runs by default — falling back to the box detector for a tarball
# published before it shipped. Resolved from what actually unpacked: the
# tarball is versioned independently of this image.
for arg in "$@"; do
  case "$arg" in
    --detector-model|--detector-model=*|--segmentor-model|--segmentor-model=*)
      SEG=""; BOX=""; break ;;
  esac
done
if [ -n "$SEG" ] && [ -f "$SEG" ]; then
  set -- --segmentor-model "$SEG" "$@"
elif [ -n "$BOX" ] && [ -f "$BOX" ]; then
  set -- --detector-model "$BOX" "$@"
fi

exec cardstream-web --host 0.0.0.0 --no-browser \
  --embed-model "$MODELS/similarity_model/onnx/model.onnx" \
  "$@"
