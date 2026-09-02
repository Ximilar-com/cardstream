#!/bin/sh
# cardstream installer — venv install from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/Ximilar-com/cardstream/main/scripts/install.sh | sh
#
# This is the ONLY copy of the installer — the site and the docs point at the
# raw-main URL above, and it is deliberately not a release asset, so a fix
# lands with a push instead of a re-release.
#
# Knobs (env vars):
#   CARDSTREAM_HOME        install root, holds venv/ + models/   (default ~/.cardstream)
#   INSTALL_DIR            where the cardstream-* shims go       (default ~/.local/bin)
#   CARDSTREAM_VERSION     release tag to install, e.g. v0.2.0   (default: latest)
#   CARDSTREAM_EXTRAS      pip extras                            (default client,onnx)
#   CARDSTREAM_MODELS_BASE_URL  where the model tarballs live
#   CARDSTREAM_MODEL_ARCHIVES   space-separated tarball names to fetch from there
#   CARDSTREAM_TRACKER_URL      the opt-in vitTracker .onnx (default: OpenCV zoo)
#   CARDSTREAM_WHEEL       path to a local wheel — skips the GitHub download (testing)
set -eu

REPO="Ximilar-com/cardstream"
CARDSTREAM_HOME="${CARDSTREAM_HOME:-$HOME/.cardstream}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
EXTRAS="${CARDSTREAM_EXTRAS:-client,onnx}"
# Each model ships as its OWN tarball, versioned independently, so a retrain
# republishes one archive without touching the others. Bump the one that
# changed here.
MODELS_BASE_URL="${CARDSTREAM_MODELS_BASE_URL:-https://github.com/Ximilar-com/cardstream/releases/download/v1.0.0}"
MODEL_ARCHIVES="${CARDSTREAM_MODEL_ARCHIVES:-cardstream-segmentation-v1.tar.gz cardstream-similarity-v1.tar.gz}"
# The opt-in visual tracker is OpenCV's own published vitTracker, fetched from
# the OpenCV zoo rather than republished. The URL floats on their main branch,
# so the pinned checksum below is what holds the version still.
TRACKER_URL="${CARDSTREAM_TRACKER_URL:-https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx}"
TRACKER_SHA256="2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d"
VERSION="${CARDSTREAM_VERSION:-}"

say()  { printf '\033[1m[cardstream]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[cardstream] error:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33m[cardstream] warning:\033[0m %s\n' "$*" >&2; }

# --- platform + required tools ----------------------------------------------
OS="$(uname -s)"
case "$OS" in
  Linux|Darwin) ;;
  *) fail "unsupported OS: $OS — on Windows, follow the pip instructions at https://cardstream.ai/download/" ;;
esac
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar  >/dev/null 2>&1 || fail "tar is required"

# --- python >= 3.11 (uv can provision one itself) ---------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 \
     && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done
HAVE_UV=""
command -v uv >/dev/null 2>&1 && HAVE_UV=1
[ -n "$PY" ] || [ -n "$HAVE_UV" ] || \
  fail "python 3.11+ not found — install it, or install uv (https://docs.astral.sh/uv/) and re-run"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1;    then shasum -a 256 "$1" | awk '{print $1}'
  elif command -v openssl >/dev/null 2>&1;   then openssl dgst -sha256 -r "$1" | awk '{print $1}'
  else fail "sha256sum, shasum, or openssl is required to verify downloads"
  fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- fetch + verify the wheel (or take a local one) -------------------------
if [ -n "${CARDSTREAM_WHEEL:-}" ]; then
  [ -f "$CARDSTREAM_WHEEL" ] || fail "CARDSTREAM_WHEEL not found: $CARDSTREAM_WHEEL"
  WHEEL_PATH="$CARDSTREAM_WHEEL"
  say "using local wheel $WHEEL_PATH (skipping GitHub download)"
else
  if [ -z "$VERSION" ]; then
    VERSION="$(curl -fsSI "https://github.com/$REPO/releases/latest" \
      | tr -d '\r' | awk -F/ 'tolower($0) ~ /^location:/ { print $NF }')"
    [ -n "$VERSION" ] || fail "could not resolve the latest release — see https://github.com/$REPO/releases or https://cardstream.ai/download/"
  fi
  BARE="${VERSION#v}"
  WHEEL="cardstream-${BARE}-py3-none-any.whl"
  BASE="https://github.com/$REPO/releases/download/$VERSION"

  say "downloading cardstream $VERSION..."
  curl -fsSL -o "$TMP/$WHEEL" "$BASE/$WHEEL" \
    || fail "download failed: $BASE/$WHEEL — see https://cardstream.ai/download/ for manual install"
  curl -fsSL -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
    || fail "download failed: $BASE/SHA256SUMS"

  # The name field may carry a "./" prefix (sha256sum ./* wrote the v0.2.0
  # sums that way) — strip it before comparing.
  WANT="$(awk -v f="$WHEEL" '{ n = $NF; sub(/^\.\//, "", n) } n == f { print $1; exit }' "$TMP/SHA256SUMS")"
  [ -n "$WANT" ] || fail "$WHEEL missing from SHA256SUMS"
  GOT="$(sha256_file "$TMP/$WHEEL")"
  [ "$WANT" = "$GOT" ] || fail "checksum mismatch for $WHEEL
expected: $WANT
actual:   $GOT"
  say "checksum OK: $WHEEL"
  WHEEL_PATH="$TMP/$WHEEL"
fi

# --- venv + install ---------------------------------------------------------
VENV="$CARDSTREAM_HOME/venv"
mkdir -p "$CARDSTREAM_HOME"
say "installing into $VENV (extras: $EXTRAS)..."
if [ -n "$HAVE_UV" ]; then
  uv venv --quiet --python '>=3.11' "$VENV"
  uv pip install --quiet --python "$VENV/bin/python" "$WHEEL_PATH[$EXTRAS]"
else
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip   # extras-on-a-wheel-path needs a recent pip
  "$VENV/bin/pip" install --quiet "$WHEEL_PATH[$EXTRAS]"
fi

# --- model weights ----------------------------------------------------------
# Layout contract: each per-model tarball unpacks a top-level directory —
# segmentation_model/, similarity_model/, tracking_model/ (detection_model/
# from an older publication) — side by side into $MODELS. Deliberately NOT
# the model/{detection,segmentation,similarity,tracking} layout a checkout
# uses: this unpacks into its own directory and the tarballs are versioned
# separately, so the two only have to agree with themselves. The shims below
# carry the paths, so nobody types either layout.
MODELS="$CARDSTREAM_HOME/models"
SEG="$MODELS/segmentation_model/onnx/model.onnx"
BOX="$MODELS/detection_model/model.onnx"
if [ -f "$SEG" ] || [ -f "$BOX" ]; then
  say "model weights already present in $MODELS"
else
  say "fetching model weights..."
  mkdir -p "$MODELS"
  for archive in $MODEL_ARCHIVES; do
    url="$MODELS_BASE_URL/$archive"
    # To a FILE first, not straight into tar: a pipe cannot be verified, and
    # this is code-adjacent data fetched over the network. The wheel a few
    # lines up is checksummed; so is this.
    curl -fsSL -o "$TMP/$archive" "$url" \
      || fail "could not fetch $url — set CARDSTREAM_MODELS_BASE_URL or see https://cardstream.ai/download/"

    # The sums file sits beside each tarball. Absent (an older publication, or
    # a base URL of your own) we warn rather than refuse -- but a sums file
    # that DISAGREES is always fatal.
    if curl -fsSL -o "$TMP/$archive.sha256" "$url.sha256" 2>/dev/null; then
      want="$(awk '{print $1}' "$TMP/$archive.sha256")"
      got="$(sha256_file "$TMP/$archive")"
      [ "$want" = "$got" ] \
        || fail "$archive checksum mismatch — expected $want, got $got. Refusing to unpack."
      say "$archive verified"
    else
      warn "no checksum published for $url — cannot verify the weights"
    fi

    tar -xzf "$TMP/$archive" -C "$MODELS" \
      || fail "could not unpack $archive"
    rm -f "$TMP/$archive" "$TMP/$archive.sha256"
  done
  [ -f "$SEG" ] || [ -f "$BOX" ] \
    || fail "model archives contained no card locator (segmentation_model/onnx/model.onnx or detection_model/model.onnx)"
fi

# The opt-in tracker, straight from the OpenCV zoo. Outside the block above so
# a re-run adds it to an install made before it was fetched. It is opt-in at
# runtime (nothing passes --tracker-model by default), so a failed download is
# a warning, not a failed install — but a download that DISAGREES with the
# pinned checksum is always fatal.
TRACKER="$MODELS/tracking_model/object_tracking_vittrack_2023sep.onnx"
if [ ! -f "$TRACKER" ]; then
  if curl -fsSL -o "$TMP/vittrack.onnx" "$TRACKER_URL" 2>/dev/null; then
    got="$(sha256_file "$TMP/vittrack.onnx")"
    [ "$got" = "$TRACKER_SHA256" ] \
      || fail "tracker checksum mismatch — expected $TRACKER_SHA256, got $got. Refusing to install it."
    mkdir -p "$MODELS/tracking_model"
    mv "$TMP/vittrack.onnx" "$TRACKER"
    say "tracker fetched (OpenCV zoo vitTracker)"
  else
    warn "could not fetch the optional tracker from $TRACKER_URL — pass --tracker-model with your own copy if you want tracking"
  fi
fi

# Which locator the shims will pass. The segmentor is what a source checkout
# runs by default — deskewed, tight crops — so prefer it, and fall back to the
# box detector for archives published before it shipped. Resolved HERE rather
# than hardcoded, because the archives are versioned independently of this script.
if [ -f "$SEG" ]; then
  LOCATOR_FLAG="--segmentor-model"
  LOCATOR_MODEL="$SEG"
else
  LOCATOR_FLAG="--detector-model"
  LOCATOR_MODEL="$BOX"
fi
say "card locator: $LOCATOR_FLAG"

# --- shims on PATH ----------------------------------------------------------
mkdir -p "$INSTALL_DIR" 2>/dev/null || true
[ -d "$INSTALL_DIR" ] && [ -w "$INSTALL_DIR" ] \
  || fail "$INSTALL_DIR is not writable — re-run with INSTALL_DIR=<dir> (or sudo for a system dir)"
for cmd in cardstream-web cardstream-client; do
  cat > "$INSTALL_DIR/$cmd" <<EOF
#!/bin/sh
# cardstream shim — points the model defaults at $MODELS; flags you pass win.
#
# The locator is applied only when you named none yourself: the two model flags
# are mutually exclusive, so appending ours unconditionally turned
# \`cardstream-web --detector-model mine.onnx\` into a "pick one" error.
locator="$LOCATOR_FLAG \"$LOCATOR_MODEL\""
for arg in "\$@"; do
  case "\$arg" in
    --detector-model|--detector-model=*|--segmentor-model|--segmentor-model=*)
      locator=""; break ;;
  esac
done
eval exec "\"$VENV/bin/$cmd\"" \$locator \
  --embed-model "\"$MODELS/similarity_model/onnx/model.onnx\"" \
  '"\$@"'
EOF
  chmod +x "$INSTALL_DIR/$cmd"
done

# --- self-check + next steps ------------------------------------------------
"$INSTALL_DIR/cardstream-web" --version >/dev/null || fail "self-check failed"
INSTALLED="$("$INSTALL_DIR/cardstream-web" --version)"

say "$INSTALLED installed."
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) say "NOTE: $INSTALL_DIR is not on your PATH — add it to your shell profile" ;;
esac
say ""
say "Next steps:"
say "  export XIMILAR_API_KEY=your-key   # https://app.ximilar.com"
say "  cardstream-web                    # opens http://127.0.0.1:8001"
say ""
say "(--listen / --ffmpeg sources need the system ffmpeg binary.)"
