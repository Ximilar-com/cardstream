#!/usr/bin/env bash
# Build the release artifacts: wheel + sdist + SHA256SUMS.
#
#   scripts/build-release.sh [vX.Y.Z] [--publish]
#
# The version is authored once, in pyproject.toml; a tag argument is only a
# consistency check against it. One pure-Python wheel covers darwin-arm64,
# linux-amd64 and windows-amd64 — platform detection lives in install.sh.
# --publish creates the GitHub release (needs `gh` authed against $REPO).
set -euo pipefail

REPO="Ximilar-com/cardstream"
cd "$(dirname "$0")/.."

TAG=""
PUBLISH=""
for arg in "$@"; do
  case "$arg" in
    --publish) PUBLISH=1 ;;
    v*) TAG="$arg" ;;
    *) echo "usage: scripts/build-release.sh [vX.Y.Z] [--publish]" >&2; exit 2 ;;
  esac
done

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

VERSION="$("$PY" -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
if [ -n "$TAG" ] && [ "$TAG" != "v$VERSION" ]; then
  echo "error: tag $TAG does not match pyproject version v$VERSION" >&2
  exit 1
fi
TAG="v$VERSION"

if [ -n "$PUBLISH" ] && [ -n "$(git status --porcelain)" ]; then
  echo "error: refusing to --publish from a dirty git tree" >&2
  exit 1
fi

"$PY" -c 'import build' 2>/dev/null || {
  echo "error: the 'build' package is missing — pip install --group dev (or pip install build)" >&2
  exit 1
}

echo "building cardstream $VERSION..."
rm -rf dist
"$PY" -m build --quiet

# Wheel metadata must agree with pyproject — catches a stale build backend.
BUILT="$(basename dist/cardstream-*.whl)"
[ "$BUILT" = "cardstream-$VERSION-py3-none-any.whl" ] || {
  echo "error: built wheel $BUILT does not match version $VERSION" >&2
  exit 1
}

# install.sh is deliberately NOT copied anywhere: the one copy lives at
# scripts/install.sh and users curl it straight from the repo (raw main),
# so a fix lands with a push instead of a re-release and a site deploy.

(
  cd dist
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- * > SHA256SUMS.tmp
  else
    shasum -a 256 -- * > SHA256SUMS.tmp
  fi
  mv SHA256SUMS.tmp SHA256SUMS
)

echo
echo "dist/:"
ls -l dist
echo
if [ -n "$PUBLISH" ]; then
  gh release create "$TAG" dist/* --repo "$REPO" \
    --title "cardstream $TAG" --generate-notes
else
  echo "Not publishing (no --publish). To release:"
  echo "  gh release create $TAG dist/* --repo $REPO --title 'cardstream $TAG' --generate-notes"
fi
