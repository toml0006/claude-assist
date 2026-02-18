#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <semver>"
  echo "Example: $0 1.3.2"
  exit 1
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid version '$VERSION'. Expected semantic version format: X.Y.Z"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="$ROOT_DIR/VERSION"
MANIFEST_FILE="$ROOT_DIR/custom_components/claude_assist/manifest.json"

printf "%s\n" "$VERSION" >"$VERSION_FILE"

if command -v perl >/dev/null 2>&1; then
  perl -0pi -e 's/"version"\s*:\s*"[0-9]+\.[0-9]+\.[0-9]+"/"version": "'"$VERSION"'"/' "$MANIFEST_FILE"
else
  # Fallback for shells without perl; assumes one version key in manifest.
  sed -i.bak -E 's/"version"[[:space:]]*:[[:space:]]*"[0-9]+\.[0-9]+\.[0-9]+"/"version": "'"$VERSION"'"/' "$MANIFEST_FILE"
  rm -f "$MANIFEST_FILE.bak"
fi

echo "Updated:"
echo "  - $VERSION_FILE"
echo "  - $MANIFEST_FILE"
echo
echo "Next:"
echo "  1) git add VERSION custom_components/claude_assist/manifest.json"
echo "  2) git commit -m \"Release v$VERSION\""
echo "  3) git tag v$VERSION"
echo "  4) git push origin main --tags"
