#!/bin/bash
set -euo pipefail

# Read current version from Cargo.toml
CURRENT=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
IFS='.' read -r MAJOR MINOR PATCH <<< "${CURRENT}"

# Bump patch
NEW="${MAJOR}.${MINOR}.$((PATCH + 1))"
TAG="v${NEW}"

echo "${CURRENT} -> ${NEW}"

# Update Cargo.toml
sed -i '' "0,/^version = \".*\"/s//version = \"${NEW}\"/" Cargo.toml

# Commit, tag, push
git add Cargo.toml
git commit -m "release ${TAG}"
git tag "${TAG}"
git push public master "${TAG}"

echo "Pushed ${TAG} — CI will build and create the release."
