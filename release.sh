#!/bin/bash
set -euo pipefail

VERSION="${1:?Usage: release.sh <version>}"
REPO="dropbox/witchcraft"
TAG="v${VERSION}"
FORMULA="Formula/pickbrain.rb"

# Build both architectures
echo "==> Building aarch64 (Apple Silicon)..."
cargo build --release --target aarch64-apple-darwin \
  --features t5-quantized,metal,progress,embed-assets \
  --example pickbrain

echo "==> Building x86_64 (Intel)..."
RUSTFLAGS='-C target-feature=+avx2,+fma' cargo build --release --target x86_64-apple-darwin \
  --features t5-quantized,fbgemm,hybrid-dequant,progress,embed-assets \
  --example pickbrain

# Package tarballs
STAGING=$(mktemp -d)
for arch in aarch64 x86_64; do
  echo "==> Packaging ${arch}..."
  DIR="${STAGING}/pickbrain-${VERSION}-${arch}-apple-darwin"
  mkdir -p "${DIR}/skills/pickbrain" "${DIR}/skills/pickbrain-codex"
  cp "target/${arch}-apple-darwin/release/examples/pickbrain" "${DIR}/"
  cp skills/pickbrain/SKILL.md "${DIR}/skills/pickbrain/"
  cp skills/pickbrain-codex/SKILL.md "${DIR}/skills/pickbrain-codex/"
  tar -czf "${STAGING}/pickbrain-${VERSION}-${arch}-apple-darwin.tar.gz" -C "${STAGING}" \
    "pickbrain-${VERSION}-${arch}-apple-darwin"
done

# Update formula with version and SHA256 hashes
echo "==> Updating ${FORMULA}..."
for arch in aarch64 x86_64; do
  TARBALL="${STAGING}/pickbrain-${VERSION}-${arch}-apple-darwin.tar.gz"
  HASH=$(shasum -a 256 "${TARBALL}" | awk '{print $1}')
  echo "  ${arch}: ${HASH}"
  # Replace the sha256 on the line after the url containing this arch
  sed -i '' "/${arch}-apple-darwin/{ n; s/sha256 \".*\"/sha256 \"${HASH}\"/; }" "${FORMULA}"
done
sed -i '' "s/^  version \".*\"/  version \"${VERSION}\"/" "${FORMULA}"

# Create GitHub release and upload
echo ""
echo "==> Creating GitHub release ${TAG}..."
gh release create "${TAG}" \
  --repo "${REPO}" \
  --title "pickbrain ${VERSION}" \
  --notes "Semantic search over Claude Code and Codex conversations.

## Install

\`\`\`
brew tap dropbox/witchcraft https://github.com/dropbox/witchcraft
brew install pickbrain
\`\`\`

Or download the binary for your platform below." \
  "${STAGING}/pickbrain-${VERSION}-aarch64-apple-darwin.tar.gz" \
  "${STAGING}/pickbrain-${VERSION}-x86_64-apple-darwin.tar.gz"

rm -rf "${STAGING}"

echo ""
echo "==> Done! Formula updated. Commit and push to make it live."
