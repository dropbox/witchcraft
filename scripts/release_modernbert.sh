#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

repo=dropbox/witchcraft
tag=modernbert-96d-gated-v1
files=(modernbert-config.json modernbert-tokenizer.json modernbert.safetensors)

command -v gh >/dev/null || { echo "Install GitHub CLI (gh) first." >&2; exit 1; }
command -v shasum >/dev/null || { echo "shasum is required to generate SHA-256 checksums." >&2; exit 1; }
for file in "${files[@]}"; do
    test -s "assets/$file" || { echo "Missing or empty assets/$file" >&2; exit 1; }
done
for file in LICENSE NOTICE; do
    test -s "licenses/granite/$file" || { echo "Missing or empty licenses/granite/$file" >&2; exit 1; }
done
test -s LICENSE || { echo "Missing or empty repository LICENSE" >&2; exit 1; }
gh auth status --hostname github.com

release_dir=$(mktemp -d)
trap 'rm -rf "$release_dir"' EXIT

cp LICENSE licenses/granite/NOTICE "$release_dir/"
cp licenses/granite/LICENSE "$release_dir/LICENSE.granite"
(
    cd assets
    shasum -a 256 "${files[@]}"
    cd "$release_dir"
    shasum -a 256 LICENSE LICENSE.granite NOTICE
) > "$release_dir/SHA256SUMS"

tar -czf "$release_dir/modernbert-assets.tar.gz" \
    -C "$PWD/assets" "${files[@]}" -C "$release_dir" SHA256SUMS LICENSE LICENSE.granite NOTICE

cat > "$release_dir/notes.md" <<'EOF'
ModernBERT retrieval weights for Witchcraft: 22 layers, 768 hidden dimensions,
a 96-dimensional token projection, and learned token gating.

`modernbert-assets.tar.gz` contains the F32 safetensors weights, config,
tokenizer, the repository `LICENSE`, the upstream `LICENSE.granite`, an
attribution and modification `NOTICE`, and `SHA256SUMS` for verifying all six files.

The weights are a modified derivative of IBM's
[granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2),
released under the Apache License, Version 2.0. Our modifications add token-level
retrieval fine-tuning, a 96-dimensional projection, and learned token gating.
The fine-tuning and other modifications are Copyright (c) 2026 Dropbox Inc.
and covered by the repository's Apache License, Version 2.0.

Download the model into the repository's `assets/` directory with `curl`.
No GitHub account or GitHub CLI is required:

```bash
mkdir -p assets
curl --fail --location --output assets/modernbert-assets.tar.gz \
  https://github.com/dropbox/witchcraft/releases/download/modernbert-96d-gated-v1/modernbert-assets.tar.gz
tar -xzf assets/modernbert-assets.tar.gz -C assets
(cd assets && shasum -a 256 --check SHA256SUMS)
make warp-cli
```

The default build derives `modernbert.gguf` locally from the safetensors weights.
For the unquantized backend, run `make warp-cli ENCODER=modernbert`.
EOF

if gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
    gh release upload "$tag" "$release_dir/modernbert-assets.tar.gz" --repo "$repo" --clobber
    gh release edit "$tag" --repo "$repo" --notes-file "$release_dir/notes.md" --draft=false --latest=false
    # Remove the old individual assets only after the archive upload succeeds.
    old_assets=$(gh release view "$tag" --repo "$repo" --json assets --jq '.assets[].name')
    while IFS= read -r file; do
        case "$file" in
            modernbert-config.json|modernbert-tokenizer.json|modernbert.safetensors|SHA256SUMS)
                gh release delete-asset "$tag" "$file" --repo "$repo" --yes
                ;;
        esac
    done <<< "$old_assets"
else
    # Keep incomplete uploads private and do not replace the latest software release.
    gh release create "$tag" "$release_dir/modernbert-assets.tar.gz" \
        --repo "$repo" --draft --latest=false \
        --title "ModernBERT 96-dimensional gated retrieval weights v1" \
        --notes-file "$release_dir/notes.md"
    gh release edit "$tag" --repo "$repo" --draft=false --latest=false
fi
gh release view "$tag" --repo "$repo" --json url --jq .url
