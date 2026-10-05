#!/bin/bash
# Compare Warp vs Spotlight on nfcorpus dataset

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WARP_CLI="./target/release/warp-cli"
BEIR_DIR="$HOME/src/xtr-warp/beir/nfcorpus"

echo "============================================"
echo "Warp vs Spotlight Benchmark (nfcorpus)"
echo "============================================"
echo

# Check prerequisites
if [ ! -f "$WARP_CLI" ]; then
    echo "Error: warp-cli not found at $WARP_CLI"
    echo "Build with: make warp-cli"
    exit 1
fi

if [ ! -d "$BEIR_DIR" ]; then
    echo "Error: BEIR nfcorpus dataset not found at $BEIR_DIR"
    echo "Please download BEIR dataset"
    exit 1
fi

if ! command -v mdfind &> /dev/null; then
    echo "Error: mdfind (Spotlight) not found - this requires macOS"
    exit 1
fi

# Decompress dataset if needed
[ -f "datasets/nfcorpus.tsv" ] || zstd -dk "datasets/nfcorpus.tsv.zst"

# Check if Warp DB is initialized
if [ ! -f "mydb.sqlite" ]; then
    echo "Initializing Warp database..."
    $WARP_CLI readcsv datasets/nfcorpus.tsv
    $WARP_CLI embed
    $WARP_CLI index
    echo
fi

# Check if Spotlight corpus is prepared
SPOTLIGHT_DIR="$HOME/Documents/nfcorpus-spotlight"
DOC_COUNT=$(find "$SPOTLIGHT_DIR" -name "doc_*.txt" 2>/dev/null | wc -l | tr -d ' ')

if [ "$DOC_COUNT" -lt 3000 ]; then
    echo "Preparing Spotlight corpus..."
    python3 tools/spotlight-nfcorpus.py prepare
    echo
    echo "Waiting 30 seconds for Spotlight to start indexing..."
    sleep 30
    echo
fi

# Check indexing status
INDEXED=$(mdfind -onlyin "$SPOTLIGHT_DIR" -count DOCID 2>/dev/null || echo "0")
echo "Spotlight indexing status: $INDEXED / ~3633 documents"
if [ "$INDEXED" -lt 3000 ]; then
    echo "WARNING: Spotlight may still be indexing. Consider waiting longer."
    echo "Press Enter to continue anyway, or Ctrl-C to abort..."
    read
fi
echo

# Run Warp benchmark
echo "=== Running Warp Benchmark (semantic search) ==="
rm -f warp-results.txt
$WARP_CLI querycsv "$BEIR_DIR/questions.test.tsv" warp-results.txt

echo
echo "=== Running Spotlight Benchmark (lexical search) ==="
rm -f spotlight-results.txt
python3 tools/spotlight-nfcorpus.py query spotlight-results.txt

echo
echo "============================================"
echo "Results"
echo "============================================"
echo

echo "=== Warp NDCG@10 ==="
cargo run --quiet --release -p trec-score -- warp-results.txt "$BEIR_DIR/collection_map.json" "$BEIR_DIR/qrels.test.json"

echo
echo "=== Spotlight NDCG@10 ==="
cargo run --quiet --release -p trec-score -- spotlight-results.txt "$BEIR_DIR/collection_map.json" "$BEIR_DIR/qrels.test.json"

echo
echo "============================================"
echo "Benchmark complete!"
echo "Results saved to:"
echo "  warp-results.txt (semantic search)"
echo "  spotlight-results.txt (lexical search)"
echo "============================================"
