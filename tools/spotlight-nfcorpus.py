#!/usr/bin/env python3
"""
Benchmark Apple Spotlight on nfcorpus dataset using NDCG@10.

This tool:
1. Extracts nfcorpus documents to ~/Documents/nfcorpus-spotlight/
2. Queries Spotlight using the same test queries as Warp
3. Outputs results in the same format for trec-score evaluation

Usage:
    python spotlight-nfcorpus.py prepare              # Extract docs for Spotlight
    python spotlight-nfcorpus.py query output.txt     # Run benchmark, save to output.txt
    python spotlight-nfcorpus.py score output.txt     # Score results using trec-score

Full workflow:
    python spotlight-nfcorpus.py prepare
    # Wait a few minutes for Spotlight to index
    python spotlight-nfcorpus.py query spotlight-results.txt
    cargo run --quiet --release -p trec-score -- spotlight-results.txt ~/src/xtr-warp/beir/nfcorpus/collection_map.json ~/src/xtr-warp/beir/nfcorpus/qrels.test.json
"""

import subprocess
import json
import sys
import time
from pathlib import Path

# Paths
BEIR_DIR = Path.home() / "src" / "xtr-warp" / "beir" / "nfcorpus"
COLLECTION_FILE = BEIR_DIR / "collection.tsv"
QUESTIONS_FILE = BEIR_DIR / "questions.test.tsv"
COLLECTION_MAP = BEIR_DIR / "collection_map.json"
QRELS_FILE = BEIR_DIR / "qrels.test.json"

OUTPUT_DIR = Path.home() / "Documents" / "nfcorpus-spotlight"
ROOT = Path(__file__).resolve().parent.parent

def prepare_corpus():
    """Extract nfcorpus documents to text files for Spotlight indexing."""
    if not COLLECTION_FILE.exists():
        print(f"Error: {COLLECTION_FILE} not found")
        print("Please ensure BEIR nfcorpus dataset is available at ~/src/xtr-warp/beir/nfcorpus/")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting documents to {OUTPUT_DIR}")
    print("This creates one .txt file per document for Spotlight to index")

    doc_count = 0
    with open(COLLECTION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue

            doc_id, text = parts
            doc_count += 1

            # Create text file with doc_id as filename
            # Spotlight indexes text files in Documents
            file_path = OUTPUT_DIR / f"doc_{doc_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as out:
                # Put doc_id at top so we can extract it from search results
                out.write(f"DOCID:{doc_id}\n\n{text}")

            if doc_count % 500 == 0:
                print(f"  Extracted {doc_count} documents...")

    print(f"\n✓ Extracted {doc_count} documents to {OUTPUT_DIR}")
    print(f"\nWaiting for Spotlight to index...")
    print("This may take 1-2 minutes for ~3600 documents")
    print("\nYou can check indexing status:")
    print(f"  mdfind -onlyin {OUTPUT_DIR} -count DOCID")
    print(f"  (should show ~{doc_count} when fully indexed)")

    # Wait a bit and check initial indexing
    time.sleep(3)
    result = subprocess.run(
        ["mdfind", "-onlyin", str(OUTPUT_DIR), "-count", "DOCID"],
        capture_output=True,
        text=True
    )
    indexed = result.stdout.strip()
    print(f"\nCurrently indexed: {indexed} files")

    if int(indexed) < doc_count * 0.5:
        print("\nWARNING: Spotlight is still indexing. Wait 1-2 minutes before querying.")
        print("To force re-index if needed:")
        print(f"  mdimport {OUTPUT_DIR}")

    return True

def query_spotlight(query_text, top_k=10):
    """Query Spotlight and return ranked document IDs."""
    cmd = ["mdfind", "-onlyin", str(OUTPUT_DIR), query_text]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return []

    # Parse file paths and extract doc IDs from DOCID: prefix in content
    doc_ids = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue

        file_path = Path(line)
        if not file_path.exists():
            continue

        # Read first line to get DOCID
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('DOCID:'):
                    doc_id = first_line[6:]  # Remove 'DOCID:' prefix
                    doc_ids.append(doc_id)
        except:
            continue

    return doc_ids[:top_k]

def run_benchmark(output_file):
    """Run Spotlight benchmark on all test queries."""
    if not QUESTIONS_FILE.exists():
        print(f"Error: {QUESTIONS_FILE} not found")
        return False

    if not OUTPUT_DIR.exists():
        print(f"Error: Corpus not prepared at {OUTPUT_DIR}")
        print("Run: python spotlight-nfcorpus.py prepare")
        return False

    print(f"\n=== Spotlight nfcorpus Benchmark ===")
    print(f"Reading queries from: {QUESTIONS_FILE}")
    print(f"Output: {output_file}\n")

    query_count = 0
    latencies = []  # Track query latencies

    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as qf, \
         open(output_file, 'w', encoding='utf-8') as out:

        for line in qf:
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue

            query_id, query_text = parts
            query_count += 1

            # Measure query latency
            start = time.time()
            results = query_spotlight(query_text, top_k=10)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

            # Write in same format as warp-cli querycsv:
            # query_id\tdoc1,doc2,doc3,...
            result_str = ','.join(results)
            out.write(f"{query_id}\t{result_str}\n")

            if query_count % 50 == 0:
                print(f"Query {query_count}: {query_text[:60]}... -> {len(results)} results, {elapsed_ms:.1f}ms")

    # Calculate latency statistics
    latencies.sort()
    mean_lat = sum(latencies) / len(latencies)
    median_lat = latencies[len(latencies) // 2]
    p95_lat = latencies[int(len(latencies) * 0.95)]
    p99_lat = latencies[int(len(latencies) * 0.99)]

    print(f"\n✓ Completed {query_count} queries")
    print(f"Results saved to: {output_file}")
    print(f"\n=== Spotlight Latency Statistics ===")
    print(f"Mean:   {mean_lat:.1f}ms")
    print(f"Median: {median_lat:.1f}ms")
    print(f"P95:    {p95_lat:.1f}ms")
    print(f"P99:    {p99_lat:.1f}ms")
    print(f"\nTo score:")
    print(f"  cargo run --quiet --release -p trec-score -- {output_file} {COLLECTION_MAP} {QRELS_FILE}")

    return True

def score_results(output_file):
    """Score results using trec-score."""
    if not Path(output_file).exists():
        print(f"Error: {output_file} not found")
        return False

    print(f"\n=== Scoring Spotlight Results ===\n")

    cmd = [
        "cargo", "run", "--quiet", "--release", "-p", "trec-score", "--",
        str(Path(output_file).resolve()),
        str(COLLECTION_MAP),
        str(QRELS_FILE)
    ]

    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    return result.returncode == 0

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "prepare":
        prepare_corpus()

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python spotlight-nfcorpus.py query OUTPUT_FILE")
            return
        output_file = sys.argv[2]
        run_benchmark(output_file)

    elif command == "score":
        if len(sys.argv) < 3:
            print("Usage: python spotlight-nfcorpus.py score OUTPUT_FILE")
            return
        output_file = sys.argv[2]
        score_results(output_file)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)

if __name__ == "__main__":
    main()
