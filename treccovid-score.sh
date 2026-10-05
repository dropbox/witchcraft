#!/usr/bin/env bash
set -euo pipefail

mode="${1:-querycsv}"
output="${2:-output-treccovid.txt}"

case "$mode" in
	querycsv|hybridcsv|fulltextcsv) ;;
	*)
		echo "usage: $0 [querycsv|hybridcsv|fulltextcsv] [output-file]" >&2
		exit 2
		;;
esac

make warp-cli trec-score treccovid-testset EXTRA_FEATURES=deterministic

echo running TRECCOVID queries with ${mode}...
./warp-cli "$mode" testset/treccovid/questions.test.tsv "$output"

echo scoring TRECCOVID NDCG@10...
target/release/trec-score "$output" testset/treccovid/collection_map.json testset/treccovid/qrels.test.json
