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

make warp-cli treccovid-testset EXTRA_FEATURES=deterministic

echo ensuring presence of pytrec-eval...
(source env/*/activate && uv pip install pytrec-eval >/dev/null)

echo running TRECCOVID queries with ${mode}...
./warp-cli "$mode" testset/treccovid/questions.test.tsv "$output"

echo scoring TRECCOVID NDCG@10...
(source env/*/activate && python score.py "$output" testset/treccovid/collection_map.json testset/treccovid/qrels.test.json)
