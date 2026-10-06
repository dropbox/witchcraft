rm -rf output.txt

cargo run --release --features t5-quantized,metal --bin warp-cli -- querycsv "$HOME/src/xtr-warp/beir/scifact/questions.test.tsv" output.txt &&\

cargo run --quiet --release -p trec-score -- output.txt "$HOME/src/xtr-warp/beir/scifact/collection_map.json" "$HOME/src/xtr-warp/beir/scifact/qrels.test.json"
