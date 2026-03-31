# Auto-detect platform and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Determine features and flags based on platform
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    # Apple Silicon: Metal GPU + Accelerate BLAS
    CLI_FEATURES := t5-quantized,metal,progress
    NAPI_FEATURES := t5-quantized,metal,napi
    RUSTFLAGS_EXTRA :=
    TARGET := aarch64-apple-darwin
  else
    # Intel Mac: CPU-only with FBGEMM + hybrid-dequant
    CLI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,progress
    NAPI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,napi
    RUSTFLAGS_EXTRA := -C target-feature=+avx2,+fma
    TARGET := x86_64-apple-darwin
  endif
else ifeq ($(UNAME_S),Linux)
  CLI_FEATURES := t5-quantized,fbgemm,progress
  NAPI_FEATURES := t5-quantized,fbgemm,napi
  RUSTFLAGS_EXTRA :=
  TARGET :=
endif

# Binary path
ifdef TARGET
  CLI_BIN := target/$(TARGET)/release/warp-cli
  BUILD_TARGET := --target $(TARGET)
else
  CLI_BIN := target/release/warp-cli
  BUILD_TARGET :=
endif

export RUSTFLAGS += $(RUSTFLAGS_EXTRA)

# === Python environment ===

env/pyvenv.cfg:
	uv venv env

env/bin/transformers: env/pyvenv.cfg
	(source env/*/activate; uv pip install -r requirements.txt)

# === Assets / weights ===

assets:
	mkdir -p assets

assets/config.json assets/tokenizer.json xtr.safetensors: env/bin/transformers | assets
	(source env/*/activate; python downloadweights.py)

assets/xtr.gguf: xtr.safetensors | assets
	cargo run -p quantize-tool xtr.safetensors assets/xtr.gguf

assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml:
	python quantize-int4.py

download: assets assets/config.json assets/tokenizer.json assets/xtr.gguf

# === Build targets ===

warp-cli: download
	cargo build --release $(BUILD_TARGET) --features $(CLI_FEATURES) --bin warp-cli
	ln -sf target/$(TARGET)/release/warp-cli ./warp-cli

claude-index: download
	cargo build --release $(BUILD_TARGET) --features $(CLI_FEATURES) --bin claude-index
	ln -sf target/$(TARGET)/release/claude-index ./claude-index

macintel:
	RUSTFLAGS='-C target-cpu=haswell' cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,progress

winintel: download
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features t5-openvino,fbgemm,progress

ifdef TARGET
  LIB_BIN := target/$(TARGET)/release/libwarp.dylib
else
  LIB_BIN := target/release/libwarp.dylib
endif

module:
	cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,napi
	cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,napi
	lipo -create target/aarch64-apple-darwin/release/libwarp.dylib target/x86_64-apple-darwin/release/libwarp.dylib -output target/release/warp-macos-universal.node

test: download
	RUST_LOG=debug cargo llvm-cov nextest --release --features napi,$(CLI_FEATURES) --lcov --output-path lcov.info
	genhtml lcov.info

bench:
	cargo run -p t5-bench --release --features hybrid-dequant,ov,fbgemm

# === Dataset targets ===

nfcorpus: warp-cli
	rm -rf mydb.sqlite
	$(CLI_BIN) readcsv datasets/nfcorpus.tsv
	$(CLI_BIN) embed
	$(CLI_BIN) index

nfcorpus-score: warp-cli
	$(CLI_BIN) hybridcsv ~/src/xtr-warp/beir/nfcorpus/questions.test.tsv warp-results.txt

run: module
	ln -sf target/release/warp-macos-universal.node warp.node
	node index.cjs

.PHONY: download build warp-cli claude-index module win test bench nfcorpus nfcorpus-score run
