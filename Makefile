SHELL := /bin/bash
.DEFAULT_GOAL := build

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
  PICKBRAIN_FEATURES := $(CLI_FEATURES),embed-assets
else ifeq ($(UNAME_S),Linux)
  NVCC := $(or $(shell which nvcc 2>/dev/null),$(wildcard /usr/local/cuda/bin/nvcc),$(wildcard /opt/cuda/bin/nvcc))
  ifneq ($(NVCC),)
    CLI_FEATURES := t5-quantized,cuda,progress
    NAPI_FEATURES := t5-quantized,cuda,napi
  else
    CLI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,progress
    NAPI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,napi
  endif
  PICKBRAIN_FEATURES := $(CLI_FEATURES),embed-assets
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

EXTRA_FEATURES :=
comma := ,
export RUSTFLAGS += $(RUSTFLAGS_EXTRA)

# === Prerequisites ===

prereqs:
	@missing=0; \
	for tool in uv cargo rustc; do \
		if ! command -v $$tool >/dev/null 2>&1; then \
			echo "missing required tool: $$tool" >&2; \
			missing=1; \
		fi; \
	done; \
	if [ $$missing -ne 0 ]; then \
		echo "Install uv and the Rust toolchain, then rerun make." >&2; \
		exit 1; \
	fi

# === Python environment ===

env/pyvenv.cfg: | prereqs
	uv venv env

env/bin/transformers: env/pyvenv.cfg requirements.txt | prereqs
	(source env/*/activate && uv pip install -r requirements.txt)

# === Assets / weights ===

assets:
	mkdir -p assets

assets/config.json assets/tokenizer.json xtr.safetensors: env/bin/transformers | assets
	(source env/*/activate && python downloadweights.py)

assets/xtr.gguf: xtr.safetensors | assets prereqs
	cargo run -p quantize-tool xtr.safetensors assets/xtr.gguf

assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml: | prereqs
	python quantize-openvino.py

download: prereqs assets assets/config.json assets/tokenizer.json assets/xtr.gguf

ovdownload: prereqs assets/config.json assets/tokenizer.json assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml

# === Build targets ===

build: warp-cli

buildemb: EXTRA_FEATURES += embed-assets
buildemb: warp-cli

warp-cli: prereqs download
	cargo build --release $(BUILD_TARGET) --features $(CLI_FEATURES)$(if $(EXTRA_FEATURES),$(comma)$(EXTRA_FEATURES)) --bin warp-cli
	ln -sf target/$(TARGET)/release/warp-cli ./warp-cli

pickbrain: prereqs download
	cargo build --release $(BUILD_TARGET) --features $(PICKBRAIN_FEATURES) --example pickbrain
	ln -sf target/$(TARGET)/release/examples/pickbrain ./pickbrain

pickbrain-install: pickbrain
	mkdir -p ~/bin ~/.claude/skills/pickbrain ~/.codex/skills/pickbrain ~/.pi/agent/skills/pickbrain ~/.pi/agent/extensions/pickbrain
	ln -f $(realpath pickbrain) ~/bin/pickbrain
	rm -f ~/.claude/skills/pickbrain/skill.md ~/.codex/skills/pickbrain/skill.md ~/.pi/agent/skills/pickbrain/skill.md
	cp skills/pickbrain/SKILL.md ~/.claude/skills/pickbrain/SKILL.md
	cp skills/pickbrain-codex/SKILL.md ~/.codex/skills/pickbrain/SKILL.md
	cp skills/pickbrain-pi/SKILL.md ~/.pi/agent/skills/pickbrain/SKILL.md
	cp extensions/pickbrain-pi/index.ts ~/.pi/agent/extensions/pickbrain/index.ts

macintel: prereqs
	RUSTFLAGS='-C target-cpu=haswell' cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,progress

winintel: prereqs ovdownload
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features t5-openvino,fbgemm,progress

win: winintel

ifdef TARGET
  LIB_BIN := target/$(TARGET)/release/libwitchcraft.dylib
else
  LIB_BIN := target/release/libwitchcraft.dylib
endif

module: prereqs
	cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,napi
	cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,napi
	lipo -create target/aarch64-apple-darwin/release/libwitchcraft.dylib target/x86_64-apple-darwin/release/libwitchcraft.dylib -output target/release/warp-macos-universal.node
	ln -sf target/release/warp-macos-universal.node warp.node

test: prereqs download
	RUST_LOG=debug cargo llvm-cov nextest --release --features napi,$(CLI_FEATURES) --lcov --output-path lcov.info
	genhtml lcov.info

bench: prereqs
	cargo run -p t5-bench --release --features hybrid-dequant,ov,fbgemm

%: %.zst
	zstd -dk $<

nfcorpus: prereqs datasets/nfcorpus.tsv
	make warp-cli EXTRA_FEATURES=deterministic
	rm -rf mydb.sqlite*
	$(CLI_BIN) readcsv datasets/nfcorpus.tsv
	$(CLI_BIN) embed
	$(CLI_BIN) index

nfcorpus-score: prereqs testset/nfcorpus/questions.test.tsv testset/nfcorpus/questions.test.tsv testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json
	make warp-cli EXTRA_FEATURES=deterministic
	echo ensuring presence of pytrec-eval...
	uv pip install pytrec-eval 2>/dev/null
	echo running queries...
	$(CLI_BIN) querycsv testset/nfcorpus/questions.test.tsv output.txt
	echo scoring...
	python score.py output.txt testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json

run: module
	ln -sf target/release/warp-macos-universal.node warp.node
	node index.cjs

distclean:
	rm -rf target env html xtr-base-en openvino_model
	rm -f warp-cli warp.node pickbrain Cargo.lock lcov.info output.txt
	rm -f *.sqlite *.sqlite-shm *.sqlite-wal
	rm -f xtr.safetensors
	rm -f datasets/nfcorpus.tsv
	rm -f testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json testset/nfcorpus/questions.test.tsv
	@if [ -d assets ]; then \
		for path in assets/* assets/.[!.]* assets/..?*; do \
			[ -e "$$path" ] || continue; \
			[ "$$path" = "assets/LICENSE.txt" ] && continue; \
			rm -rf "$$path"; \
		done; \
	fi
	rm -rf .make-stamps .stamp* stamp-* *.stamp */.stamp* */*.stamp */stamp-*

.PHONY: prereqs download ovdownload build buildemb warp-cli pickbrain pickbrain-install module macintel winintel win test bench nfcorpus nfcorpus-score run distclean
