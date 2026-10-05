SHELL := /bin/bash
PATH := $(HOME)/.cargo/bin:$(HOME)/.local/bin:/usr/local/cuda/bin:$(PATH)
export PATH
.DEFAULT_GOAL := build

# Auto-detect platform and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
  NVCC := $(or $(shell which nvcc 2>/dev/null),$(wildcard /usr/local/cuda/bin/nvcc),$(wildcard /opt/cuda/bin/nvcc))
endif

ENCODER ?= modernbert-quantized
NESO_DIR ?= ../neso
NESO_PYTHON ?= $(NESO_DIR)/env/bin/python
#ENCODER ?= modernbert
#ENCODER ?= t5-quantized

# Determine features and flags based on platform
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    # Apple Silicon: Metal GPU + Accelerate BLAS
    CLI_FEATURES := $(ENCODER),neso-metal,progress,sqlite
    CAPI_FEATURES := $(ENCODER),neso-metal,capi-embed-cache
    NAPI_FEATURES := $(ENCODER),neso-metal,napi
    PYTHON_FEATURES := $(ENCODER),neso-metal,python
    RUSTFLAGS_EXTRA :=
    TARGET := aarch64-apple-darwin
  else
    # Intel Mac: Neso-generated Metal kernels (scalar Metal 2.4 fallback)
    CLI_FEATURES := $(ENCODER),neso-metal,progress,sqlite
    CAPI_FEATURES := $(ENCODER),neso-metal,capi-embed-cache
    NAPI_FEATURES := $(ENCODER),neso-metal,napi
    PYTHON_FEATURES := $(ENCODER),neso-metal,python
    RUSTFLAGS_EXTRA := -C target-feature=+avx2,+fma
    TARGET := x86_64-apple-darwin
  endif
  PICKBRAIN_FEATURES := $(CLI_FEATURES),embed-assets
else ifeq ($(UNAME_S),Linux)
  ifneq ($(NVCC),)
    CLI_FEATURES := $(ENCODER),cuda,hybrid-dequant,progress,sqlite
    CAPI_FEATURES := $(ENCODER),cuda,hybrid-dequant,capi-embed-cache
    NAPI_FEATURES := $(ENCODER),cuda,hybrid-dequant,napi
    PYTHON_FEATURES := $(ENCODER),cuda,hybrid-dequant,python
  else ifeq ($(UNAME_M),aarch64)
    # Linux ARM (Graviton, Pi, Ampere): fbgemm/hybrid-dequant are x86-only
    CLI_FEATURES := $(ENCODER),progress
    CAPI_FEATURES := $(ENCODER),capi-embed-cache
    NAPI_FEATURES := $(ENCODER),napi
    PYTHON_FEATURES := $(ENCODER),python
  else
    # Linux x86_64 CPU-only
    CLI_FEATURES := $(ENCODER),fbgemm,hybrid-dequant,progress
    CAPI_FEATURES := $(ENCODER),fbgemm,hybrid-dequant,capi-embed-cache
    NAPI_FEATURES := $(ENCODER),fbgemm,hybrid-dequant,napi
    PYTHON_FEATURES := $(ENCODER),fbgemm,hybrid-dequant,python
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
BEIR_CACHE ?= .cache/beir
RUSTFMT_RS_FILES := $(shell git ls-files '*.rs' \
	':!build.rs' \
	':!src/main.rs' \
	':!src/db.rs' \
	':!src/histogram.rs' \
	':!src/napi.rs' \
	':!src/python.rs' \
	':!src/sql*.rs' \
	':!examples/**' \
	':!tools/**' \
	':!crates/**') tools/quantize/src/main.rs

VENV_DIR := $(abspath env)
PYTHON_BIN := $(VENV_DIR)/bin/python
MATURIN := $(VENV_DIR)/bin/maturin
PYTEST := $(VENV_DIR)/bin/pytest
PYTHON_TEST_ARGS ?= python/test_witchcraft.py

ifeq ($(UNAME_S),Darwin)
  CAPI_LIB := target/release/libwitchcraft.dylib
else ifeq ($(UNAME_S),Linux)
  CAPI_LIB := target/release/libwitchcraft.so
else
  CAPI_LIB := target/release/witchcraft.dll
endif

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
	uv pip install --python $(PYTHON_BIN) -r requirements.txt

python-build-deps: env/pyvenv.cfg | prereqs
	uv pip install --python $(PYTHON_BIN) maturin

# === Assets / weights ===

assets:
	mkdir -p assets

assets/xtr-config.json assets/xtr-tokenizer.json xtr.safetensors: env/bin/transformers | assets
	$(PYTHON_BIN) downloadweights.py

assets/xtr.gguf: xtr.safetensors | assets prereqs
	cargo run -p quantize xtr.safetensors assets/xtr.gguf

MODERNBERT_RELEASE := https://github.com/dropbox/witchcraft/releases/download/modernbert-96d-gated-v1

assets/modernbert-assets.tar.gz: | assets
	curl --fail --location --output $@.tmp $(MODERNBERT_RELEASE)/$(notdir $@)
	mv $@.tmp $@

assets/modernbert-config.json assets/modernbert-tokenizer.json assets/modernbert.safetensors assets/LICENSE assets/LICENSE.granite assets/NOTICE assets/SHA256SUMS: | assets/modernbert-assets.tar.gz
	tar -xzf assets/modernbert-assets.tar.gz -C assets $(notdir $@)

assets/modernbert.gguf: assets/modernbert.safetensors | assets prereqs
	cargo run -p quantize --release -- assets/modernbert.safetensors assets/modernbert.gguf

modernbert-assets: assets/modernbert-config.json assets/modernbert-tokenizer.json assets/modernbert.safetensors assets/LICENSE assets/LICENSE.granite assets/NOTICE assets/SHA256SUMS

modernbert-quantized-assets: assets/modernbert-config.json assets/modernbert-tokenizer.json assets/modernbert.gguf assets/LICENSE assets/LICENSE.granite assets/NOTICE assets/SHA256SUMS

ifeq ($(ENCODER),modernbert)
DOWNLOAD_TARGETS := modernbert-assets
else ifeq ($(ENCODER),modernbert-quantized)
DOWNLOAD_TARGETS := modernbert-quantized-assets
else
DOWNLOAD_TARGETS := assets assets/xtr-config.json assets/xtr-tokenizer.json assets/xtr.gguf
endif

download: prereqs $(DOWNLOAD_TARGETS)

# === Build targets ===

neso-kernels-metal:
	NESO_DIR=$(abspath $(NESO_DIR)) $(NESO_PYTHON) kernels/build.py metal

neso-kernels-metal-nosimd:
	NESO_DIR=$(abspath $(NESO_DIR)) $(NESO_PYTHON) kernels/build.py metal_nosimd

neso-kernels-hlsl:
	NESO_DIR=$(abspath $(NESO_DIR)) $(NESO_PYTHON) kernels/build.py hlsl

fmt:
	rustup run nightly rustfmt --color=never --unstable-features --skip-children --edition=2021 -- $(RUSTFMT_RS_FILES)

build: warp-cli dylib

buildemb: EXTRA_FEATURES += embed-assets
buildemb: warp-cli

warp-cli: prereqs download
	cargo build --release $(BUILD_TARGET) --features $(CLI_FEATURES)$(if $(EXTRA_FEATURES),$(comma)$(EXTRA_FEATURES)) --bin warp-cli
	ln -sf $(CLI_BIN) ./warp-cli

dylib: prereqs download
	cargo build --release --features $(CAPI_FEATURES)$(if $(EXTRA_FEATURES),$(comma)$(EXTRA_FEATURES)) --lib
	ln -sf $(CAPI_LIB) ./$(notdir $(CAPI_LIB))

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

macintel: prereqs neso-kernels-metal-nosimd download
	RUSTFLAGS='-C target-cpu=haswell' cargo build --release --target x86_64-apple-darwin --features $(ENCODER),neso-metal,progress,sqlite

winintel: prereqs neso-kernels-hlsl download
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features $(ENCODER),neso-d3d12,progress,sqlite

win: winintel

ifdef TARGET
  LIB_BIN := target/$(TARGET)/release/libwitchcraft.dylib
else
  LIB_BIN := target/release/libwitchcraft.dylib
endif

module: prereqs neso-kernels-metal neso-kernels-metal-nosimd download
	cargo build --release --target aarch64-apple-darwin --features $(ENCODER),neso-metal,napi
	cargo build --release --target x86_64-apple-darwin --features $(ENCODER),neso-metal,napi
	lipo -create target/aarch64-apple-darwin/release/libwitchcraft.dylib target/x86_64-apple-darwin/release/libwitchcraft.dylib -output target/release/warp-macos-universal.node
	ln -sf target/release/warp-macos-universal.node warp.node

test: prereqs download
	RUST_LOG=debug cargo llvm-cov nextest --release --features napi,$(CLI_FEATURES) --lcov --output-path lcov.info

#genhtml lcov.info

bench: prereqs
	cargo run -p modernbert-bench --release --features hybrid-dequant,cuda

%: %.zst
	zstd -dk $<

TRECCOVID_FILES := \
	datasets/treccovid.tsv \
	testset/treccovid/collection_map.json \
	testset/treccovid/qrels.test.json \
	testset/treccovid/questions.test.tsv
TRECCOVID_STAMP := .make-stamps/treccovid-data

$(TRECCOVID_STAMP): scripts/prepare_beir_dataset.py env/pyvenv.cfg | prereqs
	mkdir -p $(dir $@)
	env/bin/python scripts/prepare_beir_dataset.py trec-covid --output-name treccovid --cache-dir $(BEIR_CACHE)
	touch $@

$(TRECCOVID_FILES): $(TRECCOVID_STAMP)
	@if [ ! -f "$@" ]; then \
		rm -f "$(TRECCOVID_STAMP)"; \
		$(MAKE) "$(TRECCOVID_STAMP)"; \
	fi

treccovid-files: $(TRECCOVID_FILES)

treccovid-testset: \
	testset/treccovid/collection_map.json \
	testset/treccovid/qrels.test.json \
	testset/treccovid/questions.test.tsv

nfcorpus: prereqs datasets/nfcorpus.tsv
	make warp-cli EXTRA_FEATURES=deterministic
	rm -rf mydb.sqlite*
	$(CLI_BIN) readcsv datasets/nfcorpus.tsv
	$(CLI_BIN) embed
	$(CLI_BIN) index

treccovid: prereqs datasets/treccovid.tsv
	make warp-cli EXTRA_FEATURES=deterministic
	rm -rf mydb.sqlite*
	$(CLI_BIN) readcsv datasets/treccovid.tsv
	$(CLI_BIN) embed
	$(CLI_BIN) index

nfcorpus-score: prereqs env/pyvenv.cfg testset/nfcorpus/questions.test.tsv testset/nfcorpus/questions.test.tsv testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json
	make warp-cli EXTRA_FEATURES=deterministic
	echo ensuring presence of pytrec-eval...
	uv pip install --python $(PYTHON_BIN) pytrec-eval 2>/dev/null
	echo running queries...
	rm -rf output.txt
	$(CLI_BIN) querycsv testset/nfcorpus/questions.test.tsv output.txt
	echo scoring...
	$(PYTHON_BIN) score.py output.txt testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json

treccovid-score: treccovid-testset
	./treccovid-score.sh querycsv output-treccovid.txt

treccovid-hybrid-score: treccovid-testset
	./treccovid-score.sh hybridcsv output-treccovid.txt

treccovid-fulltext-score: treccovid-testset
	./treccovid-score.sh fulltextcsv output-treccovid-fulltext.txt

reindex:
	make warp-cli EXTRA_FEATURES=deterministic
	$(CLI_BIN) reindex

run: module
	ln -sf target/release/warp-macos-universal.node warp.node
	node index.cjs

python-wheel: python-build-deps download
	VIRTUAL_ENV=$(VENV_DIR) $(MATURIN) build --release $(BUILD_TARGET) --features $(PYTHON_FEATURES)

python-dev: python-build-deps download
	VIRTUAL_ENV=$(VENV_DIR) $(MATURIN) develop --features $(PYTHON_FEATURES)

python-test: python-dev
	uv pip install --python $(PYTHON_BIN) pytest
	$(PYTEST) $(PYTHON_TEST_ARGS)

distclean:
	rm -rf target env html xtr-base-en
	rm -f warp-cli warp.node pickbrain Cargo.lock lcov.info output.txt
	rm -f *.sqlite *.sqlite-shm *.sqlite-wal
	rm -f xtr.safetensors
	rm -f datasets/nfcorpus.tsv
	rm -f testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json testset/nfcorpus/questions.test.tsv
	rm -f datasets/treccovid.tsv
	rm -f testset/treccovid/collection_map.json testset/treccovid/qrels.test.json testset/treccovid/questions.test.tsv
	rm -rf .cache/beir
	@if [ -d assets ]; then \
		for path in assets/* assets/.[!.]* assets/..?*; do \
			[ -e "$$path" ] || continue; \
			[ "$$path" = "assets/LICENSE.txt" ] && continue; \
			rm -rf "$$path"; \
		done; \
	fi
	rm -rf .make-stamps .stamp* stamp-* *.stamp */.stamp* */*.stamp */stamp-*

.PHONY: \
	bench \
	build \
	buildemb \
	distclean \
	download \
	dylib \
	macintel \
	modernbert-assets \
	modernbert-quantized-assets \
	module \
	neso-kernels-hlsl \
	neso-kernels-metal \
	neso-kernels-metal-nosimd \
	nfcorpus \
	nfcorpus-score \
	pickbrain \
	pickbrain-install \
	prereqs \
	python-build-deps \
	python-dev \
	python-test \
	python-wheel \
	reindex \
	run \
	test \
	treccovid \
	treccovid-files \
	treccovid-score \
	treccovid-testset \
	warp-cli \
	win \
	winintel
