# Witchcraft (formerly known as Rust-Warp) #

This is a from-scratch reimplementation of Stanford's XTR-Warp
semantic search engine ( https://github.com/jlscheerer/xtr-warp ).

## Pickbrain: semantic search over your Claude Code history ##

![pickbrain](pickbrain.png)

Included as an example is **pickbrain**, a CLI that indexes your Claude Code
session transcripts, memory files, and authored documents into a Witchcraft
database for fast semantic search. Ever wondered "what was that conversation
where Claude helped me fix the auth middleware?" — pickbrain finds it.

```
make pickbrain
pickbrain --update              # ingest new/changed sessions
pickbrain auth middleware fix    # search across all sessions
pickbrain --session <UUID> auth  # search within one session
pickbrain --dump <UUID>          # print full conversation
```

The source lives in `examples/pickbrain.rs` and demonstrates how to use
Witchcraft as a library: document ingestion, embedding, indexing, and hybrid
search — all in about 400 lines of Rust.

---

To run, you will need the XTR weights released by Google Deepmind, and
this repo contains python scripts for downloading them from Huggingface
and quantizing to GGUF format:

* requirements.txt is for use with uv, to set up an env with
  the required python packages for the download scripts, as in:

```
uv venv env
source env/bin/activate
uv pip install -r requirements.txt
```

* downloadweights.py downloads the XTR weights, adds the extra
  XTRLinear layer, and exports into xtr.safetensors. It then quantizes
  to GGUF for use by the quantized T5 backend.

## Cheat: Do everything with make: ##
```
make warp-cli
```

Should also get you going and end up with a symlink to the warp-cli
executable in the top-level checkout directory.

## Creating an index: ##

For testing, I have used the BEIR download script from XTR-Warp to download
nfcorpus.  For your convenience, nfcorpus.tsv is included here, so you can run:
```
$ ./warp-cli readcsv datasets/nfcorpus.tsv
```
With all the documents in place, we can now create embeddings for them, with:
```
$ ./warp-cli embed
```
This will look for documents that lack embeddings, and create and insert them.
The embeddings are compressed with Haar wavelet transforms and rANS entropy
coding before storage.

Next we create the index over the embeddings, with:

```
$ ./warp-cli index
```

All state gets persisted in mydb.sqlite, and you can abort the indexer and
it will pick up where it left off. To start over, you can just delete
mydb.sqlite and get a blank slate. Sqlite is just a practical way of gathering
everything, it is not involved in the actual index search.

## Querying the index ##

When you have the index, you can query it with:

```
$ ./warp-cli query "does milk intake cause acne in teenagers?"
```

And hopefully get a bunch of relevant answers. You can also try other
variations of this, instead of "query" you can also use "hybrid", which
combines semantic search with the BM25 search functionality that comes
standard with sqlite.

There are also versions of these commands to run over tab-separate CSV files,
useful for benchmarking. Please refer to the source code, or see the
nfcorpus-score.sh and scifact-score.sh scripts for examples of doing this
with datasets from BEIR.

## Feature flags ##

When building, exactly one T5 backend must be enabled:
- `t5-quantized` -- GGUF quantized weights via candle (default)
- `t5-openvino` -- OpenVINO inference backend

Other flags:
- `metal` -- macOS GPU acceleration (Apple Silicon only)
- `accelerate` -- macOS BLAS via Accelerate framework
- `fbgemm` -- fbgemm-rs packed GEMM (bf16 weights, faster on x86)
- `hybrid-dequant` -- F32 attention + Q4K FFN with fused gated-gelu (x86, requires `fbgemm`)
- `napi` -- Node.js native module via napi-rs
- `embed-assets` -- bake weights into binary
- `progress` -- progress bars for CLI

Platform-specific recommended features:
- **Apple Silicon**: `t5-quantized,metal,accelerate`
- **Intel Mac (x86_64)**: `t5-quantized,accelerate,hybrid-dequant,fbgemm`

## Using as Node module ##

We include a Makefile for building with Napi-rs and copying things around.
```
make module
```

Builds target/release/warp.node. Which loads the compressed weights from the
"assets" dir.

## Windows build ##

To build for 64-bit x86 Windows, install the xwin tool with:

```
cargo install cargo-xwin
rustup target add x86_64-pc-windows-msvc
```

and build with

```
make win
```

## Unit tests and Code Coverage

```
cargo install cargo-nextest
cargo install cargo-llvm-cov
cargo install llvm-tools-preview
make test
```

NOTICE that nextest is necessary, simply using "cargo test" will lead
to random test failures, because individual tests run in the same process,
leading to "history effects".

# License

Unless otherwise noted:

```
Copyright (c) 2026 Dropbox Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
