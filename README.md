# Witchcraft

(formerly known as Rust-Warp)

This is a from-scratch reimplementation of Stanford's XTR-Warp semantic search
engine ( https://github.com/jlscheerer/xtr-warp ) in safe rust, using a
single-file SQLite database as backing storage, making it suitable for
client-side deployment. It runs completely stand-alone on your device, needs to
API keys, no vector database, no chunking strategy, no fancy re-rankers, and it
is lightning fast (21ms p.95 end-to-end search latency on NFCorpus, at 33%
NDCG@10, on an Apple Macbook Pro M2 Max, more than twice as fast as the
original XTR-WARP on server-class hardware, at similar accuracy.)

![pickbrain](pickbrain.png)

# Building and Running #
To run, you will need the XTR weights released by Google Deepmind, and
this repo contains python scripts for downloading them from Huggingface
and quantizing to GGUF format. We can cheat by doing everything with Make:
```
make warp-cli
```
## Creating an index: ##

For testing, we used the BEIR download script from XTR-Warp to download
nfcorpus and check that we could replicate their results.
For your convenience, nfcorpus.tsv is included here, so you can run:
```
$ ./warp-cli readcsv datasets/nfcorpus.tsv
```
With all the nfcorpus documents imported, we can now create embeddings for them, with:
```
$ ./warp-cli embed
```
Next we create the index over the embeddings, with:
```
$ ./warp-cli index
```

All state gets persisted in mydb.sqlite, and you can abort the indexer and
it will pick up where it left off. To start over, you can just delete
mydb.sqlite.

## Querying the index ##

When you have the index, you can query it with:
```
$ ./warp-cli query "does milk intake cause acne in teenagers?"
```
And hopefully get a bunch of relevant answers. You can also try other
variations of this, instead of "query" you can also use "hybrid", which
combines semantic search with the BM25 search functionality that comes
standard with sqlite.

# Pickbrain: semantic search over your AI coding sessions #

Included as an example is **pickbrain** (screenshot above), a CLI that indexes
your Claude Code and OpenAI Codex session transcripts, memory files, and
authored documents into a Witchcraft database for fast semantic search. Ever
wondered "what was that conversation where I fixed the auth middleware?" —
pickbrain finds it, and lets you resume the session directly.

```
make pickbrain
./pickbrain auth middleware fix    # search across all sessions (auto-ingests new sessions)
./pickbrain --session <UUID> auth  # search within one session
./pickbrain --dump <UUID>          # print full conversation
```

The source lives in `examples/pickbrain/` and demonstrates how to use
Witchcraft as a library: document ingestion, embedding, indexing, and hybrid
search. To install pickbrain as a skill for both Claude Code and Codex:

```
make pickbrain-install
```

This puts the binary on your `PATH` and installs the skill definitions so
you can use pickbrain directly from either tool to answer questions requiring
global knowledge of all your projects:

![skill](skill.png)

# More build info #
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
- **Intel Windows (x86_64)**: `t5-openvino,fbgemm`

## Using as Node module ##

```
make module
```
Builds target/release/warp.node. Which loads the compressed weights from the
"assets" dir.

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
