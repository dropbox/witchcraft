# Witchcraft

(formerly known as Rust-Warp)

This is a from-scratch reimplementation of Stanford's XTR-Warp semantic search
engine ( https://github.com/jlscheerer/xtr-warp ) in safe rust, using a
single-file SQLite database as backing storage, making it suitable for
client-side deployment. It runs completely stand-alone on your device, needs no
API keys, no vector database, no chunking strategy, no fancy re-rankers, and it
is lightning fast (14ms p.95 end-to-end search latency on NFCorpus, at 34%
NDCG@10, on an Apple Macbook Pro M4 Max, more than twice as fast as the
original XTR-WARP on server-class hardware.)

Version 0.2.0 adds native support for Windows GPUs without relying on OpenVINO
etc, a much more compact on-disk index format, better scalability with index
updates, better search accuracy, and many updates to the Pickbrain agent memory
tool, among them the ability to register as a launchd service on MacOS, so that
the index is kept updated in the background, reducing the risk of having to
wait on index updates during queries.

![pickbrain](pickbrain.png)

# Prerequisites #

Needs uv, make, and the rust toolchain.

# Building and Running #
The default build uses our ModernBERT retrieval model with 96-dimensional
token embeddings and token gating. Make downloads a compressed archive of the
safetensors weights, config, and tokenizer from the
[versioned model release](https://github.com/dropbox/witchcraft/releases/tag/modernbert-96d-gated-v1)
into `assets/` and derives the quantized GGUF model locally:
```
make warp-cli
```

Downloads use public URLs with `curl`; no GitHub account or GitHub CLI is required.

The same weights support the unquantized backend
(`make warp-cli ENCODER=modernbert`). The weights derive from IBM's
[Granite Embedding English R2](https://huggingface.co/ibm-granite/granite-embedding-english-r2),
licensed under Apache 2.0. Our fine-tuning and other modifications are
Copyright (c) 2026 Dropbox Inc. and covered by the repository's Apache 2.0 license.
The archive includes the repository `LICENSE`, the upstream `LICENSE.granite`,
a `NOTICE` identifying our modifications, and `SHA256SUMS` for verifying downloads.
Existing local assets are kept. To use your own checkpoint, export it with
`env/bin/python scripts/export_modernbert.py <checkpoint> assets`, then quantize
it with `cargo run -p quantize --release -- assets/modernbert.safetensors assets/modernbert.gguf`.

For Google's original XTR model, run `make warp-cli ENCODER=t5-quantized`;
the included Python scripts download its weights from Hugging Face and
quantize them to GGUF.

## Creating an index: ##
For testing, we used the BEIR download script from XTR-Warp to download
nfcorpus and check that we could replicate their results.
For your convenience, nfcorpus.tsv is included here, so you can run:
```
$ make nfcorpus
```
With all the nfcorpus documents imported, embeddings will be created,
and the index updated with them.

All state gets persisted in mydb.sqlite, and you can abort the indexer and
it will pick up where it left off. To start over, you can just delete
mydb.sqlite.

You can rerun the nfcorpus scoring result with:
```
$ make nfcorpus-score
```

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
your Pi, Claude Code, and OpenAI Codex session transcripts, memory files, and
authored documents into a Witchcraft database for fast semantic search. Ever
wondered "what was that conversation where I fixed the auth middleware?" —
pickbrain finds it, and lets you resume the session directly.

```
make pickbrain
./pickbrain auth middleware fix    # search across all sessions (auto-ingests new sessions)
./pickbrain --session <UUID> auth  # search within one session
./pickbrain --dump <UUID>          # print full conversation
```

Set `PRE_INGEST_COMMAND` to a shell command that runs before pickbrain checks for new sessions (for example, an `rsync` from another host). Set `EXTRA_CODEX_DIRS`, `EXTRA_CLAUDE_DIRS`, or `EXTRA_PI_DIRS` to additional data directories separated by `:` on Unix or `;` on Windows. Each entry should have the same layout as `~/.codex`, `~/.claude`, or `~/.pi/agent`, respectively. Pickbrain scans these alongside the default directories.

`scripts/sync_codex_sessions.sh` takes an SSH host and an absolute local directory. It copies only session `*.jsonl` files and writes the host to `pickbrain.remote`. Pickbrain labels those results as downloaded; press `r` in the browser to see the SSH command for that host.

```bash
PRE_INGEST_COMMAND="$PWD/scripts/sync_codex_sessions.sh your-ssh-alias $HOME/.pickbrain/remote-codex" \
EXTRA_CODEX_DIRS="$HOME/.pickbrain/remote-codex" ./pickbrain auth middleware fix
```

For an existing extra Codex directory, put its SSH host name in `<extra-codex-dir>/pickbrain.remote` to tag those sessions.

The source lives in `examples/pickbrain/` and demonstrates how to use
Witchcraft as a library: document ingestion, embedding, indexing, and hybrid
search. To install pickbrain as a skill/extension for Pi and as a skill for both Claude Code and Codex:

```
make pickbrain-install
```

For automatic ingestion, run Pickbrain in watcher mode:

```
make pickbrain
./pickbrain --watch
```

`pickbrain --watch` watches local Claude, Codex, and Pi directories, configured extra
directories, and Slack's IndexedDB blobs on macOS. It uses native notifications,
waits for 30 seconds without changes, and caps the delay at five minutes. Set
`--delay SECONDS` and `--max-delay SECONDS` to adjust these intervals. It checks
for outstanding work at startup and rescans when notifications are lost. New or
recreated source directories are detected automatically.

On macOS, FSEvents can defer notifications for session files kept open by Codex.
The watcher also checks session file sizes and change times every quiet-delay
interval (30 seconds by default). This reads metadata only; ingestion starts when
changes are detected and the debounce delay expires.

The watcher spawns its own executable with `--ingest-only --quiet`, with one child at a time.
It logs each completed run as `pickbrain: no changes` or
`pickbrain: ingested and indexed N documents`. Counts refer to processed documents
(conversation turns and files), including documents reread from changed sessions.
The summary appears after indexing succeeds; errors remain visible.
The watcher retains changes arriving during ingestion for another pass. Failed children
are retried after the quiet delay. Model memory and GPU resources belong to the
child and are released when it exits; watcher mode enters before database,
model, or GPU initialization and waits between notifications and metadata checks. Pickbrain's database, logs, and watermarks
do not trigger ingestion. The normal ingestion lock also protects against
interactive Pickbrain runs. Incomplete headless ingestion is retried even if
source watermarks have already advanced.

`make pickbrain-install` installs the single binary in `~/bin`, then registers or
updates the background watcher. To register an existing binary directly, run:

```
pickbrain --register
```

Registration starts the watcher immediately and at login, using a macOS user
LaunchAgent (`com.dropbox.pickbrain`), Linux user systemd service
(`pickbrain.service`), or Windows Task Scheduler logon task (`Pickbrain-<user SID>`).
It replaces the same job on subsequent registrations. `--delay` and `--max-delay`
also work with `--register`. Registration saves the current working directory,
`PATH`, source-directory settings, `PICKBRAIN_DIR`, `WARP_ASSETS`, and
`PRE_INGEST_COMMAND` in `~/.pickbrain/watch.json`, so the background job can use the
same settings outside your shell. Re-register to change them. Logs, including
ingestion summaries and errors, go to `~/.pickbrain/watch.log`:

```
tail -f ~/.pickbrain/watch.log
```

On macOS, stop the job with
`launchctl bootout gui/$(id -u)/com.dropbox.pickbrain`; remove
`~/Library/LaunchAgents/com.dropbox.pickbrain.plist` to prevent launch at the next
login. On Linux use `systemctl --user disable --now pickbrain.service`; on Windows,
stop and delete the named task in Task Scheduler. Linux registration requires a
running user systemd manager. On Windows, `USERPROFILE` supplies the home directory
when `HOME` is unset. The watcher uses its own executable path, so symlinks and
`PATH` resolution need no configuration. Remote hosts still need a separate
periodic synchronization job, since local filesystem notifications cannot detect
changes on those hosts.

This puts the binary on your `PATH` and installs the skill/extension definitions so
you can use pickbrain directly from Pi, Claude Code, or Codex to answer questions requiring
global knowledge of all your projects:

![skill](skill.png)

# More build info #
## Feature flags ##

When building, exactly one encoder backend must be enabled:
- `t5-quantized` -- GGUF quantized weights via candle
- `modernbert-quantized` -- GGUF ModernBERT weights (default)
- `modernbert` -- full-precision ModernBERT weights

Other flags:
- `metal` -- Candle Metal acceleration
- `neso-metal` -- Neso-generated Metal kernels
- `neso-d3d12` -- Neso-generated D3D12 kernels
- `fbgemm` -- fbgemm-rs packed GEMM (bf16 weights, faster on x86)
- `hybrid-dequant` -- F32 attention + Q4K FFN with fused gated-gelu (x86, requires `fbgemm`)
- `napi` -- Node.js native module via napi-rs
- `python` -- Python extension module via PyO3/maturin
- `embed-assets` -- bake weights into binary
- `progress` -- progress bars for CLI

Neso GPU kernels are cached as compressed archives in `kernels/out/` and embedded
in the binary. Cargo rebuilds them when the Neso environment is available (at
`../neso`, or `NESO_DIR`); otherwise it uses the checked-in caches, including their
Rust loaders. This fallback needs no Python, Neso, or shader compiler. Metal uses
the scalar kernels on both Mac architectures; Windows uses the DXIL cache.
To refresh the caches after changing kernels, run `make neso-kernels-metal-nosimd`
and `make neso-kernels-hlsl` with Neso installed. Cache generation also needs `zstd`.

Platform-specific recommended features (these are what `make` uses automatically):
- **Apple Silicon**: `modernbert-quantized,neso-metal`
- **Intel Mac (x86_64)**: `modernbert-quantized,neso-metal`
- **Intel Windows (x86_64)**: `modernbert-quantized,neso-d3d12`
- **Linux x86_64 (CPU)**: `modernbert-quantized,fbgemm,hybrid-dequant`
- **Linux x86_64 (CUDA)**: `modernbert-quantized,cuda`
- **Linux ARM (Graviton, Pi, Ampere)**: `modernbert-quantized` (`fbgemm`/`hybrid-dequant` are x86-only)

## Using as a Python module ##

Build a wheel for your current platform (auto-selects the right backend features):

```
make python-wheel
```

Or install directly into the repo-managed virtualenv for development:

```
make python-dev
```

The Makefile picks the correct feature set automatically (`metal` on Apple Silicon,
`fbgemm,hybrid-dequant` on Intel, `cuda` on Linux with a GPU). The wheel works on
Python 3.8+, including 3.14+.

To build/install the extension and run the Python test suite:

```
make python-test
```

Override `PYTHON_TEST_ARGS` to pass custom pytest arguments, for example
`make python-test PYTHON_TEST_ARGS="-q tests/test_witchcraft.py"`.

Then in Python:

```python
import witchcraft

wc = witchcraft.Witchcraft('/path/to/db.sqlite', '/path/to/assets')

# Add documents (fire-and-forget, processed by background thread)
wc.add('550e8400-e29b-41d4-a716-446655440000',
       '2024-01-15T10:00:00Z',
       '{"source": "dropbox"}',
       'The document text goes here')

# Build pending embeddings and index data; blocks until complete
wc.index()

# Hybrid semantic + BM25 search
results = wc.search('does milk intake cause acne?', threshold=0.3, top_k=5)
for r in results:
    print(r['score'], r['body'])

# Score individual sentences against a query
scores = wc.score('acne and diet', ['milk causes acne', 'exercise helps skin'])

# Shut down the background indexer cleanly
wc.shutdown()
```

`search` returns a list of dicts with keys: `score`, `metadata`, `body`, `idx`, `date`.

## Using as Node module ##

```
make module
```
Builds a universal macOS binary at `target/release/warp-macos-universal.node`
(lipo'd from aarch64 + x86_64 builds with platform-appropriate features).

To use in another project:

```
cd /path/to/your-project
npm install /path/to/witchcraft
```

Then in JavaScript:

```js
const { Witchcraft } = require('warp');
const wc = new Witchcraft('/path/to/db.sqlite', '/path/to/assets');
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

## Preparing a PR

Before submitting a PR, please make sure that
```
make test
```
and
```
make nfcorpus
```

Run, and that

```
make nfcorpus-score
```

Runs and scores in the 0.31-0.33 range

NOTICE that nextest is necessary, simply using "cargo test" will lead
to random test failures, because individual tests run in the same process,
leading to "history effects".

## TRECCOVID ##

TRECCOVID is prepared from the BEIR `trec-covid` archive on demand:

```
make treccovid-files
make treccovid
make treccovid-score
```

`make treccovid` builds `mydb.sqlite` from `datasets/treccovid.tsv`, and
`make treccovid-score` runs `treccovid-score.sh` against the test queries and
prints NDCG@10.

Scoring runs entirely in Rust, without Python or `pytrec_eval`:

```
make trec-score
target/release/trec-score output.txt testset/nfcorpus/collection_map.json testset/nfcorpus/qrels.test.json
```

The tool reads `querycsv` results and prints mean NDCG@10 with trec_eval's
linear relevance gains. An optional fourth argument changes the cutoff.
Use a JSON `null` collection map when results already contain original document
IDs. Like the previous scorer, it averages submitted queries that have qrels;
empty results count as zero and missing or unjudged queries are excluded.
The standalone workspace tool builds without encoder features, weights, or GPU
dependencies. NFCorpus, TRECCOVID, SciFact, and Spotlight scoring use it.

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
