# Layers

Witchcraft's layer model adds overlay-aware indexing on top of the basic
document model. It is inspired by union/overlay filesystems (like OverlayFS):
a read-only baseline holds the bulk of your data, and lightweight overlays
capture deltas without copying the whole index.

## When to use layers

The basic document model (`add_doc` / `search`) is the right choice for most
applications. Use layers when:

- Your corpus is large and changes arrive as branch-scoped deltas (e.g., a
  monorepo where each feature branch modifies a handful of files).
- You need multiple concurrent views of the same index without duplicating it.
- You want to expire stale views cheaply and revive them later.

If your corpus fits comfortably in a single mutable index and you don't need
branching, layers add complexity for no benefit. Stick with the basic model.

For small-scale branching (2-3 concurrent views), querying separate database
files is simpler and works fine. Layers pay off when the index is too large to
duplicate: they share the embedding store and IVF globally, so overlays only
store delta rows. Content-hash dedup means a branch that touches 50 files out
of 100K shares 99.95% of embeddings with the baseline.

## How it works

A layer tree is a parent-pointer tree stored in the `layer` table. The typical
topology is flat: one baseline (root) plus N overlay children.

```
baseline (mutable)
├── overlay-branch-A
├── overlay-branch-B
└── overlay-branch-C
```

**Resolution**: when you query from an overlay's perspective, witchcraft walks
the chain from overlay to root. The shallowest entry for each chunk wins.
Tombstones in an overlay suppress baseline entries. For the common two-layer
case this is a simple two-way dedup, not a recursive walk.

**Lifecycle**: layers move through three statuses:
- `active` -- accepting writes
- `sealed` -- advisory immutability (application enforced)
- `compacted` -- replaced by compaction output, pending cleanup

## API

```rust
// Create a baseline
let baseline = db.create_layer(None, "main", None)?;

// Create an overlay for a branch
let overlay = db.create_layer(Some(baseline), "feature/foo", Some(metadata_json))?;

// Resolve the full chain (for query context)
let chain = db.layer_chain(overlay)?;
// chain.chain == [(overlay_id, 0), (baseline_id, 1)]

// Freeze an overlay
db.seal_layer(overlay)?;

// Compact a deep sub-chain into a single layer
let merged = db.compact_layers(leaf_id, Some(stop_at_id))?;

// Clean up (fails if layer has children -- reparent or delete them first)
db.delete_layer(overlay)?;
```

## Why a single mutable baseline

For large corpora the index can be tens of gigabytes. Maintaining multiple
baselines is not feasible. Instead, the baseline is mutated in place when the
mainline source advances. Overlays automatically see baseline changes through
layer resolution -- no rebasing needed unless the overlay and baseline modify
the same content.

Stale overlays are cheap to keep (only delta rows) and cheap to revive (most
embeddings still exist in the content-addressed embedding store). Expiry policy
is application-defined.

## What layers don't do

Layers are a structural primitive. They don't affect embedding, IVF indexing,
or scoring. The IVF is global and layer-agnostic -- layer resolution is a
post-filter after candidate retrieval. Ranking bias (recency, proximity,
language preference) belongs in the application layer.
