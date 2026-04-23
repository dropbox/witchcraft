---
name: pickbrain
description: Semantic search over past Claude Code conversations and memories. Use when the user wants to recall, find, or reference something from a previous Claude session — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my Claude history for Z".
allowed-tools: Bash
---

# Pickbrain — Semantic Search for Claude History

Search past Claude Code conversations, memory files, and authored files using semantic search.

## Installation

Build and install from the repo root:

```bash
make pickbrain
# then copy or symlink the binary to somewhere on your PATH:
ln -sf "$(pwd)/pickbrain" ~/bin/pickbrain
```

The binary embeds its own model weights (`embed-assets` feature), so no separate assets directory is needed.

## Usage

Run `pickbrain` via Bash with the user's query:

```bash
pickbrain "$ARGUMENTS"
```

Pickbrain automatically ingests new sessions, memories, and project config files (CLAUDE.md, AGENTS.md, and their @ references) before each search.

## Interpreting Results

Each result includes:
- **Timestamp** and **project directory**
- **Session ID** and **turn number** — identifies the exact conversation turn
- **Matching text** — the relevant chunk from the conversation

Present results as a concise summary. Quote the most relevant excerpts. If the user wants to dig deeper into a specific session, use `--dump`:

```bash
pickbrain --dump <session-id> --turns <start>-<end>
```

## Filtering

To search within the current (calling) session:

```bash
pickbrain --current "<query>"
```

To search within a specific session by ID:

```bash
pickbrain --session <session-id> "<query>"
```

To exclude the current (calling) session from results:

```bash
pickbrain --exclude-current "<query>"
```

To filter by git branch:

```bash
pickbrain --branch <branch-name> "<query>"
pickbrain --branch . "<query>"          # . = current branch
```

To exclude specific sessions from results (comma-separated or repeated):

```bash
pickbrain --exclude <uuid1>,<uuid2> "<query>"
pickbrain --exclude <uuid1> --exclude <uuid2> "<query>"
```

To search only recent history:

```bash
pickbrain --since 24h "<query>"
pickbrain --since 7d "<query>"
pickbrain --since 2w "<query>"
```

## Notes

- First run requires a full ingest+embed pass (~7s). Subsequent searches auto-ingest incrementally.
- The database lives at `~/.pickbrain/pickbrain.db`.
- Results are ranked by semantic similarity — they may not contain the exact query words.
- The active session's JSONL is skipped during ingest if it was indexed less than 10 minutes ago. If the active session can't be detected (e.g. on Windows), everything is ingested eagerly.
