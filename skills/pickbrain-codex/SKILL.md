---
name: pickbrain
description: Semantic search over past Pi, Claude Code, and Codex conversations and memories. Use when the user wants to recall, find, or reference something from a previous coding session — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my history for Z".
---

# Pickbrain — Semantic Search for AI Coding History

Search past Pi, Claude Code, and Codex conversations, memory files, and authored files using semantic search.

## Usage

Run `pickbrain` with the user's query:

```bash
pickbrain "$ARGUMENTS"
```

Pickbrain automatically ingests new Pi/Claude/Codex sessions, memories, and project config files (CLAUDE.md, AGENTS.md, and their @ references) before each search.

## Interpreting Results

Each result includes:
- **Timestamp** and **project directory**
- **Session ID** and **turn number** — identifies the exact conversation turn
- **Matching text** — the relevant chunk from the conversation

Present results as a concise summary. Quote the most relevant excerpts. To dig deeper into a specific session:

```bash
pickbrain --dump <session-id> --turns <start>-<end>
```

## Filtering

Search within the current (calling) session:

```bash
pickbrain --current "<query>"
```

Search within a specific session by ID:

```bash
pickbrain --session <session-id> "<query>"
```

Exclude the current (calling) session from results:

```bash
pickbrain --exclude-current "<query>"
```

Exclude specific sessions by ID (comma-separated or repeated):

```bash
pickbrain --exclude <uuid1>,<uuid2> "<query>"
pickbrain --exclude <uuid1> --exclude <uuid2> "<query>"
```

Search only recent history:

```bash
pickbrain --since 24h "<query>"
pickbrain --since 7d "<query>"
pickbrain --since 2w "<query>"
```

## Notes

- First run requires a full ingest+embed pass (~7s). Subsequent searches are incremental.
- The database lives at `~/.pickbrain/pickbrain.db`.
- Results are ranked by semantic similarity — they may not contain the exact query words.
- The active session's JSONL is skipped during ingest if it was indexed less than 10 minutes ago. If the active session can't be detected, everything is ingested eagerly.
