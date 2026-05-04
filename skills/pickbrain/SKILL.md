---
name: pickbrain
description: Semantic search over past Claude Code conversations, memories, and Slack messages. Use when the user wants to recall, find, or reference something from a previous Claude session or Slack conversation — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my history for Z", "what did someone say about X in Slack".
allowed-tools: Bash
---

# Pickbrain — Semantic Search for Claude History and Slack

Search past Claude Code conversations, Codex sessions, Slack messages, memory files, and authored files using semantic search.

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

Pickbrain automatically ingests new Claude/Codex sessions, Slack conversations (from the desktop app's local IndexedDB), memories, and project config files before each search.

## Interpreting Results

Each result includes:
- **Timestamp** and **project directory** (or **channel name** for Slack results)
- **Source** — `claude`, `codex`, or `slack`
- **Session ID** and **turn number** (Claude/Codex) or **channel + thread ID** (Slack)
- **Matching text** — the relevant chunk from the conversation

Slack results show `thr:TIMESTAMP` for threaded conversations. This thread ID can be passed to `--session` or `--dump` to filter/view that thread.

Present results as a concise summary. Quote the most relevant excerpts. If the user wants to dig deeper into a specific session, use `--dump`:

```bash
pickbrain --dump <session-id> [--turns <start>-<end>]
pickbrain --dump <channel-name> [--turns <start>-<end>] [--since 7d]
pickbrain --dump thr:<thread-ts>
```

Slack results include `(you: Name (@handle))` to identify the logged-in user.

## Filtering

To search within the current (calling) session:

```bash
pickbrain --current "<query>"
```

To search within a specific session, channel, or thread:

```bash
pickbrain --session <session-id> "<query>"
pickbrain --channel <channel-name> "<query>"
pickbrain --session thr:<thread-ts> "<query>"
```

`--channel` is an alias for `--session`.

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

To filter by source type (claude, codex, slack):

```bash
pickbrain --type slack "<query>"
pickbrain --type claude,codex "<query>"
```

To filter Slack results by DM or non-DM:

```bash
pickbrain --dm "<query>"          # only DMs
pickbrain --no-dm "<query>"       # only channels (exclude DMs)
```

To show only Slack conversations with unread messages:

```bash
pickbrain --unread "<query>"
```

To control the number of results (0 = unlimited; default: unlimited in TUI, 20 in pipe):

```bash
pickbrain -n 50 "<query>"
pickbrain -n 0 "<query>"          # all results
```

When any filter is active (`--type`, `--session`, `--channel`, `--dm`, `--no-dm`, `--unread`, `--since`), the query can be omitted to browse matching results sorted by date:

```bash
pickbrain --type slack --unread
pickbrain --channel general --since 7d
pickbrain --dm --since 24h
```

## Notes

- First run requires a full ingest+embed pass (~7s). Subsequent searches auto-ingest incrementally.
- The database lives at `~/.pickbrain/pickbrain.db`.
- Results are ranked by semantic similarity — they may not contain the exact query words.
- The active session's JSONL is skipped during ingest if it was indexed less than 10 minutes ago. If the active session can't be detected (e.g. on Windows), everything is ingested eagerly.
