---
name: pickbrain
description: Semantic search over past Pi, Claude Code, Codex, and Slack conversations and memories. Use when the user wants to recall, find, or reference something from a previous coding-agent session — e.g. "what did we discuss about X", "find that conversation where we fixed Y", "search my history for Z".
---

# Pickbrain — Semantic Search for AI Coding History

Search past Pi, Claude Code, Codex, and Slack conversations, memory files, and authored files using semantic search.

## Preferred Pi Usage

If the `pickbrain_search` tool is available, use it first. It passes the current Pi session to pickbrain so `current` and `excludeCurrent` filters work.

Examples:
- Search all history: `pickbrain_search` with `{ "query": "auth middleware fix" }`
- Search current Pi session: `{ "query": "install target", "current": true }`
- Exclude current Pi session: `{ "query": "dropbox witchcraft", "excludeCurrent": true }`
- Restrict to Pi: `{ "query": "extension api", "type": "pi" }`
- Dump a session: `{ "dump": "<session-id>", "turns": "2-4" }`

## Bash Fallback

If the tool is unavailable, run `pickbrain` via Bash:

```bash
pickbrain "$ARGUMENTS"
```

Pickbrain automatically ingests new Pi/Claude/Codex sessions, Slack conversations, memories, and project config files before each search.

## Interpreting Results

Each result includes:
- Timestamp and project directory (or channel for Slack)
- Source: `pi`, `claude`, `codex`, or `slack`
- Session ID and turn number for coding-agent sessions
- Matching text from the conversation

Present results as a concise summary and quote the most relevant excerpts. To dig deeper:

```bash
pickbrain --dump <session-id> --turns <start>-<end>
```

## Useful Filters

```bash
pickbrain --current "<query>"              # current calling session when detectable
pickbrain --exclude-current "<query>"      # exclude current calling session
pickbrain --session <session-id> "<query>" # one session
pickbrain --type pi "<query>"              # only Pi sessions
pickbrain --type claude,codex,pi "<query>" # coding sessions
pickbrain --since 7d "<query>"             # recent history
pickbrain --branch . "<query>"             # current git branch when available
pickbrain -n 20 "<query>"                  # limit results
```

## Notes

- The database lives at `~/.pickbrain/pickbrain.db`.
- Results are ranked by semantic similarity and may not contain exact query words.
- For best Pi integration, keep the Pi extension installed in `~/.pi/agent/extensions/pickbrain/`.
