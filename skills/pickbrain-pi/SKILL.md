---
name: pickbrain
description: Recall and semantic search over previous Pi coding sessions. Use when the user wants to find, remember, or reference prior Pi work — e.g. "what did we discuss about X", "find the session where we fixed Y", "search my history for Z".
---

# Pickbrain — Recall for Pi

Search previous Pi sessions semantically and use the results to answer the user with citations/excerpts. Pickbrain can also search Claude Code, Codex, and Slack when explicitly requested, but in Pi it should default to Pi history.

## Preferred Pi Usage

If the `pickbrain_search` tool is available, use it first. It automatically:

- defaults to `type: "pi"`
- passes the current Pi session to pickbrain
- suppresses ingest progress noise
- limits results to a small useful set unless asked otherwise

Examples:

- Search Pi history: `{ "query": "auth middleware fix" }`
- Search current Pi session: `{ "query": "install target", "current": true }`
- Exclude current Pi session: `{ "query": "dropbox witchcraft", "excludeCurrent": true }`
- Search all coding agents: `{ "query": "extension api", "type": "pi,claude,codex" }`
- Search everything, including Slack: `{ "query": "launch plan", "allSources": true }`
- Dump a session: `{ "dump": "<session-id>", "turns": "2-4" }`

## Slash Command

The Pi extension also registers:

```text
/pickbrain <query>
```

This searches Pi sessions by default. Pass flags to override:

```text
/pickbrain --type pi,claude,codex auth middleware
/pickbrain --dump <session-id> --turns 2-4
```

## Bash Fallback

If the tool is unavailable, run `pickbrain` via Bash:

```bash
pickbrain --quiet --type pi "$ARGUMENTS"
```

Pickbrain automatically ingests new sessions before each search. First run can take longer while building the local database.

## Interpreting Results

Each result includes:

- timestamp and project directory
- source: usually `pi`, or `claude`, `codex`, `slack` if requested
- session ID and turn number
- relevant matching text

Present results as a concise summary and quote the most relevant excerpts. To dig deeper:

```bash
pickbrain --quiet --dump <session-id> --turns <start>-<end>
```

## Useful CLI Filters

```bash
pickbrain --quiet --type pi "<query>"       # only Pi sessions
pickbrain --quiet --current "<query>"       # current calling session when detectable
pickbrain --quiet --exclude-current "<query>"
pickbrain --quiet --session <session-id> "<query>"
pickbrain --quiet --since 7d "<query>"
pickbrain --quiet -n 20 "<query>"
```

## Notes

- The database lives at `~/.pickbrain/pickbrain.db`.
- Results are ranked by semantic similarity and may not contain exact query words.
- For best Pi integration, keep the Pi extension installed in `~/.pi/agent/extensions/pickbrain/`.
