---
name: pickbrain
description: Semantic search across historical AI pair-programming sessions (Gemini, Claude, Codex, Pi, Slack). Use this to recall context, find previous implementations, or resume older discussions.
---

# Pickbrain for Gemini

`pickbrain` provides semantic search and retrieval for your past AI interactions (via `pickbrain --mcp` over stdio). Instead of searching by keywords, it understands the *meaning* of your questions to find relevant conversation turns. Run `pickbrain` once to build `~/.pickbrain/pickbrain.db` and `pickbrain --watch` (or `make pickbrain-install`) to keep the index updated in the background.

## Tools

### 1. `search_ai_history`
Searches your indexed history for relevant context.
- **`query`**: (Required) Non-empty natural language description of what you are looking for.
- **`session_id`**: (Optional) Filter by session ID, Slack channel name/ID, or Slack thread key (`thr:...`).
- **`source`**: (Optional) Filter by `gemini`, `claude`, `codex`, `pi`, or `slack`.
- **`project`**: (Optional) Filter by workspace directory basename (exact match, e.g. `witchcraft`; see `list_projects`).
- **`since`**: (Optional) Duration filter like `"24h"`, `"7d"`, `"2w"`.
- **`limit`**: (Optional) Maximum number of results to return (default: `5`, `0` = unlimited).

**When to use**: Whenever the user asks "What did we talk about regarding X?" or "Find where we implemented Y."

### 2. `read_session`
Retrieves conversation turns from a specific session, Slack channel, or Slack thread (`thr:...`).
- **`session_id`**: (Required) The unique ID of the session or Slack `ConvKey` (`thr:...`) found in search results.
- **`turns`**: (Optional) Turn range like `"7-9"` or single turn `"7"` to conserve context tokens (defaults to the first 20 turns when omitted).
- **`since`**: (Optional) Duration filter like `"24h"`, `"7d"`, `"2w"`.
- **`include_thinking`**: (Optional) Include intermediate thinking blocks (default: `false`).

**When to use**: After finding a relevant turn via `search_ai_history`, use this (passing `turns` around the matched turn) to get the full context of the surrounding discussion.

### 3. `list_recent_sessions`
Shows the 20 most recent indexed AI sessions with a 150-character preview.

**When to use**: When the user wants to see what they were working on recently or wants a quick list of recent topics.

### 4. `list_projects`
Lists all indexed project directory basenames and their session counts.

**When to use**: To get an overview of which projects have searchable history.

## Best Practices
- Always start with `search_ai_history` if the user's request is specific ("find the part where...").
- Use `list_recent_sessions` for vague requests ("what was I doing yesterday?").
- If a search result looks promising but is missing context, always call `read_session` to see the full interaction before summarizing for the user.
