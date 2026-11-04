---
name: pickbrain
description: Semantic search across historical AI pair-programming sessions (Gemini, Claude, Codex). Use this to recall context, find previous implementations, or resume older discussions.
---

# Pickbrain for Gemini

`pickbrain` provides semantic search and retrieval for your past AI interactions. Instead of searching by keywords, it understands the *meaning* of your questions to find relevant conversation turns.

## Tools

### 1. `search_ai_history`
Searches your entire history for relevant context.
- **`query`**: (Required) The natural language description of what you are looking for.
- **`source`**: (Optional) Filter by `gemini`, `claude`, or `codex`.
- **`project`**: (Optional) Filter by a specific repository name.

**When to use**: Whenever the user asks "What did we talk about regarding X?" or "Find where we implemented Y."

### 2. `read_session`
Retrieves the full text of a specific conversation session.
- **`session_id`**: (Required) The unique ID of the session (usually found in search results).

**When to use**: After finding a relevant turn via `search_ai_history`, use this to get the full context of the surrounding discussion.

### 3. `list_recent_sessions`
Shows the 20 most recent AI sessions.

**When to use**: When the user wants to see what they were working on recently or wants a quick list of recent topics.

### 4. `list_projects`
Lists all repositories that have been indexed.

**When to use**: To get an overview of which projects have searchable history.

## Best Practices
- Always start with `search_ai_history` if the user's request is specific ("find the part where...").
- Use `list_recent_sessions` for vague requests ("what was I doing yesterday?").
- If a search result looks promising but is missing context, always call `read_session` to see the full interaction before summarizing for the user.
