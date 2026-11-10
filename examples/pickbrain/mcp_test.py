import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_binary_path():
    if "PICKBRAIN_BIN" in os.environ and os.path.exists(os.environ["PICKBRAIN_BIN"]):
        return os.environ["PICKBRAIN_BIN"]
    candidates = [
        str(REPO_ROOT / "pickbrain"),
        str(REPO_ROOT / "target/release/examples/pickbrain"),
        str(REPO_ROOT / "target/aarch64-apple-darwin/release/examples/pickbrain"),
        str(REPO_ROOT / "target/x86_64-apple-darwin/release/examples/pickbrain"),
        str(REPO_ROOT / "target/debug/examples/pickbrain"),
        os.path.expanduser("~/.cargo/bin/pickbrain"),
        os.path.expanduser("~/bin/pickbrain"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find pickbrain binary. Build it first.")


class McpClient:
    """Cross-platform stdio MCP test client with non-blocking timeout reads and stderr draining."""

    def __init__(self, binary: str, env: dict):
        self.proc = subprocess.Popen(
            [binary, "--mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self.stdout_queue: queue.Queue = queue.Queue()
        self.stderr_lines: list[str] = []
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        assert self.proc.stdout is not None
        for raw_line in iter(self.proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                self.stdout_queue.put(line)

    def _read_stderr(self):
        assert self.proc.stderr is not None
        for raw_line in iter(self.proc.stderr.readline, b""):
            self.stderr_lines.append(raw_line.decode("utf-8", errors="replace"))

    def send_raw(self, raw_line: str, timeout: float = 5.0):
        assert self.proc.stdin is not None
        self.proc.stdin.write((raw_line + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        try:
            resp_line = self.stdout_queue.get(timeout=timeout)
            return json.loads(resp_line)
        except queue.Empty:
            return None

    def send_request(self, request: dict, timeout: float = 15.0):
        return self.send_raw(json.dumps(request), timeout=timeout)

    def send_notification(self, notification: dict):
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(notification) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def has_stdout_data(self, timeout: float = 0.2) -> bool:
        try:
            item = self.stdout_queue.get(timeout=timeout)
            # Put it back if unexpected data arrived
            self.stdout_queue.put(item)
            return True
        except queue.Empty:
            return False

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def seed_hermetic_sessions(home_dir: Path):
    """Seeds temporary Gemini sessions across two projects (>10 sessions in noise-proj, 1 in alpha-proj)."""
    brain_root = home_dir / ".gemini" / "antigravity" / "brain"

    # Seed 11 sessions in 'noise-proj' so any naive top-10 retrieval caps out before 'alpha-proj'
    for i in range(11):
        sess_dir = brain_root / f"noise-sess-{i:02d}" / ".system_generated" / "logs"
        sess_dir.mkdir(parents=True, exist_ok=True)
        transcript = sess_dir / "transcript.jsonl"
        long_preview = "P" * 160
        lines = [
            json.dumps(
                {
                    "type": "USER_INPUT",
                    "content": f"Command Working Directory: /workspace/noise-proj\nConfigure OAuth authentication middleware variant {i} {long_preview}",
                    "created_at": f"2026-11-10T10:{i:02d}:00Z",
                }
            ),
            json.dumps(
                {
                    "type": "PLANNER_RESPONSE",
                    "content": f"Updated OAuth authentication middleware variant {i}.",
                    "created_at": f"2026-11-10T10:{i:02d}:05Z",
                }
            ),
        ]
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Seed 1 multi-turn session (25 turns) in 'alpha-proj'
    alpha_dir = brain_root / "alpha-sess-main" / ".system_generated" / "logs"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    alpha_transcript = alpha_dir / "transcript.jsonl"
    alpha_lines = []
    for turn_idx in range(25):
        cwd_prefix = "Command Working Directory: /workspace/alpha-proj\n" if turn_idx == 0 else ""
        alpha_lines.append(
            json.dumps(
                {
                    "type": "USER_INPUT",
                    "content": f"{cwd_prefix}Investigate alpha authentication middleware turn {turn_idx}",
                    "created_at": f"2026-11-09T09:{turn_idx:02d}:00Z",
                }
            )
        )
        alpha_lines.append(
            json.dumps(
                {
                    "type": "PLANNER_RESPONSE",
                    "content": f"Resolved alpha authentication middleware turn {turn_idx}.",
                    "created_at": f"2026-11-09T09:{turn_idx:02d}:05Z",
                }
            )
        )
    alpha_transcript.write_text("\n".join(alpha_lines) + "\n", encoding="utf-8")


def test_empty_pickbrain_dir(binary: str, base_env: dict):
    """Verifies that an uninitialized PICKBRAIN_DIR returns isError=True with an actionable hint."""
    with tempfile.TemporaryDirectory(prefix="pb_empty_") as tmp:
        env = dict(base_env)
        env["HOME"] = tmp
        env["PICKBRAIN_DIR"] = os.path.join(tmp, ".pickbrain")
        client = McpClient(binary, env)
        try:
            resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "list_projects", "arguments": {}},
                }
            )
            assert resp is not None, "Expected response when DB is missing"
            assert "error" not in resp, f"Runtime DB missing failure must use result.isError, got: {resp}"
            assert resp.get("result", {}).get("isError") is True, f"Expected isError=True, got: {resp}"
            text = resp["result"]["content"][0]["text"]
            assert "pickbrain" in text.lower(), f"Expected actionable hint to run pickbrain, got: {text}"
        finally:
            client.close()


def test_mcp_server():
    binary = get_binary_path()
    print(f"Testing pickbrain MCP server with binary: {binary}")

    base_env = dict(os.environ)
    default_assets = str(REPO_ROOT / "assets")
    if "WARP_ASSETS" not in base_env and os.path.exists(default_assets):
        base_env["WARP_ASSETS"] = default_assets

    # 0. Test unindexed / empty PICKBRAIN_DIR behavior
    test_empty_pickbrain_dir(binary, base_env)

    with tempfile.TemporaryDirectory(prefix="pb_hermetic_") as tmp:
        tmp_path = Path(tmp)
        env = dict(base_env)
        env["HOME"] = str(tmp_path)
        env["PICKBRAIN_DIR"] = str(tmp_path / ".pickbrain")
        env.pop("EXTRA_GEMINI_DIRS", None)
        env.pop("EXTRA_CLAUDE_DIRS", None)
        env.pop("EXTRA_CODEX_DIRS", None)
        env.pop("EXTRA_PI_DIRS", None)

        seed_hermetic_sessions(tmp_path)

        # Build the hermetic index via --ingest-only
        ingest_res = subprocess.run(
            [binary, "--ingest-only", "--type", "gemini", "-q"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert ingest_res.returncode == 0, f"Hermetic ingest failed: {ingest_res.stderr}"

        client = McpClient(binary, env)
        try:
            # 1. Test 'initialize' request
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            }
            resp = client.send_request(init_req)
            assert resp is not None, "Expected response for 'initialize'"
            assert resp.get("id") == 1
            assert "result" in resp
            assert resp["result"].get("protocolVersion") == "2024-11-05"
            assert "tools" in resp["result"].get("capabilities", {})
            assert resp["result"].get("serverInfo", {}).get("name") == "pickbrain"

            # 2. Test 'notifications/initialized' (no response expected)
            client.send_notification({"jsonrpc": "2.0", "method": "notifications/initialized"})
            assert not client.has_stdout_data(timeout=0.2), "Expected no response for notification"

            # 3. Test malformed / invalid requests do NOT hang the client (Finding #1)
            bad_json_resp = client.send_raw('{"jsonrpc":"2.0","id":9,"method":', timeout=2.0)
            assert bad_json_resp is not None, "Unparseable JSON must return -32700, not hang"
            assert bad_json_resp.get("error", {}).get("code") == -32700

            no_method_resp = client.send_raw('{"jsonrpc":"2.0","id":10}', timeout=2.0)
            assert no_method_resp is not None, "Missing method must return -32600, not hang"
            assert no_method_resp.get("error", {}).get("code") == -32600
            assert no_method_resp.get("id") == 10

            batch_resp = client.send_raw('[{"jsonrpc":"2.0","id":11,"method":"ping"}]', timeout=2.0)
            assert batch_resp is not None, "Batch array must return response, not hang"
            assert batch_resp.get("error", {}).get("code") == -32600

            null_id_resp = client.send_raw('{"jsonrpc":"2.0","id":null,"method":"ping"}', timeout=2.0)
            assert null_id_resp is not None, "Explicit id:null must return response, not hang"

            neg_id_resp = client.send_raw('{"jsonrpc":"2.0","id":-5,"method":"ping"}', timeout=2.0)
            assert neg_id_resp is not None, "Negative id must return response, not hang"
            assert neg_id_resp.get("id") == -5

            # 4. Test 'ping', 'resources/list', 'prompts/list', unknown method
            resp = client.send_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
            assert resp is not None and resp.get("id") == 2 and resp.get("result") == {}

            resp = client.send_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
            assert resp is not None and resp.get("result") == {"resources": []}

            resp = client.send_request({"jsonrpc": "2.0", "id": 4, "method": "prompts/list"})
            assert resp is not None and resp.get("result") == {"prompts": []}

            resp = client.send_request({"jsonrpc": "2.0", "id": 5, "method": "unknown/nonexistent_method"})
            assert resp is not None and resp.get("error", {}).get("code") == -32601

            # 5. Test 'tools/list'
            resp = client.send_request({"jsonrpc": "2.0", "id": 6, "method": "tools/list"})
            assert resp is not None and "result" in resp
            tool_map = {t["name"]: t for t in resp["result"].get("tools", [])}
            assert {"search_ai_history", "read_session", "list_projects", "list_recent_sessions"} <= set(
                tool_map.keys()
            )
            assert "since" in tool_map["search_ai_history"]["inputSchema"]["properties"], (
                "Expected 'since' parameter in search_ai_history inputSchema"
            )

            # 6. Test argument validation & error codes (Finding #2 & #5)
            unknown_tool_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 20,
                    "method": "tools/call",
                    "params": {"name": "no_such_tool", "arguments": {}},
                }
            )
            assert unknown_tool_resp.get("error", {}).get("code") == -32602

            empty_query_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 21,
                    "method": "tools/call",
                    "params": {"name": "search_ai_history", "arguments": {"query": "   "}},
                }
            )
            assert empty_query_resp.get("error", {}).get("code") == -32602

            bad_limit_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 22,
                    "method": "tools/call",
                    "params": {"name": "search_ai_history", "arguments": {"query": "auth", "limit": "5"}},
                }
            )
            assert bad_limit_resp.get("error", {}).get("code") == -32602

            missing_sess_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 23,
                    "method": "tools/call",
                    "params": {"name": "read_session", "arguments": {"session_id": "nope-missing"}},
                }
            )
            assert "error" not in missing_sess_resp, f"Expected isError=True, got: {missing_sess_resp}"
            assert missing_sess_resp.get("result", {}).get("isError") is True

            # 7. Test 'list_projects' and 'list_recent_sessions' on hermetic DB (Finding #12)
            proj_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 30,
                    "method": "tools/call",
                    "params": {"name": "list_projects", "arguments": {}},
                }
            )
            proj_text = proj_resp["result"]["content"][0]["text"]
            assert "noise-proj (11 sessions)" in proj_text, f"Unexpected projects output: {proj_text}"
            assert "alpha-proj (1 sessions)" in proj_text, f"Unexpected projects output: {proj_text}"

            recent_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "tools/call",
                    "params": {"name": "list_recent_sessions", "arguments": {}},
                }
            )
            recent_text = recent_resp["result"]["content"][0]["text"]
            assert "noise-sess-10" in recent_text
            assert "alpha-sess-main" in recent_text
            # Verify SUBSTR(body, 1, 150) returns full 150 characters before '...'
            previews = [
                block.split("\nPreview: ", 1)[1][:-3]
                for block in recent_text.split("\n\n---\n\n")
                if "\nPreview: " in block
            ]
            assert any(len(p) == 150 for p in previews), (
                f"Expected 150-char preview, got lengths: {[len(p) for p in previews]}"
            )

            # 8. Test 'search_ai_history' with project pushdown, limit=0, and metadata (Findings #3, #4, #10)
            search_proj_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 40,
                    "method": "tools/call",
                    "params": {
                        "name": "search_ai_history",
                        "arguments": {
                            "query": "OAuth authentication middleware",
                            "project": "alpha-proj",
                            "limit": 5,
                        },
                    },
                }
            )
            search_proj_text = search_proj_resp["result"]["content"][0]["text"]
            assert "alpha-sess-main" in search_proj_text, (
                f"Expected project='alpha-proj' filter to push down to SQL and find alpha-sess-main, got: {search_proj_text}"
            )

            search_unlim_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 41,
                    "method": "tools/call",
                    "params": {
                        "name": "search_ai_history",
                        "arguments": {
                            "query": "authentication middleware",
                            "limit": 0,
                        },
                    },
                }
            )
            search_unlim_text = search_unlim_resp["result"]["content"][0]["text"]
            assert "No relevant history found." not in search_unlim_text, (
                "limit: 0 must act as unlimited, not return empty results"
            )
            assert search_unlim_text.count("Session:") > 10, (
                f"Expected >10 results for limit=0, got {search_unlim_text.count('Session:')}"
            )

            # 9. Test 'read_session' default turn window & explicit turns (Finding #9)
            read_default_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 50,
                    "method": "tools/call",
                    "params": {
                        "name": "read_session",
                        "arguments": {"session_id": "alpha-sess-main"},
                    },
                }
            )
            read_default_text = read_default_resp["result"]["content"][0]["text"]
            assert "turns" in read_default_text, (
                f"Expected truncation notice mentioning 'turns' on 25-turn session, got: {read_default_text}"
            )

            read_range_resp = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 51,
                    "method": "tools/call",
                    "params": {
                        "name": "read_session",
                        "arguments": {"session_id": "alpha-sess-main", "turns": "0-1"},
                    },
                }
            )
            read_range_text = read_range_resp["result"]["content"][0]["text"]
            assert "turn 0" in read_range_text and "turn 1" in read_range_text
            assert "turn 2" not in read_range_text

            # 10. Test live DB reload after --nuke + re-ingest while MCP server is running (Finding #7)
            live_dir = tmp_path / ".gemini" / "antigravity" / "brain" / "live-sess-99" / ".system_generated" / "logs"
            live_dir.mkdir(parents=True, exist_ok=True)
            (live_dir / "transcript.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "USER_INPUT",
                                "content": "Command Working Directory: /workspace/live-reloaded-proj\nVerify hot reload of recreated database",
                                "created_at": "2026-11-10T22:00:00Z",
                            }
                        ),
                        json.dumps(
                            {
                                "type": "PLANNER_RESPONSE",
                                "content": "Database recreated and reloaded cleanly.",
                                "created_at": "2026-11-10T22:00:05Z",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            nuke_res = subprocess.run([binary, "--nuke"], env=env, capture_output=True, text=True, timeout=15)
            assert nuke_res.returncode == 0, f"--nuke failed: {nuke_res.stderr}"
            reingest_res = subprocess.run(
                [binary, "--ingest-only", "--type", "gemini", "-q"],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert reingest_res.returncode == 0, f"Re-ingest failed: {reingest_res.stderr}"

            proj_after_reload = client.send_request(
                {
                    "jsonrpc": "2.0",
                    "id": 60,
                    "method": "tools/call",
                    "params": {"name": "list_projects", "arguments": {}},
                }
            )
            proj_reload_text = proj_after_reload["result"]["content"][0]["text"]
            assert "live-reloaded-proj (1 sessions)" in proj_reload_text, (
                f"Expected running MCP server to detect recreated DB and list live-reloaded-proj, got: {proj_reload_text}"
            )

        finally:
            client.close()


def main():
    test_mcp_server()
    print("All MCP server tests PASSED.")


if __name__ == "__main__":
    main()
