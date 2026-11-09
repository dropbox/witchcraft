import json
import os
import select
import subprocess


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


def send_request(proc, request):
    line = json.dumps(request) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()
    resp_line = proc.stdout.readline().decode().strip()
    if not resp_line:
        return None
    return json.loads(resp_line)


def send_notification(proc, notification):
    line = json.dumps(notification) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()


def has_stdout_data(proc, timeout=0.2):
    r, _, _ = select.select([proc.stdout], [], [], timeout)
    return bool(r)


def test_mcp_server():
    binary = get_binary_path()
    print(f"Testing pickbrain MCP server with binary: {binary}")

    env = dict(os.environ)
    default_assets = str(REPO_ROOT / "assets")
    if "WARP_ASSETS" not in env and os.path.exists(default_assets):
        env["WARP_ASSETS"] = default_assets

    proc = subprocess.Popen(
        [binary, "--mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
    )

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
        resp = send_request(proc, init_req)
        assert resp is not None, "Expected response for 'initialize'"
        assert resp.get("id") == 1
        assert "result" in resp
        assert resp["result"].get("protocolVersion") == "2024-11-05"
        assert "tools" in resp["result"].get("capabilities", {})
        assert resp["result"].get("serverInfo", {}).get("name") == "pickbrain"

        # 2. Test 'notifications/initialized' (no response expected)
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        send_notification(proc, notif)
        assert not has_stdout_data(proc, timeout=0.2), "Expected no response for notification"

        # 3. Test 'ping'
        ping_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
        }
        resp = send_request(proc, ping_req)
        assert resp is not None, "Expected response for 'ping'"
        assert resp.get("id") == 2
        assert resp.get("result") == {}
        assert resp.get("error") is None

        # 4. Test 'resources/list'
        res_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
        }
        resp = send_request(proc, res_req)
        assert resp is not None, "Expected response for 'resources/list'"
        assert resp.get("id") == 3
        assert resp.get("result") == {"resources": []}

        # 5. Test 'prompts/list'
        prompts_req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/list",
        }
        resp = send_request(proc, prompts_req)
        assert resp is not None, "Expected response for 'prompts/list'"
        assert resp.get("id") == 4
        assert resp.get("result") == {"prompts": []}

        # 6. Test unknown method with an 'id' returns error -32601
        unknown_req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/nonexistent_method",
        }
        resp = send_request(proc, unknown_req)
        assert resp is not None, "Expected response for unknown method"
        assert resp.get("id") == 5
        assert resp.get("result") is None
        assert "error" in resp
        assert resp["error"].get("code") == -32601
        assert "Method not found: unknown/nonexistent_method" in resp["error"].get("message", "")

        # 7. Test 'tools/list' and schema properties
        tools_req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/list",
        }
        resp = send_request(proc, tools_req)
        assert resp is not None, "Expected response for 'tools/list'"
        assert resp.get("id") == 6
        assert "result" in resp
        tool_map = {t["name"]: t for t in resp["result"].get("tools", [])}

        # Assert search_ai_history schema
        assert "search_ai_history" in tool_map
        search_tool = tool_map["search_ai_history"]
        search_props = search_tool["inputSchema"]["properties"]
        assert "source" in search_props
        expected_sources = {"gemini", "claude", "codex", "pi", "slack"}
        assert set(search_props["source"]["enum"]) == expected_sources
        assert "limit" in search_props
        assert search_props["limit"]["type"] == "integer"

        # Assert read_session schema
        assert "read_session" in tool_map
        read_tool = tool_map["read_session"]
        read_props = read_tool["inputSchema"]["properties"]
        assert "session_id" in read_props
        assert "turns" in read_props
        assert read_props["turns"]["type"] == "string"
        assert "since" in read_props
        assert read_props["since"]["type"] == "string"
        assert "include_thinking" in read_props
        assert read_props["include_thinking"]["type"] == "boolean"

        # 8. Test 'tools/call' with 'list_projects' if assets are available
        assets_dir = os.environ.get("WARP_ASSETS", str(REPO_ROOT / "assets"))
        if os.path.exists(assets_dir):
            call_req = {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "list_projects",
                    "arguments": {},
                },
            }
            resp = send_request(proc, call_req)
            assert resp is not None, "Expected response for 'tools/call'"
            assert resp.get("id") == 7
            assert "result" in resp
            assert "content" in resp["result"]

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    test_mcp_server()
    print("All MCP server tests PASSED.")


if __name__ == "__main__":
    main()
