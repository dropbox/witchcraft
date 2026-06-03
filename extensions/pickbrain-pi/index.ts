import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { accessSync, constants } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const MAX_OUTPUT_BYTES = 50 * 1024;

type RunResult = {
	stdout: string;
	stderr: string;
	code: number | null;
	truncated: boolean;
};

function truncateOutput(text: string): { text: string; truncated: boolean } {
	const bytes = Buffer.byteLength(text, "utf8");
	if (bytes <= MAX_OUTPUT_BYTES) return { text, truncated: false };
	let end = MAX_OUTPUT_BYTES;
	while (end > 0 && (Buffer.from(text.slice(0, end)).at(-1) ?? 0) >= 0x80) end--;
	return {
		text: text.slice(0, end) + `\n\n[Output truncated to ${MAX_OUTPUT_BYTES} bytes]`,
		truncated: true,
	};
}

function pickbrainCommand(): string {
	if (process.env.PICKBRAIN_BIN) return process.env.PICKBRAIN_BIN;
	const homeBin = join(homedir(), "bin", "pickbrain");
	try {
		accessSync(homeBin, constants.X_OK);
		return homeBin;
	} catch {
		return "pickbrain";
	}
}

function runPickbrain(
	args: string[],
	cwd: string,
	env: Record<string, string>,
	signal?: AbortSignal,
): Promise<RunResult> {
	return new Promise((resolve, reject) => {
		const child = spawn(pickbrainCommand(), args, {
			cwd,
			env: { ...process.env, ...env },
		});

		let stdout = "";
		let stderr = "";
		let settled = false;
		const timeout = setTimeout(() => {
			child.kill("SIGTERM");
		}, 120_000);
		const abort = () => child.kill("SIGTERM");
		signal?.addEventListener("abort", abort, { once: true });

		child.stdout.on("data", (chunk) => {
			stdout += chunk.toString();
		});
		child.stderr.on("data", (chunk) => {
			stderr += chunk.toString();
		});
		child.on("error", (error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abort);
			reject(error);
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abort);
			const out = truncateOutput(stdout);
			const err = truncateOutput(stderr);
			resolve({ stdout: out.text, stderr: err.text, code, truncated: out.truncated || err.truncated });
		});
	});
}

function sessionEnv(ctx: any): Record<string, string> {
	const env: Record<string, string> = {};
	const sessionId = ctx.sessionManager?.getSessionId?.();
	const sessionFile = ctx.sessionManager?.getSessionFile?.();
	if (sessionId) env.PICKBRAIN_ACTIVE_SESSION_ID = String(sessionId);
	if (sessionFile) env.PICKBRAIN_ACTIVE_SESSION_FILE = String(sessionFile);
	return env;
}

function renderResult(result: RunResult): string {
	const parts: string[] = [];
	if (result.stderr.trim()) parts.push(result.stderr.trimEnd());
	if (result.stdout.trim()) parts.push(result.stdout.trimEnd());
	if (result.code && result.code !== 0) parts.push(`[pickbrain exited with code ${result.code}]`);
	return parts.join("\n").trim() || "pickbrain returned no output";
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "pickbrain_search",
		label: "Pickbrain",
		description:
			"Semantic search over past Pi, Claude Code, Codex, and Slack conversations. Use for recalling previous coding-agent sessions, dumping sessions, or searching history.",
		promptSnippet: "Search previous Pi/Claude/Codex/Slack conversations with pickbrain semantic search",
		promptGuidelines: [
			"Use pickbrain_search when the user asks to recall, find, or reference a previous Pi, Claude Code, Codex, or Slack conversation.",
		],
		parameters: Type.Object({
			query: Type.Optional(Type.String({ description: "Search query. May be omitted with filters to browse recent matches." })),
			current: Type.Optional(Type.Boolean({ description: "Search only the current Pi session." })),
			excludeCurrent: Type.Optional(Type.Boolean({ description: "Exclude the current Pi session." })),
			session: Type.Optional(Type.String({ description: "Session id, Slack channel, or thr:<timestamp> to search within." })),
			type: Type.Optional(Type.String({ description: "Source filter, e.g. pi, claude, codex, slack, or comma-separated." })),
			since: Type.Optional(Type.String({ description: "Recent-history filter like 24h, 7d, or 2w." })),
			branch: Type.Optional(Type.String({ description: "Git branch filter. Use . for the current branch." })),
			numResults: Type.Optional(Type.Number({ description: "Number of results. 0 means unlimited." })),
			dump: Type.Optional(Type.String({ description: "Dump this session/channel/thread id instead of searching." })),
			turns: Type.Optional(Type.String({ description: "Turn range for dumps, e.g. 2-5." })),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const args: string[] = [];
			if (params.current) args.push("--current");
			if (params.excludeCurrent) args.push("--exclude-current");
			if (params.session) args.push("--session", params.session);
			if (params.type) args.push("--type", params.type);
			if (params.since) args.push("--since", params.since);
			if (params.branch) args.push("--branch", params.branch);
			if (typeof params.numResults === "number") args.push("-n", String(params.numResults));
			if (params.dump) args.push("--dump", params.dump);
			if (params.turns) args.push("--turns", params.turns);
			if (params.query) args.push(params.query);

			const result = await runPickbrain(args, ctx.cwd, sessionEnv(ctx), signal);
			const text = renderResult(result);
			return {
				content: [{ type: "text", text }],
				details: { args, code: result.code, truncated: result.truncated },
			};
		},
	});

	pi.registerCommand("pickbrain", {
		description: "Run pickbrain with the current Pi session in the environment",
		handler: async (args, ctx) => {
			const argv = args.trim() ? args.trim().split(/\s+/) : [];
			const result = await runPickbrain(argv, ctx.cwd, sessionEnv(ctx), ctx.signal);
			pi.sendMessage(
				{
					customType: "pickbrain",
					content: renderResult(result),
					display: true,
					details: { args: argv, code: result.code, truncated: result.truncated },
				},
				{ triggerTurn: false },
			);
		},
	});
}
