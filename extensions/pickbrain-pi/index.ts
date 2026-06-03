import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { accessSync, constants } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const MAX_OUTPUT_BYTES = 50 * 1024;
const DEFAULT_RESULTS = 5;

type RunResult = {
	stdout: string;
	stderr: string;
	code: number | null;
	truncated: boolean;
};

type PickbrainParams = {
	query?: string;
	current?: boolean;
	excludeCurrent?: boolean;
	session?: string;
	type?: string;
	since?: string;
	branch?: string;
	numResults?: number;
	dump?: string;
	turns?: string;
	allSources?: boolean;
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
			env: { ...process.env, PICKBRAIN_QUIET: "1", ...env },
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

function addDefaultScope(args: string[], params: PickbrainParams) {
	if (params.dump || params.allSources) return;
	if (params.type) {
		if (params.type !== "all") args.push("--type", params.type);
		return;
	}
	args.push("--type", "pi");
}

function buildArgs(params: PickbrainParams): string[] {
	const args: string[] = ["--quiet"];
	if (params.current) args.push("--current");
	if (params.excludeCurrent) args.push("--exclude-current");
	if (params.session) args.push("--session", params.session);
	if (params.since) args.push("--since", params.since);
	if (params.branch) args.push("--branch", params.branch);
	if (typeof params.numResults === "number") args.push("-n", String(params.numResults));
	else if (!params.dump) args.push("-n", String(DEFAULT_RESULTS));
	if (params.dump) args.push("--dump", params.dump);
	if (params.turns) args.push("--turns", params.turns);
	addDefaultScope(args, params);
	if (params.query) args.push(params.query);
	return args;
}

function parseSlashArgs(raw: string): string[] {
	const text = raw.trim();
	if (!text) return ["--quiet", "--type", "pi", "-n", String(DEFAULT_RESULTS)];
	if (text.startsWith("-")) return ["--quiet", ...text.split(/\s+/)];
	return ["--quiet", "--type", "pi", "-n", String(DEFAULT_RESULTS), text];
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "pickbrain_search",
		label: "Pickbrain",
		description:
			"Semantic search over past Pi, Claude Code, Codex, and Slack conversations. Defaults to Pi sessions; set type or allSources for broader search.",
		promptSnippet: "Search previous Pi coding sessions with pickbrain semantic search",
		promptGuidelines: [
			"Use pickbrain_search when the user asks to recall, find, or reference a previous Pi coding session.",
			"pickbrain_search defaults to Pi sessions. Set type to claude, codex, slack, or a comma-separated list only when the user asks to search outside Pi.",
		],
		parameters: Type.Object({
			query: Type.Optional(Type.String({ description: "Search query. May be omitted with filters to browse recent matches." })),
			current: Type.Optional(Type.Boolean({ description: "Search only the current Pi session." })),
			excludeCurrent: Type.Optional(Type.Boolean({ description: "Exclude the current Pi session." })),
			session: Type.Optional(Type.String({ description: "Session id, Slack channel, or thr:<timestamp> to search within." })),
			type: Type.Optional(Type.String({ description: "Source filter, e.g. pi, claude, codex, slack, all, or comma-separated." })),
			allSources: Type.Optional(Type.Boolean({ description: "Search all sources instead of defaulting to Pi sessions." })),
			since: Type.Optional(Type.String({ description: "Recent-history filter like 24h, 7d, or 2w." })),
			branch: Type.Optional(Type.String({ description: "Git branch filter. Use . for the current branch." })),
			numResults: Type.Optional(Type.Number({ description: `Number of results. Defaults to ${DEFAULT_RESULTS}; 0 means unlimited.` })),
			dump: Type.Optional(Type.String({ description: "Dump this session/channel/thread id instead of searching." })),
			turns: Type.Optional(Type.String({ description: "Turn range for dumps, e.g. 2-5." })),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const args = buildArgs(params as PickbrainParams);
			const result = await runPickbrain(args, ctx.cwd, sessionEnv(ctx), signal);
			const text = renderResult(result);
			return {
				content: [{ type: "text", text }],
				details: { args, code: result.code, truncated: result.truncated },
			};
		},
	});

	pi.registerCommand("pickbrain", {
		description: "Search prior Pi sessions with pickbrain (use flags to override)",
		handler: async (args, ctx) => {
			const argv = parseSlashArgs(args);
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
