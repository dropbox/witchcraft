#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "testset/nfcorpus/questions.test.tsv"
COLLECTION_MAP = ROOT / "testset/nfcorpus/collection_map.json"
QRELS = ROOT / "testset/nfcorpus/qrels.test.json"
SCORE = ROOT / "score.py"
DEFAULT_OUT_DIR = ROOT / "bench-results/nfcorpus-bucket-io"


@dataclass
class RunResult:
    repeat: int
    p95_embed_ms: int
    p95_total_ms: int
    ndcg10: float
    output_file: Path


def run_command(cmd, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def ensure_inputs():
    missing = [
        path
        for path in (QUESTIONS, COLLECTION_MAP, QRELS, SCORE)
        if not path.exists()
    ]
    missing += [
        path
        for path in (ROOT / "mydb.sqlite", ROOT / "mydb.sqlite.index")
        if not path.exists()
    ]
    if missing:
        for path in missing:
            print(f"missing {path.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit("run `make nfcorpus` before this benchmark")


def build_warp_cli(skip_build):
    if skip_build:
        return
    run_command(["make", "warp-cli", "EXTRA_FEATURES=deterministic"])


def install_pytrec_eval(skip_install):
    if skip_install:
        return
    run_command(["uv", "pip", "install", "pytrec-eval"])


def cli_path():
    linked = ROOT / "warp-cli"
    if linked.exists():
        return linked
    target = ROOT / "target/aarch64-apple-darwin/release/warp-cli"
    if target.exists():
        return target
    target = ROOT / "target/release/warp-cli"
    if target.exists():
        return target
    raise SystemExit("missing warp-cli; rerun without --no-build")


def parse_latency(output):
    embed = re.search(r"p95 embedder latency = (\d+) ms", output)
    total = re.search(r"p95 total search latency = (\d+) ms", output)
    if not embed or not total:
        print(output, end="")
        raise SystemExit("querycsv output did not contain p95 latency lines")
    return int(embed.group(1)), int(total.group(1))


def score_output(output_file, python):
    output = run_command(
        [
            python,
            str(SCORE),
            str(output_file),
            str(COLLECTION_MAP),
            str(QRELS),
        ]
    )
    return float(output.strip().splitlines()[-1])


def run_one(repeat, out_dir, python):
    output_file = out_dir / f"default-run{repeat}.txt"
    output = run_command(
        [
            str(cli_path()),
            "querycsv",
            str(QUESTIONS),
            str(output_file),
        ]
    )
    p95_embed_ms, p95_total_ms = parse_latency(output)
    ndcg10 = score_output(output_file, python)
    return RunResult(
        repeat=repeat,
        p95_embed_ms=p95_embed_ms,
        p95_total_ms=p95_total_ms,
        ndcg10=ndcg10,
        output_file=output_file,
    )


def print_rows(results):
    print()
    print("| run | p95 embed ms | p95 total ms | ndcg@10 | file |")
    print("|---:|---:|---:|---:|---|")
    for result in results:
        rel = result.output_file.relative_to(ROOT)
        print(
            f"| {result.repeat} | {result.p95_embed_ms} | "
            f"{result.p95_total_ms} | {result.ndcg10:.12f} | {rel} |"
        )


def print_summary(results):
    print()
    print("| runs | avg p95 embed ms | avg p95 total ms | best p95 total ms | avg ndcg@10 |")
    print("|---:|---:|---:|---:|---:|")
    print(
        f"| {len(results)} | "
        f"{mean(result.p95_embed_ms for result in results):.1f} | "
        f"{mean(result.p95_total_ms for result in results):.1f} | "
        f"{min(result.p95_total_ms for result in results)} | "
        f"{mean(result.ndcg10 for result in results):.12f} |"
    )


def write_csv(results, out_dir):
    path = out_dir / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "run",
                "p95_embed_ms",
                "p95_total_ms",
                "ndcg10",
                "output_file",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    "default",
                    result.repeat,
                    result.p95_embed_ms,
                    result.p95_total_ms,
                    f"{result.ndcg10:.12f}",
                    result.output_file.relative_to(ROOT),
                ]
            )
    print()
    print(f"wrote {path.relative_to(ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nfcorpus-score-style querycsv/score loops."
    )
    parser.add_argument("--runs", type=int, default=3, help="runs; default: 3")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-build", action="store_true", help="skip make warp-cli")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="skip uv pip install pytrec-eval",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", "python"),
        help="python executable for score.py; default: python",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    ensure_inputs()
    build_warp_cli(args.no_build)
    install_pytrec_eval(args.skip_install)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for repeat in range(1, args.runs + 1):
        print(f"running default run {repeat}/{args.runs}...")
        results.append(run_one(repeat, out_dir, args.python))

    print_rows(results)
    print_summary(results)
    write_csv(results, out_dir)


if __name__ == "__main__":
    main()
