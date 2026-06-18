#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def clean_field(value: str | None) -> str:
    return " ".join((value or "").split())


def find_beir_root(path: Path, dataset: str) -> Path:
    candidates = [
        path,
        path / dataset,
        path / dataset.replace("-", ""),
        path / dataset.replace("_", "-"),
    ]
    for candidate in candidates:
        if has_beir_files(candidate):
            return candidate

    for corpus in path.rglob("corpus.jsonl"):
        candidate = corpus.parent
        if has_beir_files(candidate):
            return candidate

    raise FileNotFoundError(f"could not find BEIR files under {path}")


def has_beir_files(path: Path) -> bool:
    return (
        (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    )


def download_and_extract(dataset: str, cache_dir: Path, base_url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{urllib.parse.quote(dataset)}.zip"
    archive = cache_dir / f"{dataset}.zip"

    if not archive.is_file():
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)

    root = cache_dir / dataset
    if not has_beir_files(root):
        print(f"extracting {archive}", file=sys.stderr)
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(cache_dir)

    return find_beir_root(cache_dir, dataset)


def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "query-id":
                continue
            if len(row) < 3:
                raise ValueError(f"invalid qrels row in {qrels_path}: {row!r}")
            query_id, corpus_id, score = row[0], row[1], row[2]
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def write_corpus(source_dir: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (source_dir / "corpus.jsonl").open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for line in source:
            doc = json.loads(line)
            doc_id = doc["_id"]
            title = clean_field(doc.get("title"))
            text = clean_field(doc.get("text"))
            body = f"{title} {text}".strip()
            writer.writerow([doc_id, body])
            count += 1
    return count


def write_queries(source_dir: Path, output_path: Path, qrels: dict[str, dict[str, int]]) -> int:
    queries: dict[str, str] = {}
    with (source_dir / "queries.jsonl").open(encoding="utf-8") as source:
        for line in source:
            query = json.loads(line)
            queries[query["_id"]] = clean_field(query.get("text"))

    missing = [query_id for query_id in qrels if query_id not in queries]
    if missing:
        raise ValueError(f"{len(missing)} qrel queries are missing from queries.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for query_id in qrels:
            writer.writerow([query_id, queries[query_id]])
    return len(qrels)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"))
        output.write("\n")


def convert(source_dir: Path, output_name: str) -> None:
    dataset_path = Path("datasets") / f"{output_name}.tsv"
    testset_dir = Path("testset") / output_name
    qrels = load_qrels(source_dir / "qrels" / "test.tsv")

    doc_count = write_corpus(source_dir, dataset_path)
    query_count = write_queries(source_dir, testset_dir / "questions.test.tsv", qrels)
    write_json(testset_dir / "qrels.test.json", qrels)
    write_json(testset_dir / "collection_map.json", None)

    qrel_count = sum(len(docs) for docs in qrels.values())
    print(
        f"wrote {dataset_path} ({doc_count} docs), "
        f"{testset_dir}/questions.test.tsv ({query_count} queries), "
        f"and {qrel_count} qrels"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BEIR dataset into Witchcraft TSV/testset files."
    )
    parser.add_argument("dataset", help="BEIR dataset name, for example trec-covid")
    parser.add_argument(
        "--output-name",
        default=None,
        help="local dataset/testset name; defaults to the dataset name without '-'",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/beir",
        type=Path,
        help="directory used for downloaded BEIR archives and extracted files",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="existing extracted BEIR dataset directory; skips download",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BEIR_BASE_URL,
        help="base URL containing BEIR dataset zip files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or args.dataset.replace("-", "")
    if args.source_dir:
        source_dir = find_beir_root(args.source_dir, args.dataset)
    else:
        source_dir = download_and_extract(args.dataset, args.cache_dir, args.base_url)
    convert(source_dir, output_name)


if __name__ == "__main__":
    main()
