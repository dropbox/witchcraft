#!/usr/bin/env python3
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("WITCHCRAFT_BENCH_URL", "http://127.0.0.1:7860/v1/models/witchcraft:predict")
NFCORPUS_TSV = Path(
    os.environ.get("WITCHCRAFT_BENCH_INPUT", Path.home() / "src/witchcraft/datasets/nfcorpus.tsv")
)
BATCH_SIZE = int(os.environ.get("WITCHCRAFT_BENCH_BATCH_SIZE", "32"))
TIMEOUT_SECONDS = 300

CACHE_BATCH_MAGIC = b"WECB0001"
CACHE_ENTRY_MAGIC = b"WEMB0001"
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_texts():
    with NFCORPUS_TSV.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            row = line.rstrip("\n").split("\t", 1)
            if len(row) != 2:
                raise ValueError(f"malformed nfcorpus row at line {line_number}")
            yield row[1]


def read_u32(view, offset):
    end = offset + U32.size
    if end > len(view):
        raise ValueError("truncated u32 field")
    return U32.unpack_from(view, offset)[0], end


def read_u64(view, offset):
    end = offset + U64.size
    if end > len(view):
        raise ValueError("truncated u64 field")
    return U64.unpack_from(view, offset)[0], end


def model_dim(model):
    if "-d" not in model:
        return 128
    return int(model.rsplit("-d", 1)[1])


def count_tokens(counts):
    return sum(int(count) for count in counts.split(",") if count)


def parse_cache_entry(entry):
    offset = 0
    if (
        len(entry) < len(CACHE_ENTRY_MAGIC)
        or bytes(entry[: len(CACHE_ENTRY_MAGIC)]) != CACHE_ENTRY_MAGIC
    ):
        raise ValueError("bad cache-entry magic")
    offset += len(CACHE_ENTRY_MAGIC)

    model_len, offset = read_u32(entry, offset)
    counts_len, offset = read_u32(entry, offset)
    embedding_count, offset = read_u64(entry, offset)
    embeddings_len, offset = read_u64(entry, offset)

    model_end = offset + model_len
    counts_end = model_end + counts_len
    embeddings_end = counts_end + embeddings_len
    if embeddings_end > len(entry):
        raise ValueError("truncated cache-entry payload")

    model = bytes(entry[offset:model_end]).decode("utf-8")
    counts = bytes(entry[model_end:counts_end]).decode("utf-8")
    if count_tokens(counts) != embedding_count:
        raise ValueError("cache-entry counts do not match embedding count")

    packed = entry[counts_end:embeddings_end]
    if len(packed) < 6:
        raise ValueError("packed embedding payload is truncated")
    packed_rows = U32.unpack_from(packed, 2)[0]
    if packed_rows != embedding_count:
        raise ValueError("packed row count does not match header")

    return embedding_count, embeddings_len, embedding_count * model_dim(model) * 4


def parse_cache_batch(body):
    if body.lstrip().startswith(b"{"):
        payload = json.loads(body)
        raise ValueError(payload.get("error", "expected compressed response, got JSON"))

    view = memoryview(body)
    offset = 0
    if len(view) < len(CACHE_BATCH_MAGIC) or bytes(view[: len(CACHE_BATCH_MAGIC)]) != CACHE_BATCH_MAGIC:
        raise ValueError("bad cache-batch magic")
    offset += len(CACHE_BATCH_MAGIC)

    item_count, offset = read_u32(view, offset)
    vectors = 0
    packed_bytes = 0
    raw_bytes = 0
    for _ in range(item_count):
        entry_len, offset = read_u64(view, offset)
        entry_end = offset + entry_len
        if entry_end > len(view):
            raise ValueError("truncated cache-batch entry")
        entry_vectors, entry_packed_bytes, entry_raw_bytes = parse_cache_entry(view[offset:entry_end])
        vectors += entry_vectors
        packed_bytes += entry_packed_bytes
        raw_bytes += entry_raw_bytes
        offset = entry_end

    if offset != len(view):
        raise ValueError("cache-batch response has trailing bytes")
    return item_count, vectors, len(body), packed_bytes, raw_bytes


def request_batch(texts):
    body = json.dumps({"text": texts, "response_format": "cache_entries"}).encode("utf-8")
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    print(f"url: {URL}", flush=True)
    print(f"input: {NFCORPUS_TSV}", flush=True)
    print(f"batch-size: {BATCH_SIZE}", flush=True)

    start = time.monotonic()
    total_docs = 0
    total_vectors = 0
    total_response_bytes = 0
    total_packed_bytes = 0
    total_raw_bytes = 0
    for batch_number, texts in enumerate(batched(read_texts(), BATCH_SIZE), start=1):
        request_start = time.monotonic()
        body = request_batch(texts)
        docs, vectors, response_bytes, packed_bytes, raw_bytes = parse_cache_batch(body)
        if docs != len(texts):
            raise ValueError(f"expected {len(texts)} cache entries, got {docs}")
        total_docs += docs
        total_vectors += vectors
        total_response_bytes += response_bytes
        total_packed_bytes += packed_bytes
        total_raw_bytes += raw_bytes
        elapsed = time.monotonic() - start
        request_s = time.monotonic() - request_start
        print(
            f"batch {batch_number}: docs={docs} vectors={vectors} "
            f"request_s={request_s:.3f} total_docs={total_docs} "
            f"docs_per_s={total_docs / elapsed:.2f} "
            f"vectors_per_s={total_vectors / elapsed:.2f} "
            f"response_kib={response_bytes / 1024:.1f} "
            f"raw_to_packed={ratio(raw_bytes, packed_bytes):.2f}x",
            flush=True,
        )

    elapsed = time.monotonic() - start
    print(
        "done: "
        f"docs={total_docs} vectors={total_vectors} elapsed_s={elapsed:.3f} "
        f"docs_per_s={total_docs / elapsed:.2f} "
        f"vectors_per_s={total_vectors / elapsed:.2f} "
        f"response_mib={total_response_bytes / 1024 / 1024:.2f} "
        f"packed_embedding_mib={total_packed_bytes / 1024 / 1024:.2f} "
        f"raw_embedding_mib={total_raw_bytes / 1024 / 1024:.2f} "
        f"raw_to_response={ratio(total_raw_bytes, total_response_bytes):.2f}x "
        f"raw_to_packed={ratio(total_raw_bytes, total_packed_bytes):.2f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
