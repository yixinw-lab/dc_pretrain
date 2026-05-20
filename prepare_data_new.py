"""
Preprocess DCLM into pre-tokenized train/val splits using the GPT-2 tokenizer.

Usage:
    python prepare_data.py
    python prepare_data.py --train_tokens 100_000_000 --val_tokens 10_000_000 --local_dir dclm_data
    python prepare_data.py --train_tokens 800_000_000 --overflow_dataset mlfoundations/dclm-baseline-1.0
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
from array import array
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants

SEQUENCE_LENGTH = 2048
SEQUENCE_SIZE = SEQUENCE_LENGTH + 1  # input + target
BATCH_SIZE = 16  # device batch size, used for chunking
DCLM_TRAIN_DATASET = "konwoo/dclm-164k-docs-train"
DCLM_TRAIN_REVISION = "c4f5716"
DCLM_OVERFLOW_DATASET = "mlfoundations/dclm-baseline-1.0"
DCLM_OVERFLOW_REVISION = "a3b142c"

# Expected SHA-256 hashes. Left empty until the new DCLM/GPT-2 artifacts are generated once.
EXPECTED_HASHES = {}

# -----------------------------------------------------------------------------
# Helpers

def stream_dataset(dataset_name, revision, split="train"):
    """Load a dataset split as a streaming iterator."""
    return load_dataset(
        dataset_name,
        revision=revision,
        split=split,
        streaming=True,
    )


def iter_dclm_baseline_documents(dataset_name, revision):
    """
    Iterate the upstream DCLM baseline in repo file order without requiring the
    Python `zstandard` package.

    The upstream dataset stores documents as many `.jsonl.zst` shard files. We
    download shards on demand through the Hugging Face cache and stream them
    through the installed `zstd -dc` CLI, yielding one JSON object per line.
    """
    repo_files = sorted(
        file
        for file in list_repo_files(dataset_name, repo_type="dataset", revision=revision)
        if file.endswith(".jsonl.zst")
    )
    if not repo_files:
        raise ValueError(f"No .jsonl.zst shards found in {dataset_name}@{revision}")

    print(f"  Found {len(repo_files):,} overflow shard files in {dataset_name}@{revision}")

    for index, repo_file in enumerate(repo_files, start=1):
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=revision,
            filename=repo_file,
        )
        local_path = os.path.realpath(local_path)
        print(f"  Overflow shard {index:,}/{len(repo_files):,}: {repo_file}")

        proc = subprocess.Popen(
            ["zstd", "-dc", local_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        finally:
            stderr_text = ""
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                stderr_text = proc.stderr.read()
                proc.stderr.close()
            return_code = proc.wait()
            if return_code not in (0, -13):
                raise RuntimeError(
                    f"`zstd -dc` failed for {local_path} with exit code {return_code}.\n{stderr_text}"
                )


def canonicalize_text(text):
    """Normalize line endings before hashing for exact-text deduplication."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def document_hash(text):
    """Hash a document by its exact text content."""
    return hashlib.sha1(canonicalize_text(text).encode("utf-8")).hexdigest()


def tokenize_documents(
    dataset_iter,
    encoder,
    total_tokens,
    *,
    record_hashes=None,
    skip_seen_hashes=None,
    desc=None,
):
    """
    Tokenize documents from an iterator until we have total_tokens tokens.

    `record_hashes` tracks every document we touched, including documents that
    were partially truncated at the token boundary. This lets overflow skip
    duplicates while still preserving the exact base-prefix behavior of
    `prepare_data.py`.

    `skip_seen_hashes` is only for overflow deduplication. We intentionally do
    not use it on the base subset, because the hard requirement is that the
    eval prefix and the subset-backed train prefix remain byte-for-byte
    identical to the current `prepare_data.py` behavior.
    """
    eot = encoder._special_tokens['<|endoftext|>']
    tokens = array("H")
    docs_used = 0
    docs_skipped = 0
    pbar = tqdm(total=total_tokens, unit="tok", desc=desc)
    for doc in dataset_iter:
        text = doc["text"]
        text_id = None
        if record_hashes is not None or skip_seen_hashes is not None:
            text_id = document_hash(text)

        if skip_seen_hashes is not None:
            if text_id in skip_seen_hashes:
                docs_skipped += 1
                continue
            skip_seen_hashes.add(text_id)

        if record_hashes is not None:
            record_hashes.add(text_id)

        doc_tokens = [eot] + encoder.encode_ordinary(text)
        remaining = total_tokens - len(tokens)
        if remaining <= 0:
            break
        if len(doc_tokens) > remaining:
            doc_tokens = doc_tokens[:remaining]

        tokens.extend(doc_tokens)
        docs_used += 1
        pbar.update(len(doc_tokens))
        if len(tokens) >= total_tokens:
            break
    pbar.close()
    if len(tokens) < total_tokens:
        print(
            f"  Warning: source stream ended early at {len(tokens):,}/{total_tokens:,} tokens. "
            "Proceeding with the tokens collected so far."
        )
    if skip_seen_hashes is not None:
        print(f"  Used {docs_used:,} documents, skipped {docs_skipped:,} exact-text duplicates")
    else:
        print(f"  Used {docs_used:,} documents")
    return tokens


def create_sequences(tokens, sequence_size):
    """Split a flat token list into fixed-size sequences, discarding any remainder."""
    if isinstance(tokens, array):
        tokens = np.frombuffer(tokens, dtype=np.uint16)
    else:
        tokens = np.asarray(tokens, dtype=np.uint16)
    num_sequences = len(tokens) // sequence_size
    tokens = tokens[:num_sequences * sequence_size]
    sequences = tokens.reshape(num_sequences, sequence_size)
    return sequences


def write_datafile(filename, sequences, batch_size):
    """
    Write sequences to a chunked .pt file with padding metadata.

    Format:
    {
        'chunks': List[Tensor],       # each chunk is batch_size * sequence_size tokens
        'valid_counts': List[int],     # real (non-padding) sequences per chunk
        'batch_size': int,
        'sequence_size': int,
    }
    """
    if len(sequences) == 0:
        print(f"Warning: no sequences to write to {filename}")
        return

    sequence_size = sequences.shape[1]
    num_sequences = len(sequences)
    num_full_batches = num_sequences // batch_size
    leftover = num_sequences % batch_size

    chunks = []
    valid_counts = []

    # Full batches
    for i in range(num_full_batches):
        start = i * batch_size
        chunk = sequences[start:start + batch_size].reshape(-1)
        chunks.append(chunk)
        valid_counts.append(batch_size)

    # Leftover with zero-padding
    if leftover > 0:
        leftover_data = sequences[num_full_batches * batch_size:]
        padding = np.zeros((batch_size - leftover, sequence_size), dtype=np.uint16)
        padded = np.concatenate([leftover_data, padding], axis=0).reshape(-1)
        chunks.append(padded)
        valid_counts.append(leftover)

    print(f"Writing {len(chunks):,} chunks to {filename}")
    print(f"  {num_sequences:,} sequences ({num_full_batches} full batches of {batch_size})")
    if leftover > 0:
        print(f"  Last chunk: {leftover}/{batch_size} valid, {batch_size - leftover} padded")

    data = {
        'chunks': [torch.from_numpy(chunk.copy()) for chunk in chunks],
        'valid_counts': valid_counts,
        'batch_size': batch_size,
        'sequence_size': sequence_size,
    }
    torch.save(data, filename)


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(filepath):
    """Check file hash against expected value. Print actual hash if mismatch or unset."""
    basename = os.path.basename(filepath)
    actual = sha256_file(filepath)
    expected = EXPECTED_HASHES.get(basename)
    if expected is None:
        print(f"  Hash for {basename}: {actual}")
        print(f"  (no expected hash set — paste this value into EXPECTED_HASHES to lock it in)")
    elif actual == expected:
        print(f"  Hash OK for {basename}: {actual}")
    else:
        print(f"  HASH MISMATCH for {basename}!")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")


# -----------------------------------------------------------------------------
# Main

def preprocess(
    train_tokens,
    val_tokens,
    local_dir,
    train_dataset=DCLM_TRAIN_DATASET,
    train_revision=DCLM_TRAIN_REVISION,
    overflow_dataset=DCLM_OVERFLOW_DATASET,
    overflow_revision=DCLM_OVERFLOW_REVISION,
    no_overflow=False,
):
    encoder = tiktoken.get_encoding("gpt2")

    val_seqs = val_tokens // SEQUENCE_SIZE
    train_seqs = train_tokens // SEQUENCE_SIZE

    print(f"{'='*60}")
    print(f"Preprocessing DCLM with GPT-2 tokenizer")
    print(f"{'='*60}")
    print(f"Sequence length: {SEQUENCE_LENGTH} (size {SEQUENCE_SIZE})")
    print(f"Tokenizer: gpt2")
    print(f"Base train dataset: {train_dataset}@{train_revision}")
    if no_overflow:
        print("Overflow dataset: disabled")
    else:
        print(f"Overflow dataset: {overflow_dataset}@{overflow_revision}")
    print(f"Eval slice:  first {val_tokens:>13,} raw tokens -> {val_seqs:,} sequences")
    print(f"Train slice: next  {train_tokens:>13,} raw tokens -> {train_seqs:,} sequences")
    print(f"Output: {local_dir}/")
    print(f"{'='*60}")

    os.makedirs(local_dir, exist_ok=True)

    seen_hashes = set()
    base_stream = stream_dataset(train_dataset, train_revision)
    dataset_iter = iter(base_stream)

    print(f"\nTokenizing val ({val_tokens:,} tokens)...")
    val_raw = tokenize_documents(
        dataset_iter,
        encoder,
        val_tokens,
        record_hashes=seen_hashes,
        desc="val",
    )
    val_sequences = create_sequences(val_raw, SEQUENCE_SIZE)
    np.random.seed(42)
    np.random.shuffle(val_sequences)
    print(f"  {len(val_sequences):,} val sequences ({len(val_sequences) * SEQUENCE_LENGTH:,} trainable tokens)")

    print(f"\nTokenizing train ({train_tokens:,} tokens)...")
    train_raw = tokenize_documents(
        dataset_iter,
        encoder,
        train_tokens,
        record_hashes=seen_hashes,
        desc="train/base",
    )
    if len(train_raw) < train_tokens and not no_overflow:
        missing_tokens = train_tokens - len(train_raw)
        print(
            f"\nBase subset exhausted after {len(train_raw):,} train tokens. "
            f"Extending with unseen documents from {overflow_dataset}@{overflow_revision}..."
        )
        if overflow_dataset == DCLM_OVERFLOW_DATASET and overflow_revision == DCLM_OVERFLOW_REVISION:
            overflow_stream = iter_dclm_baseline_documents(overflow_dataset, overflow_revision)
        else:
            overflow_stream = stream_dataset(overflow_dataset, overflow_revision)
        overflow_raw = tokenize_documents(
            iter(overflow_stream),
            encoder,
            missing_tokens,
            skip_seen_hashes=seen_hashes,
            desc="train/overflow",
        )
        train_raw.extend(overflow_raw)
        print(f"  Extended train split to {len(train_raw):,}/{train_tokens:,} tokens")

    train_sequences = create_sequences(train_raw, SEQUENCE_SIZE)
    np.random.seed(43)
    np.random.shuffle(train_sequences)
    print(f"  {len(train_sequences):,} train sequences ({len(train_sequences) * SEQUENCE_LENGTH:,} trainable tokens)")

    # Write
    print()
    val_path = os.path.join(local_dir, "dclm_val.pt")
    train_path = os.path.join(local_dir, "dclm_train.pt")
    write_datafile(val_path, val_sequences, BATCH_SIZE)
    write_datafile(train_path, train_sequences, BATCH_SIZE)

    # Verify hashes
    print()
    verify_hash(val_path)
    verify_hash(train_path)

    print(f"\nDone! Files saved to {local_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DCLM with GPT-2 tokenizer")
    parser.add_argument("--train_tokens", type=int, default=100_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--local_dir", type=str, default="dclm_data")
    parser.add_argument("--train_dataset", type=str, default=DCLM_TRAIN_DATASET)
    parser.add_argument("--train_revision", type=str, default=DCLM_TRAIN_REVISION)
    parser.add_argument("--overflow_dataset", type=str, default=DCLM_OVERFLOW_DATASET)
    parser.add_argument("--overflow_revision", type=str, default=DCLM_OVERFLOW_REVISION)
    parser.add_argument("--no_overflow", action="store_true")
    args = parser.parse_args()

    preprocess(
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        local_dir=args.local_dir,
        train_dataset=args.train_dataset,
        train_revision=args.train_revision,
        overflow_dataset=args.overflow_dataset,
        overflow_revision=args.overflow_revision,
        no_overflow=args.no_overflow,
    )
    # The datasets streaming stack in this environment can crash during interpreter
    # finalization after successful writes, so exit immediately once preprocessing finishes.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
