"""Resumable Qwen usefulness labeling for the eligible non-FinRank corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import closing
import json
from pathlib import Path
import resource
import random
import signal
import sqlite3
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace

from dearborn.constants import DATA_ROOT
from dearborn.finrank_policy import FINRANK_TICKERS, fetch_finrank_ciks

from . import repository
from .five_label_comparison import ComparisonBackend
from .label_run import worker_lock
from .models import MODEL_IDS
from .usefulness_comparison import PROMPT_PATH, load_prompt

# Source audit confirmed that this call's prepared remarks include Q&A.
EXCLUDED_CALLS = ("1035002:2025:Q1",)
METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS worker_metrics (
    run_id TEXT NOT NULL,
    saved_count INTEGER NOT NULL,
    peak_rss_gb REAL NOT NULL,
    peak_mlx_gb REAL NOT NULL,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def diverse_order(rows, seed):
    """Freeze a randomized issuer-balanced order for useful partial runs."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["cik"])].append(row)
    queues = {}
    for cik in sorted(grouped):
        rng.shuffle(grouped[cik])
        queues[cik] = deque(grouped[cik])
    ordered = []
    while queues:
        issuers = list(queues)
        rng.shuffle(issuers)
        for cik in issuers:
            ordered.append(queues[cik].popleft())
            if not queues[cik]:
                del queues[cik]
    return ordered


def prepare(args):
    from huggingface_hub import snapshot_download

    instruction, examples = load_prompt(PROMPT_PATH)
    snapshot = snapshot_download(
        MODEL_IDS["qwen"], local_files_only=True,
        allow_patterns=["*.json", "*.txt", "*.jinja", "model*.safetensors"],
    )
    with closing(sqlite3.connect(args.corpus.resolve().as_uri() + "?mode=ro", uri=True)) as corpus:
        corpus.row_factory = sqlite3.Row
        corpus.execute("BEGIN")
        blocked = {int(value) for value in fetch_finrank_ciks(corpus)}
        records = corpus.execute(
            """SELECT r.remark_chunk_id, r.text, r.parser_version, e.ticker, e.call_id, e.cik
               FROM remark_chunk r JOIN prepared_remark p USING(remark_id)
               JOIN earnings_call e USING(call_id)
               WHERE r.eligible_for_synthesis = 1
               ORDER BY e.cik, e.year, e.quarter, p.segment_index, r.chunk_index, r.remark_chunk_id"""
        ).fetchall()
    eligible = [
        row for row in records if int(row["cik"]) not in blocked
        and row["ticker"].upper() not in FINRANK_TICKERS and row["call_id"] not in EXCLUDED_CALLS
    ]
    eligible = diverse_order(eligible, args.seed)
    if args.limit:
        eligible = eligible[:args.limit]
    if not eligible:
        raise ValueError("No eligible remarks.")
    config = {
        "task": "usefulness", "version": "qwen-corpus-v1", "model_id": MODEL_IDS["qwen"],
        "qwen_snapshot": snapshot, "model_revision": Path(snapshot).name,
        "instruction": instruction, "examples": examples, "labels": ["yes", "no"],
        "max_new_tokens": 256, "prompt_sha256": repository.fingerprint(PROMPT_PATH.read_text()),
        "corpus": str(args.corpus.resolve()), "excluded_call_ids": EXCLUDED_CALLS,
        "excluded_finrank_tickers": sorted(FINRANK_TICKERS),
        "parser_versions": sorted({row["parser_version"] for row in eligible}),
        "queue_order": "randomized_issuer_round_robin_v1", "seed": args.seed,
    }
    frozen = [
        (SimpleNamespace(sentence_id=row["remark_chunk_id"], ticker=row["ticker"],
                         call_id=row["call_id"], text=row["text"]),
         SimpleNamespace(sentence_id="unused", text=""), 0.0)
        for row in eligible
    ]
    with closing(repository.connect(args.db, create=True)) as connection:
        repository.create_run(connection, args.run_id, config, frozen)
        connection.executescript(METRICS_SCHEMA)
        print(json.dumps(repository.status(connection, args.run_id), indent=2))


def worker(args):
    import mlx.core as mx

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with worker_lock(args.db.resolve()), closing(repository.connect(args.db)) as connection:
        config = json.loads(repository.get_run(connection, args.run_id)["config_json"])
        if config.get("version") != "qwen-corpus-v1":
            raise ValueError("Expected a corpus usefulness run.")
        rows = repository.pending(connection, args.run_id, args.chunk_size)
        if not rows or stopping:
            return 0
        print(f"Loading Qwen for up to {len(rows)} remarks.", flush=True)
        backend = ComparisonBackend("qwen", config)
        repository.bind_engine(connection, args.run_id, backend.engine_version)
        saved = 0
        consecutive_failures = 0
        for row in rows:
            if stopping:
                break
            result = None
            error = None
            for _ in range(2):
                if stopping:
                    break
                try:
                    result = backend(row)
                    break
                except ValueError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
            if result is None:
                if stopping:
                    break
                repository.save_outcome(connection, row, error=error or "Inference failed.")
                consecutive_failures += 1
                print(f"FAILED {row['remark_chunk_id']}: {error}", flush=True)
                if consecutive_failures >= 3:
                    return 1
                continue
            repository.save_outcome(connection, row, result=result)
            consecutive_failures = 0
            saved += 1
            if saved == 1 or saved % 50 == 0 or result[3] >= args.memory_limit_gb:
                # macOS reports ru_maxrss in bytes.
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
                with connection:
                    connection.execute(
                        "INSERT INTO worker_metrics(run_id,saved_count,peak_rss_gb,peak_mlx_gb) VALUES(?,?,?,?)",
                        (args.run_id, saved, rss, result[3]),
                    )
                status = repository.status(connection, args.run_id)
                print(f"completed={status['completed']}/{status['planned']} failed={status['failed']} "
                      f"last={result[2]:.2f}s peak_MLX={result[3]:.2f}GB peak_RSS={rss:.2f}GB", flush=True)
                swap = subprocess.run(
                    ["sysctl", "vm.swapusage"], text=True, capture_output=True, timeout=5
                )
                if swap.returncode == 0:
                    print(swap.stdout.strip(), flush=True)
            mx.clear_cache()
            if result[3] >= args.memory_limit_gb:
                print("Recycling worker after memory threshold.", flush=True)
                break
    return 0


def supervise(args):
    stopping = False
    process = None

    def stop(sig, _frame):
        nonlocal stopping
        stopping = True
        if process is not None and process.poll() is None:
            process.send_signal(sig)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    # Separate lock lets each child own the standard SQLite-worker lock.
    with worker_lock(args.db.with_name(args.db.name + ".supervisor").resolve()):
        with closing(repository.connect(args.db)) as connection:
            repository.get_run(connection, args.run_id)
            if args.retry_failed:
                with worker_lock(args.db.resolve()), connection:
                    connection.execute(
                        "UPDATE remark_label SET status='pending' WHERE run_id=? AND status='failed'",
                        (args.run_id,),
                    )
        while not stopping:
            with closing(repository.connect(args.db)) as connection:
                before = repository.status(connection, args.run_id)
            if not before["pending"]:
                break
            command = [
                sys.executable, "-u", "-m", "dearborn.labeling.corpus_run",
                "--db", str(args.db.resolve()), "run",
                "--run-id", args.run_id, "--worker", "--chunk-size", str(args.chunk_size),
                "--memory-limit-gb", str(args.memory_limit_gb),
            ]
            process = subprocess.Popen(command)
            started = monotonic()
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if monotonic() - started > 3600:
                        process.kill()
                        process.wait()
                        print("STOPPED: worker exceeded one hour. Committed results are preserved.", flush=True)
                        return 1
            if process.returncode != 0 and not stopping:
                print("STOPPED: worker failed. Fix the reported error, then resume.", flush=True)
                return 1
            with closing(repository.connect(args.db)) as connection:
                after = repository.status(connection, args.run_id)
            if not stopping and after["pending"] >= before["pending"]:
                raise RuntimeError("Worker made no progress.")
        with closing(repository.connect(args.db)) as connection:
            result = repository.status(connection, args.run_id)
        print(json.dumps(result, indent=2), flush=True)
        print("PAUSED" if result["pending"] else "DONE WITH FAILURES" if result["failed"] else "DONE", flush=True)
        return 1 if result["failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DATA_ROOT / "qwen_corpus_labels.sqlite")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--corpus", type=Path, default=DATA_ROOT / "corpus.sqlite")
    plan.add_argument("--limit", type=int)
    plan.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--chunk-size", type=int, default=500)
    run.add_argument("--memory-limit-gb", type=float, default=8.0)
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    status = commands.add_parser("status")
    status.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive.")
        prepare(args)
        return 0
    if args.command == "status":
        with closing(repository.connect(args.db)) as connection:
            print(json.dumps(repository.status(connection, args.run_id), indent=2))
        return 0
    if args.chunk_size < 1 or args.memory_limit_gb <= 0:
        parser.error("Chunk size and memory limit must be positive.")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
