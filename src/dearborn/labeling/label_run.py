"""Plan, run, and inspect resumable JEV prepared-remark labeling."""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import fcntl
import json
from pathlib import Path
import random
import signal
import sqlite3

from dearborn.constants import DATA_ROOT
from dearborn.finrank_policy import fetch_finrank_ciks, is_finrank_ticker

from . import repository
from .jev_backend import JevLabeler, MODEL_ID, PROMPT_VERSION
from .qwen_backend import QwenLabeler, MODEL_ID as QWEN_ID, PROMPT_VERSION as QWEN_PROMPT_VERSION

SELECTION_VERSION = "immediate-prior-tfidf-v1"


def plan_inputs(corpus, limit, issuer_count, seed, *, exclude_call_ids=()):
    # The inference worker runs in an isolated JEV environment.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from .similarity import load_call_sentences

    blocked = {int(cik) for cik in fetch_finrank_ciks(corpus)}
    calls = load_call_sentences(corpus)
    index = {(call.cik, call.period_index): call for call in calls}
    grouped = {}
    for call in calls:
        if call.call_id in exclude_call_ids:
            continue
        prior = index.get((call.cik, call.period_index - 1))
        if prior and int(call.cik) not in blocked and not is_finrank_ticker(call.ticker):
            if call.ticker == "EW" and (call.year, call.quarter) == (2025, 1):
                continue
            grouped.setdefault(call.cik, []).append((call, prior))

    rng = random.Random(seed)
    if limit is None:
        selection = [(call, prior, call.sentences) for pairs in grouped.values() for call, prior in pairs]
    else:
        base, remainder = divmod(limit, issuer_count)
        eligible = [
            cik for cik, pairs in grouped.items()
            if any(len(call.sentences) >= base + bool(remainder) for call, _ in pairs)
        ]
        if len(eligible) < issuer_count:
            raise ValueError("Too few issuers with enough eligible remarks for this sample.")
        selection = []
        for i, cik in enumerate(rng.sample(sorted(eligible), issuer_count)):
            count = base + (i < remainder)
            call, prior = next(
                (call, prior) for call, prior in reversed(grouped[cik])
                if len(call.sentences) >= count
            )
            positions = sorted(rng.sample(range(len(call.sentences)), count))
            selection.append((call, prior, tuple(call.sentences[pos] for pos in positions)))

    results = []
    for call, prior, targets in selection:
        # Fit on both complete calls so a smaller sample has the same prior matches.
        matrix = TfidfVectorizer(lowercase=True, sublinear_tf=True).fit_transform(
            sentence.text for sentence in (*call.sentences, *prior.sentences)
        )
        positions = {sentence.sentence_id: i for i, sentence in enumerate(call.sentences)}
        scores = cosine_similarity(
            matrix[[positions[target.sentence_id] for target in targets]],
            matrix[len(call.sentences):],
        )
        for target, row in zip(targets, scores, strict=True):
            best = int(row.argmax())
            results.append((target, prior.sentences[best], float(row[best])))
    if not results:
        raise ValueError("No remarks with usable prior-quarter context.")
    return results


@contextmanager
def worker_lock(path: Path):
    lock_path = path.with_name(path.name + ".worker.lock")
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A labeling worker already owns this database.") from exc
        yield


def run_queue(connection, rows, backend, should_stop, notify=print):
    for row in rows:
        if should_stop():
            break
        try:
            result = backend(row)
        except Exception as exc:
            repository.save_outcome(connection, row, error=f"{type(exc).__name__}: {exc}")
            notify(f"FAILED {row['remark_chunk_id']}: {exc}")
            break
        repository.save_outcome(connection, row, result=result)
        notify(f"Saved {row['remark_chunk_id']} {result[0]} ({result[2]:.2f}s)")


def positive_int(text):
    number = int(text)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DATA_ROOT / "remark_labels.sqlite")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze a labeling queue without loading JEV.")
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--snapshot", required=True, type=Path)
    plan.add_argument("--corpus", type=Path, default=DATA_ROOT / "corpus.sqlite")
    mode = plan.add_mutually_exclusive_group()
    mode.add_argument("--limit", type=positive_int, default=100)
    mode.add_argument("--all", action="store_true")
    plan.add_argument("--issuer-count", type=positive_int, default=5)
    plan.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("plan-qwen", help="Queue Qwen on a prior run's completed inputs.")
    compare.add_argument("--source-run-id", required=True)
    compare.add_argument("--run-id", required=True)
    run = commands.add_parser("run", help="Process remaining work for a frozen queue.")
    run.add_argument("--run-id", required=True)
    run.add_argument("--max-items", type=positive_int, help="Stop after this many items.")
    run.add_argument("--retry-failed", action="store_true")
    status = commands.add_parser("status", help="Show durable counts and timing.")
    status.add_argument("--run-id", required=True)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        with closing(repository.connect(args.db, create=args.command == "plan")) as connection:
            if args.command == "plan":
                snapshot = args.snapshot.resolve()
                if not (snapshot / "model.safetensors.index.json").is_file():
                    raise ValueError("Expected the cached JEV snapshot directory.")
                if not args.all and args.limit < args.issuer_count:
                    raise ValueError("The remark limit must be at least the issuer count.")
                limit = None if args.all else args.limit
                with closing(sqlite3.connect(args.corpus.resolve().as_uri() + "?mode=ro", uri=True)) as corpus:
                    corpus.row_factory = sqlite3.Row
                    corpus.execute("BEGIN")
                    rows = plan_inputs(corpus, limit, args.issuer_count, args.seed)
                repository.create_run(connection, args.run_id, {
                    "model_id": MODEL_ID, "snapshot": str(snapshot), "revision": snapshot.name,
                    "prompt_version": PROMPT_VERSION, "selection_version": SELECTION_VERSION,
                    "seed": args.seed, "corpus": str(args.corpus.resolve()),
                    "limit": limit, "issuer_count": None if args.all else args.issuer_count,
                    "finrank_issuers_excluded": True, "locked_ew_call_excluded": True,
                }, rows)
            elif args.command == "plan-qwen":
                repository.clone_completed_run(connection, args.source_run_id, args.run_id, {
                    "model_id": QWEN_ID, "prompt_version": QWEN_PROMPT_VERSION,
                })
            elif args.command == "run":
                with worker_lock(args.db.resolve()):
                    run = repository.get_run(connection, args.run_id)
                    config = json.loads(run["config_json"])
                    expected_prompt = QWEN_PROMPT_VERSION if config["model_id"] == QWEN_ID else PROMPT_VERSION
                    if config["prompt_version"] != expected_prompt:
                        raise ValueError("Prompt version changed. Plan a new run.")
                    rows = repository.pending(
                        connection, args.run_id, args.max_items, retry_failed=args.retry_failed
                    )
                    if rows:
                        run_worker(connection, args.run_id, config, rows)
            result = repository.status(connection, args.run_id)
            print(json.dumps(result, indent=2), flush=True)
            if args.command == "run":
                if result["failed"]:
                    print("STOPPED: failed rows remain. Inspect last_error before using --retry-failed.")
                    return 1
                print("PAUSED: rerun the same command to resume." if result["pending"] else "DONE")
    except (ValueError, LookupError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    return 0


def run_worker(connection, run_id, config, rows):
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        print(f"Loading cached {config['model_id']} for {len(rows)} queued remarks.", flush=True)
        backend = QwenLabeler() if config["model_id"] == QWEN_ID else JevLabeler(Path(config["snapshot"]))
        repository.bind_engine(connection, run_id, backend.engine_version)
        run_queue(connection, rows, backend, lambda: stopping, lambda message: print(message, flush=True))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
