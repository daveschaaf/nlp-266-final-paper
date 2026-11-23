"""Paired generation-usefulness evaluation excluding the prompt demonstrations."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import re
import signal
from types import SimpleNamespace

from dearborn.constants import DATA_ROOT, PROJECT_ROOT

from . import repository
from .audit_app import USEFULNESS_LABELS, current_labels, load_audit, load_label_events
from .five_label_comparison import ComparisonBackend
from .label_run import run_queue, worker_lock

PROMPT_PATH = PROJECT_ROOT / "docs/labeling/prompts/GENERATION_USEFULNESS_PROMPT.md"
AUDIT_DIR = DATA_ROOT / "experiments/labeling_evaluation/human-substantive-usefulness"


def load_prompt(path):
    text = path.read_text(encoding="utf-8")
    instruction = text.split("## Instruction\n", 1)[1].split("\n## Fixed examples", 1)[0].strip()
    section = text.split("## Fixed examples", 1)[1].split("\n## Model input", 1)[0]
    examples = []
    for match in re.finditer(
        r'Remark: “(.*?)”\s+Label: `(yes|no)`\s+Reason: (.*?)(?=\n###|\Z)', section, re.S
    ):
        examples.append({
            "current_statement": " ".join(match[1].split()),
            "label": match[2], "reason": " ".join(match[3].split()),
        })
    if len(examples) != 10 or {item["label"] for item in examples} != USEFULNESS_LABELS:
        raise ValueError("Expected ten binary demonstrations in the shared prompt.")
    return instruction, examples


def split_cases(rows, labels, examples):
    demonstration_texts = {item["current_statement"] for item in examples}
    cases, excluded = [], []
    for item in rows:
        audit_id = item["audit_id"]
        if audit_id not in labels:
            continue
        if " ".join(item["current_sentence"].split()) in demonstration_texts:
            excluded.append(item["sentence_id"])
        else:
            cases.append(item)
    if not cases:
        raise ValueError("No labeled non-demonstration cases.")
    return cases, excluded


def prepare(args):
    if args.db.exists():
        raise FileExistsError(f"Choose a fresh database: {args.db}")
    instruction, examples = load_prompt(PROMPT_PATH)
    items = load_audit(AUDIT_DIR / "audit.jsonl")
    labels = current_labels(load_label_events(
        AUDIT_DIR / "audit_usefulness_labels.jsonl", allowed_labels=USEFULNESS_LABELS
    ))
    if len(labels) != len(items):
        raise ValueError("The human usefulness audit is incomplete.")
    cases, excluded = split_cases(items, labels, examples)
    snapshot = args.snapshot.resolve()
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise ValueError("Missing cached JEV checkpoint.")
    config = {
        "task": "usefulness", "version": "usefulness-comparison-v1",
        "instruction": instruction, "labels": ["yes", "no"], "examples": examples,
        "prompt_sha256": repository.fingerprint(PROMPT_PATH.read_text()),
        "jev_snapshot": str(snapshot),
        "excluded_demonstration_ids": excluded,
        "gold": {row["sentence_id"]: labels[row["audit_id"]] for row in cases},
        "evaluation_note": "Non-demonstration replay of previously inspected examples; development evaluation.",
    }
    frozen = []
    for row in cases:
        remark = SimpleNamespace(
            sentence_id=row["sentence_id"], ticker=row.get("ticker", ""),
            call_id=row["sentence_id"].split(":remark:", 1)[0], text=row["current_sentence"],
        )
        prior = SimpleNamespace(sentence_id="unused", text="")
        frozen.append((remark, prior, 0.0))
    with closing(repository.connect(args.db, create=True)) as connection:
        for model in ("jev", "qwen"):
            repository.create_run(connection, model + "-usefulness", config, frozen)
    print(f"Prepared {len(cases)} labeled cases per model; excluded {len(excluded)} demonstrations.")
    print(dict(Counter(config["gold"].values())))


def report(connection, model):
    run_id = model + "-usefulness"
    config = json.loads(repository.get_run(connection, run_id)["config_json"])
    rows = connection.execute(
        "SELECT remark_chunk_id,label FROM remark_label WHERE run_id=? AND status='completed'", (run_id,)
    ).fetchall()
    counts = Counter((config["gold"][row["remark_chunk_id"]], row["label"]) for row in rows)
    result = repository.status(connection, run_id)
    result["correct"] = sum(value for (gold, prediction), value in counts.items() if gold == prediction)
    result["confusion"] = {f"{gold}->{prediction}": value for (gold, prediction), value in counts.items()}
    result["missed_useful"] = counts["yes", "no"]
    result["retained_not_useful"] = counts["no", "yes"]
    print(json.dumps(result, indent=2))


def run(args):
    run_id = args.model + "-usefulness"
    with worker_lock(args.db.resolve()), closing(repository.connect(args.db)) as connection:
        config = json.loads(repository.get_run(connection, run_id)["config_json"])
        rows = repository.pending(connection, run_id, args.max_items, retry_failed=args.retry_failed)
        stopping = False

        def stop(*_):
            nonlocal stopping
            stopping = True

        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if rows:
                print(f"Loading {args.model} for {len(rows)} usefulness decisions.", flush=True)
                backend = ComparisonBackend(args.model, config)
                repository.bind_engine(connection, run_id, backend.engine_version)
                run_queue(connection, rows, backend, lambda: stopping,
                          lambda message: print(message, flush=True))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        report(connection, args.model)
        result = repository.status(connection, run_id)
        print("FAILED" if result["failed"] else "PAUSED" if result["pending"] else "DONE")
        return 1 if result["failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DATA_ROOT / "remark_usefulness_eval.sqlite")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare")
    plan.add_argument("--snapshot", type=Path, required=True)
    worker = commands.add_parser("run")
    worker.add_argument("--model", choices=("jev", "qwen"), required=True)
    worker.add_argument("--max-items", type=int)
    worker.add_argument("--retry-failed", action="store_true")
    commands.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        if args.max_items is not None and args.max_items < 1:
            parser.error("--max-items must be positive.")
        return run(args)
    else:
        with closing(repository.connect(args.db)) as connection:
            for model in ("jev", "qwen"):
                report(connection, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
