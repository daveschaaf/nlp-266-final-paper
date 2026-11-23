"""Prepare usefulness judgments for remarks already hand-labeled substantive."""

import argparse
import json
from pathlib import Path

from dearborn.constants import DATA_ROOT
from .audit_app import load_audit, load_label_events

SUBSTANTIVE = {"novel_substantive", "repeated_substantive"}


def collect_human_substantive(sources):
    latest = {}
    for audit_path, ledger_path in sources:
        audit = {int(row["audit_id"]): row for row in load_audit(audit_path)}
        active = {}
        for event in load_label_events(ledger_path):
            audit_id = int(event["audit_id"])
            if event["label"] is None:
                active.pop(audit_id, None)
            else:
                active[audit_id] = event
        for audit_id, event in active.items():
            item = audit[audit_id]
            sid = item["sentence_id"]
            key = event["reviewed_at"]
            if sid not in latest or key > latest[sid][0]:
                latest[sid] = (key, item, event["label"], str(ledger_path))

    return [
        dict(item, content_label=label, source_ledger=path)
        for _, item, label, path in latest.values()
        if label in SUBSTANTIVE
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    original = DATA_ROOT / "experiments/boilerplate_similarity/pilot-10"
    comparison = DATA_ROOT / "experiments/labeling_evaluation"
    old = comparison / "jev-qwen-disagreement-18"
    new = comparison / "five-label-disagreement-10"
    sources = [
        (original / "blinded_audit.jsonl", original / "blinded_audit_labels.jsonl"),
        (old / "blinded_audit.jsonl", old / "blinded_audit_labels.jsonl"),
        (old / "blinded_audit.jsonl", old / "five_label_audit.jsonl"),
        (new / "blinded_audit.jsonl", new / "blinded_audit_labels.jsonl"),
    ]
    rows = sorted(collect_human_substantive(sources), key=lambda row: row["sentence_id"])
    if not rows:
        raise ValueError("No human-labeled substantive remarks found.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "audit.jsonl").open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            output.write(json.dumps(dict(row, audit_id=index), ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} unique hand-labeled substantive remarks at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
