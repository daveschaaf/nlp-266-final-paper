"""SQLite work queue and durable results for prepared-remark labeling."""

import hashlib
import json
from pathlib import Path
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS label_run (
    run_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    planned_count INTEGER NOT NULL CHECK (planned_count > 0),
    engine_version TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS remark_label (
    run_id TEXT NOT NULL REFERENCES label_run(run_id),
    ordinal INTEGER NOT NULL,
    remark_chunk_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    call_id TEXT NOT NULL,
    remark_text TEXT NOT NULL,
    remark_hash TEXT NOT NULL,
    prior_chunk_id TEXT NOT NULL,
    prior_text TEXT NOT NULL,
    prior_hash TEXT NOT NULL,
    similarity REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    label TEXT CHECK (label IN ('boilerplate', 'administrative', 'repeated_substantive', 'novel_substantive', 'general_business_context', 'yes', 'no')),
    scores_json TEXT,
    inference_seconds REAL,
    peak_mlx_gb REAL,
    last_error TEXT,
    PRIMARY KEY (run_id, remark_chunk_id),
    UNIQUE (run_id, ordinal)
);
CREATE INDEX IF NOT EXISTS remark_label_queue ON remark_label(run_id, status, ordinal);
"""


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def connect(path: Path, *, create=False):
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    mode = "rwc" if create else "rw"
    connection = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if create:
        connection.executescript(SCHEMA)
    return connection


def create_run(connection, run_id, config, rows):
    with connection:
        connection.execute(
            "INSERT INTO label_run(run_id, config_json, planned_count) VALUES (?, ?, ?)",
            (run_id, json.dumps(config, sort_keys=True), len(rows)),
        )
        connection.executemany(
            """INSERT INTO remark_label (
                run_id, ordinal, remark_chunk_id, ticker, call_id, remark_text,
                remark_hash, prior_chunk_id, prior_text, prior_hash, similarity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                (run_id, index, remark.sentence_id, remark.ticker, remark.call_id,
                 remark.text, fingerprint(remark.text), prior.sentence_id, prior.text,
                 fingerprint(prior.text), score)
                for index, (remark, prior, score) in enumerate(rows)
            ),
        )


def get_run(connection, run_id):
    row = connection.execute("SELECT * FROM label_run WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise LookupError(f"No labeling run named {run_id!r}.")
    return row


def clone_completed_run(connection, source_id, run_id, config):
    """Copy completed inputs into a fresh queue, without copying outcomes."""
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        source = get_run(connection, source_id)
        count = connection.execute(
            "SELECT COUNT(*) FROM remark_label WHERE run_id = ? AND status = 'completed'",
            (source_id,),
        ).fetchone()[0]
        if not count:
            raise ValueError(f"No completed inputs in {source_id!r}.")
        connection.execute(
            "INSERT INTO label_run(run_id, config_json, planned_count) VALUES (?, ?, ?)",
            (run_id, json.dumps(config | {"source_run_id": source["run_id"]}, sort_keys=True), count),
        )
        connection.execute(
            """INSERT INTO remark_label (
                run_id, ordinal, remark_chunk_id, ticker, call_id, remark_text,
                remark_hash, prior_chunk_id, prior_text, prior_hash, similarity
            )
            SELECT ?, ordinal, remark_chunk_id, ticker, call_id, remark_text,
                   remark_hash, prior_chunk_id, prior_text, prior_hash, similarity
            FROM remark_label WHERE run_id = ? AND status = 'completed'""",
            (run_id, source_id),
        )


def bind_engine(connection, run_id, engine_version):
    run = get_run(connection, run_id)
    if run["engine_version"] not in (None, engine_version):
        raise ValueError("JEV engine version changed. Use the original engine or plan a new run.")
    with connection:
        connection.execute(
            "UPDATE label_run SET engine_version = ? WHERE run_id = ?", (engine_version, run_id)
        )


def pending(connection, run_id, limit=None, *, retry_failed=False):
    statuses = ("pending", "failed") if retry_failed else ("pending",)
    placeholders = ",".join("?" for _ in statuses)
    return connection.execute(
        f"""SELECT * FROM remark_label WHERE run_id = ? AND status IN ({placeholders})
            ORDER BY ordinal LIMIT ?""",
        (run_id, *statuses, -1 if limit is None else limit),
    ).fetchall()


def save_outcome(connection, row, result=None, error=None):
    label, scores, seconds, peak = result if result is not None else (None, None, None, None)
    with connection:
        updated = connection.execute(
            """UPDATE remark_label SET status = ?, attempts = attempts + 1, label = ?,
               scores_json = ?, inference_seconds = ?, peak_mlx_gb = ?, last_error = ?
               WHERE run_id = ? AND remark_chunk_id = ? AND status IN ('pending', 'failed')""",
            ("failed" if error is not None else "completed", label,
             json.dumps(scores) if scores is not None else None, seconds, peak, error,
             row["run_id"], row["remark_chunk_id"]),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Label row changed while it was being processed.")


def status(connection, run_id):
    run = get_run(connection, run_id)
    counts = {"pending": 0, "completed": 0, "failed": 0}
    counts.update(dict(connection.execute(
        "SELECT status, COUNT(*) FROM remark_label WHERE run_id = ? GROUP BY status", (run_id,)
    )))
    labels = dict(connection.execute(
        """SELECT label, COUNT(*) FROM remark_label
           WHERE run_id = ? AND status = 'completed' GROUP BY label""", (run_id,)
    ))
    timing = connection.execute(
        """SELECT COALESCE(SUM(inference_seconds), 0), MAX(peak_mlx_gb)
           FROM remark_label WHERE run_id = ? AND status = 'completed'""", (run_id,)
    ).fetchone()
    return {
        "run_id": run_id, "planned": run["planned_count"], **counts, "labels": labels,
        "inference_seconds": round(timing[0], 2), "peak_mlx_gb": timing[1],
    }
