"""Export manuscript counts from the corpus database and the original FinRank JSONL."""
import argparse
import csv
import gzip
import json
import sqlite3
from pathlib import Path


def summary(db, records):
    issuers = sorted({r["ticker"] for r in records})
    def passage_texts(field):
        return {p["text"] for r in records for p in r[field] if p["text"].strip()}

    gold = passage_texts("passages")
    negatives = passage_texts("hard_negatives")
    for name, value in [
        ("Human-authored questions", len(records)),
        ("Quantitative evaluation questions", sum(r["reasoning_type"] == "Quantitative" for r in records)),
        ("Unique candidate passages", len(gold | negatives)),
        ("Unique gold passages", len(gold)),
        ("Unique hard-negative passages", len(negatives)),
        ("Passages in both roles", len(gold & negatives)),
        ("Benchmark issuers", len(issuers)),
    ]:
        yield "FinRank", name, value

    for name, query in [
        ("Earnings-call transcripts", "SELECT COUNT(*) FROM earnings_call"),
        ("Companies with transcripts", "SELECT COUNT(DISTINCT cik) FROM earnings_call"),
        ("SEC filings", "SELECT COUNT(*) FROM filing WHERE parse_complete = 1"),
        ("Companies with parsed filings", "SELECT COUNT(DISTINCT cik) FROM filing WHERE parse_complete = 1"),
        ("Filing passage chunks", "SELECT COUNT(*) FROM filing_chunk JOIN filing USING(accession_number) WHERE parse_complete = 1"),
        ("Prepared-remark segments", "SELECT COUNT(*) FROM prepared_remark"),
        ("Remark chunks", "SELECT COUNT(*) FROM remark_chunk"),
    ]:
        yield "All parsed", name, db.execute(query).fetchone()[0]

    matched = """SELECT e.call_id, e.cik, f.accession_number, f.form
                 FROM earnings_call e JOIN filing f USING(company_quarter_id)
                 WHERE e.cik = f.cik AND f.parse_complete = 1"""
    for exclude in [False, True]:
        placeholders = ",".join("?" for _ in issuers)
        subset = matched + (f" AND e.ticker NOT IN ({placeholders})" if exclude else "")
        population = "Matched, ex-FinRank" if exclude else "Matched corpus"
        metrics = [
            ("Call–filing pairs", "SELECT COUNT(*) FROM pairs"),
            ("Issuers", "SELECT COUNT(DISTINCT cik) FROM pairs"),
        ]
        if not exclude:
            metrics.extend([
                ("10-K filings", "SELECT COUNT(*) FROM pairs WHERE form = '10-K'"),
                ("10-Q filings", "SELECT COUNT(*) FROM pairs WHERE form = '10-Q'"),
                ("Filing passage chunks", "SELECT COUNT(*) FROM pairs JOIN filing_chunk USING(accession_number)"),
            ])
        segments = "FROM pairs JOIN prepared_remark USING(call_id)"
        chunks = segments + " JOIN remark_chunk USING(remark_id)"
        metrics.extend([("Prepared-remark segments", "SELECT COUNT(*) " + segments),
                        ("Remark chunks", "SELECT COUNT(*) " + chunks)])
        rounds = "FROM pairs JOIN qa_round USING(call_id)"
        turns = rounds + " JOIN qa_turn USING(round_id)"
        metrics.extend([
            ("Q&A rounds", "SELECT COUNT(*) " + rounds),
            ("Question turns", "SELECT COUNT(*) " + turns + " WHERE role = 'question'"),
            ("Answer turns", "SELECT COUNT(*) " + turns + " WHERE role = 'answer'"),
        ])
        if exclude:
            metrics.append(("Eligible remark chunks", "SELECT COUNT(*) " + chunks + " WHERE eligible_for_synthesis = 1"))
        for name, query in metrics:
            value = db.execute(f"WITH pairs AS ({subset}) {query}", issuers if exclude else ()).fetchone()[0]
            yield population, name, value


def training_summary(db, training_root, benchmark_issuers):
    linked = {}
    for accession, ticker, form, call in db.execute("""
        SELECT f.accession_number, e.ticker, f.form, e.call_id
        FROM filing f JOIN earnings_call e USING(company_quarter_id)
        WHERE e.cik = f.cik AND f.parse_complete = 1
    """):
        if accession in linked:
            raise ValueError(f"Multiple calls for filing {accession}")
        linked[accession] = ticker, form, call
    for population, run in [
        ("Synthetic questions, train", "hard-negatives-ablation-v1"),
        ("Direct remarks, train", "remark561-metadata-v1"),
    ]:
        with gzip.open(training_root / run / "records.jsonl.gz", "rt") as source:
            rows = [r for r in map(json.loads, source) if r["split"] == "train"]
        filings = {}
        for r in rows:
            accession = r["positive"]["accession_number"]
            issuer, form, call = linked[accession]
            if issuer != r["ticker"] or issuer in benchmark_issuers:
                raise ValueError(f"Unexpected training issuer for {accession}")
            filings[accession] = issuer, form, call
        yield population, "Training pairs", len(rows)
        yield population, "Issuers", len({issuer for issuer, _, _ in filings.values()})
        yield population, "Earnings-call transcripts", len({call for _, _, call in filings.values()})
        yield population, "SEC filings", len(filings)
        for form in ("10-Q", "10-K"):
            yield population, f"{form} filings", sum(f == form for _, f, _ in filings.values())
        yield population, "Distinct positive passages", len({r["positive"]["passage_id"] for r in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--finrank", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/data-section.csv")
    args = parser.parse_args()
    with args.finrank.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        rows = list(summary(db, records))
        rows.extend(training_summary(db, args.database.parent / "experiments/finrank_training",
                                     {r["ticker"] for r in records}))
    labels_path = args.database.parent / "qwen_corpus_labels.sqlite"
    with sqlite3.connect(labels_path.resolve().as_uri() + "?mode=ro", uri=True) as labels:
        yes, no = labels.execute("""
            SELECT COUNT(*) FILTER (WHERE label = 'yes'), COUNT(*) FILTER (WHERE label = 'no')
            FROM remark_label WHERE run_id = 'qwen-corpus-v1' AND status = 'completed'
        """).fetchone()
    rows.extend([("Remark usefulness", "Labeled chunks", yes + no),
                 ("Remark usefulness", "Useful", yes), ("Remark usefulness", "Not useful", no)])
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("Population", "Measure", "Count"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
