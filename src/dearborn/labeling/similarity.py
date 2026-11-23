"""Score prepared remarks against prior-company and peer-call language.

This experiment adapts Brown and Tucker's (2011) year-over-year TF-IDF
similarity measure from complete MD&A documents to sentence-level prepared
remarks.  It compares each sentence in a target call with:

1. every eligible prepared sentence in the same issuer's immediately prior
   available call; and
2. every eligible prepared sentence in one or more different-issuer calls
   from the same year and quarter.

The experiment is read-only with respect to the corpus.  It writes complete
scores, a blinded audit sample, and a compact run summary as JSON/JSONL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dearborn.company_data import sqlite_connection
from dearborn.logging_config import setup_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemarkSentence:
    sentence_id: str
    call_id: str
    cik: str
    ticker: str
    industry: str | None
    year: int
    quarter: int
    speaker: str
    text: str
    word_count: int


@dataclass(frozen=True)
class CallSentences:
    call_id: str
    cik: str
    ticker: str
    industry: str | None
    year: int
    quarter: int
    sentences: tuple[RemarkSentence, ...]

    @property
    def period_index(self) -> int:
        return self.year * 4 + self.quarter


@dataclass(frozen=True)
class CallComparison:
    target: CallSentences
    previous: CallSentences
    peers: tuple[CallSentences, ...]


@dataclass(frozen=True)
class SentenceSimilarityScore:
    sentence_id: str
    call_id: str
    cik: str
    ticker: str
    year: int
    quarter: int
    speaker: str
    text: str
    word_count: int
    previous_call_id: str
    same_company_similarity: float
    same_company_match_id: str
    same_company_match_text: str
    cross_company_similarity: float
    cross_company_match_id: str
    cross_company_match_call_id: str
    cross_company_match_ticker: str
    cross_company_match_text: str
    quadrant: str = ""


def load_call_sentences(
    connection,
    *,
    minimum_words: int = 10,
    eligible_only: bool = True,
    cik: str | None = None,
) -> tuple[CallSentences, ...]:
    """Load sentence-level prepared remarks grouped in chronological calls."""
    eligibility = "AND rc.eligible_for_synthesis = 1" if eligible_only else ""
    issuer_filter = "AND e.cik = ?" if cik is not None else ""
    rows = connection.execute(
        f"""
        SELECT e.call_id, e.cik, e.ticker, c.industry, e.year, e.quarter,
               pr.speaker, pr.segment_index, rc.chunk_index,
               rc.remark_chunk_id AS sentence_id, rc.text, rc.word_count
        FROM earnings_call AS e
        JOIN company AS c ON c.cik = e.cik
        JOIN prepared_remark AS pr ON pr.call_id = e.call_id
        JOIN remark_chunk AS rc ON rc.remark_id = pr.remark_id
        WHERE rc.word_count >= ? {eligibility} {issuer_filter}
        ORDER BY e.cik, e.year, e.quarter, pr.segment_index, rc.chunk_index,
                 rc.remark_chunk_id
        """,
        (minimum_words,) if cik is None else (minimum_words, cik),
    ).fetchall()

    grouped: dict[str, list[RemarkSentence]] = {}
    headers: dict[str, tuple[str, str, str | None, int, int]] = {}
    for row in rows:
        values = dict(row)
        call_id = values["call_id"]
        headers[call_id] = (
            values["cik"],
            values["ticker"],
            values["industry"],
            values["year"],
            values["quarter"],
        )
        grouped.setdefault(call_id, []).append(
            RemarkSentence(
                sentence_id=values["sentence_id"],
                call_id=call_id,
                cik=values["cik"],
                ticker=values["ticker"],
                industry=values["industry"],
                year=values["year"],
                quarter=values["quarter"],
                speaker=values["speaker"],
                text=values["text"],
                word_count=values["word_count"],
            )
        )

    calls = []
    for call_id, sentences in grouped.items():
        cik, ticker, industry, year, quarter = headers[call_id]
        calls.append(
            CallSentences(
                call_id=call_id,
                cik=cik,
                ticker=ticker,
                industry=industry,
                year=year,
                quarter=quarter,
                sentences=tuple(sentences),
            )
        )
    return tuple(sorted(calls, key=lambda call: (call.cik, call.period_index)))


def select_comparisons(
    calls: Sequence[CallSentences],
    *,
    company_count: int,
    peer_call_count: int,
    seed: int,
) -> tuple[CallComparison, ...]:
    """Select one latest adjacent-call pair per issuer and same-period peers."""
    if company_count < 1:
        raise ValueError("company_count must be at least 1")
    if peer_call_count < 1:
        raise ValueError("peer_call_count must be at least 1")

    by_cik: dict[str, list[CallSentences]] = {}
    for call in calls:
        if call.sentences:
            by_cik.setdefault(call.cik, []).append(call)

    candidates: list[tuple[CallSentences, CallSentences]] = []
    for issuer_calls in by_cik.values():
        ordered = sorted(issuer_calls, key=lambda call: call.period_index)
        if len(ordered) >= 2:
            candidates.append((ordered[-1], ordered[-2]))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    comparisons = []
    for target, previous in candidates:
        exact_period = [
            call
            for call in calls
            if call.cik != target.cik
            and call.year == target.year
            and call.quarter == target.quarter
            and call.sentences
        ]
        same_industry = [
            call for call in exact_period if call.industry == target.industry
        ]
        peer_pool = same_industry or exact_period
        if not peer_pool:
            continue
        peer_pool = sorted(peer_pool, key=lambda call: (call.cik, call.call_id))
        peers = tuple(rng.sample(peer_pool, min(peer_call_count, len(peer_pool))))
        comparisons.append(CallComparison(target, previous, peers))
        if len(comparisons) >= company_count:
            break

    if not comparisons:
        raise LookupError("No calls had both an earlier issuer call and a same-period peer.")
    return tuple(comparisons)


def _best_matches(left_matrix, right_matrix) -> tuple[np.ndarray, np.ndarray]:
    similarities = cosine_similarity(left_matrix, right_matrix)
    indexes = similarities.argmax(axis=1)
    scores = similarities[np.arange(similarities.shape[0]), indexes]
    return indexes, scores


def score_comparisons(
    comparisons: Sequence[CallComparison],
    *,
    ngram_max: int = 1,
) -> tuple[SentenceSimilarityScore, ...]:
    """Fit one TF-IDF space and find each target sentence's closest matches."""
    if ngram_max not in (1, 2):
        raise ValueError("ngram_max must be 1 or 2")

    all_sentences: dict[str, RemarkSentence] = {}
    for comparison in comparisons:
        for call in (comparison.target, comparison.previous, *comparison.peers):
            for sentence in call.sentences:
                all_sentences[sentence.sentence_id] = sentence

    ordered_sentences = tuple(all_sentences.values())
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(sentence.text for sentence in ordered_sentences)
    row_by_id = {
        sentence.sentence_id: index
        for index, sentence in enumerate(ordered_sentences)
    }

    results = []
    for comparison in comparisons:
        target = comparison.target.sentences
        previous = comparison.previous.sentences
        peers = tuple(
            sentence for call in comparison.peers for sentence in call.sentences
        )
        target_matrix = matrix[[row_by_id[item.sentence_id] for item in target]]
        previous_matrix = matrix[[row_by_id[item.sentence_id] for item in previous]]
        peer_matrix = matrix[[row_by_id[item.sentence_id] for item in peers]]
        previous_indexes, previous_scores = _best_matches(
            target_matrix, previous_matrix
        )
        peer_indexes, peer_scores = _best_matches(target_matrix, peer_matrix)

        for index, sentence in enumerate(target):
            previous_match = previous[int(previous_indexes[index])]
            peer_match = peers[int(peer_indexes[index])]
            results.append(
                SentenceSimilarityScore(
                    sentence_id=sentence.sentence_id,
                    call_id=sentence.call_id,
                    cik=sentence.cik,
                    ticker=sentence.ticker,
                    year=sentence.year,
                    quarter=sentence.quarter,
                    speaker=sentence.speaker,
                    text=sentence.text,
                    word_count=sentence.word_count,
                    previous_call_id=comparison.previous.call_id,
                    same_company_similarity=float(previous_scores[index]),
                    same_company_match_id=previous_match.sentence_id,
                    same_company_match_text=previous_match.text,
                    cross_company_similarity=float(peer_scores[index]),
                    cross_company_match_id=peer_match.sentence_id,
                    cross_company_match_call_id=peer_match.call_id,
                    cross_company_match_ticker=peer_match.ticker,
                    cross_company_match_text=peer_match.text,
                )
            )
    return tuple(results)


def assign_quadrants(
    scores: Sequence[SentenceSimilarityScore],
) -> tuple[tuple[SentenceSimilarityScore, ...], float, float]:
    """Assign high/low groups using experiment-wide median thresholds."""
    if not scores:
        raise ValueError("Cannot assign quadrants without scores.")
    same_threshold = float(np.median([x.same_company_similarity for x in scores]))
    cross_threshold = float(np.median([x.cross_company_similarity for x in scores]))
    assigned = []
    for score in scores:
        same = "high" if score.same_company_similarity >= same_threshold else "low"
        cross = "high" if score.cross_company_similarity >= cross_threshold else "low"
        assigned.append(
            SentenceSimilarityScore(
                **{**asdict(score), "quadrant": f"same_{same}__cross_{cross}"}
            )
        )
    return tuple(assigned), same_threshold, cross_threshold


def sample_blinded_audit(
    scores: Sequence[SentenceSimilarityScore],
    *,
    per_quadrant: int,
    seed: int,
) -> tuple[dict, ...]:
    """Return a balanced audit without similarity values or score labels."""
    if per_quadrant < 1:
        raise ValueError("per_quadrant must be at least 1")
    grouped: dict[str, list[SentenceSimilarityScore]] = {}
    for score in scores:
        grouped.setdefault(score.quadrant, []).append(score)

    rng = random.Random(seed)
    selected = []
    for quadrant in sorted(grouped):
        records = grouped[quadrant]
        chosen = rng.sample(records, min(per_quadrant, len(records)))
        selected.extend(chosen)
    rng.shuffle(selected)

    return tuple(
        {
            "audit_id": index,
            "sentence_id": score.sentence_id,
            "ticker": score.ticker,
            "year": score.year,
            "quarter": score.quarter,
            "current_sentence": score.text,
            "previous_call_match": score.same_company_match_text,
            "cross_company_match": score.cross_company_match_text,
            "label": None,
            "allowed_labels": [
                "boilerplate",
                "repeated_substantive",
                "novel_substantive",
                "administrative",
            ],
        }
        for index, score in enumerate(selected)
    )


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(
    connection,
    *,
    output_dir: Path,
    company_count: int = 10,
    peer_call_count: int = 1,
    minimum_words: int = 10,
    audit_per_quadrant: int = 25,
    ngram_max: int = 1,
    seed: int = 42,
) -> dict:
    calls = load_call_sentences(connection, minimum_words=minimum_words)
    comparisons = select_comparisons(
        calls,
        company_count=company_count,
        peer_call_count=peer_call_count,
        seed=seed,
    )
    raw_scores = score_comparisons(comparisons, ngram_max=ngram_max)
    scores, same_threshold, cross_threshold = assign_quadrants(raw_scores)
    audit = sample_blinded_audit(
        scores, per_quadrant=audit_per_quadrant, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "sentence_similarity_scores.jsonl"
    audit_path = output_dir / "blinded_audit.jsonl"
    summary_path = output_dir / "summary.json"
    _write_jsonl(score_path, (asdict(score) for score in scores))
    _write_jsonl(audit_path, audit)

    quadrant_counts: dict[str, int] = {}
    for score in scores:
        quadrant_counts[score.quadrant] = quadrant_counts.get(score.quadrant, 0) + 1
    summary = {
        "method": "tfidf_cosine",
        "ngram_range": [1, ngram_max],
        "seed": seed,
        "minimum_words": minimum_words,
        "requested_company_count": company_count,
        "comparison_count": len(comparisons),
        "peer_call_count": peer_call_count,
        "sentence_count": len(scores),
        "audit_count": len(audit),
        "same_company_median_threshold": same_threshold,
        "cross_company_median_threshold": cross_threshold,
        "quadrant_counts": quadrant_counts,
        "target_call_ids": [item.target.call_id for item in comparisons],
        "previous_call_ids": [item.previous.call_id for item in comparisons],
        "peer_call_ids": [
            [peer.call_id for peer in item.peers] for item in comparisons
        ],
        "score_path": str(score_path),
        "audit_path": str(audit_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "DONE comparisons=%d sentences=%d audit=%d output=%s",
        len(comparisons),
        len(scores),
        len(audit),
        output_dir,
    )
    return summary


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the prepared-remark boilerplate-similarity experiment."
    )
    parser.add_argument("--company-count", type=int, default=10)
    parser.add_argument("--peer-call-count", type=int, default=1)
    parser.add_argument("--minimum-words", type=int, default=10)
    parser.add_argument("--audit-per-quadrant", type=int, default=25)
    parser.add_argument("--ngram-max", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/experiments/boilerplate_similarity"),
    )
    return parser


def main() -> int:
    setup_logging()
    args = argument_parser().parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_root / run_name
    connection = sqlite_connection()
    try:
        summary = run_experiment(
            connection,
            output_dir=output_dir,
            company_count=args.company_count,
            peer_call_count=args.peer_call_count,
            minimum_words=args.minimum_words,
            audit_per_quadrant=args.audit_per_quadrant,
            ngram_max=args.ngram_max,
            seed=args.seed,
        )
    finally:
        connection.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
