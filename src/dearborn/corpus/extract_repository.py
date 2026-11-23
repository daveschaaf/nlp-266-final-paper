"""Read models and scoped queries for the three-view extract pipeline.

This module is the only place embedding, dataset, or collation code should
obtain corpus records.  It reconstructs a model-neutral ``ParsedEarningsCall``
from normalized rows rather than asking downstream code to join tables.
"""

from __future__ import annotations

from dataclasses import dataclass

from .company import CompanyFiling, CompanyQuarter, FilingChunk
from .transcript import EarningsCall
from ..extract.contracts import (
    CallParseIssue,
    ExcludedTranscriptSegment,
    ParsedEarningsCall,
    PreparedRemarkUnit,
    QARound,
    QATurn,
)


@dataclass(frozen=True)
class ExtractScope:
    """One unambiguous company-quarter with all three extract inputs."""

    company_quarter_id: str
    cik: str
    call_id: str
    accession_number: str


@dataclass(frozen=True)
class ExtractSource:
    """The complete persisted source material for one extract example."""

    scope: ExtractScope
    earnings_call: ParsedEarningsCall
    filing: CompanyFiling
    filing_chunks: tuple[FilingChunk, ...]


def _row_as_dict(row) -> dict:
    if row is None:
        raise LookupError("Expected one corpus row, but found none.")
    return dict(row)


def fetch_company_quarter(connection, company_quarter_id: str) -> CompanyQuarter:
    row = connection.execute(
        "SELECT * FROM company_quarter WHERE company_quarter_id = ?",
        (company_quarter_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"No company quarter: {company_quarter_id}")
    return CompanyQuarter(**_row_as_dict(row))


def fetch_company_filing(connection, accession_number: str) -> CompanyFiling:
    row = connection.execute(
        "SELECT * FROM filing WHERE accession_number = ?", (accession_number,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No filing: {accession_number}")
    values = _row_as_dict(row)
    values["parse_complete"] = bool(values["parse_complete"])
    return CompanyFiling(**values)


def fetch_earnings_call(connection, call_id: str) -> EarningsCall:
    row = connection.execute(
        "SELECT * FROM earnings_call WHERE call_id = ?", (call_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"No earnings call: {call_id}")
    return EarningsCall(**_row_as_dict(row))


def fetch_filing_chunks(connection, accession_number: str) -> tuple[FilingChunk, ...]:
    rows = connection.execute(
        """
        SELECT * FROM filing_chunk
        WHERE accession_number = ?
        ORDER BY chunk_index, chunk_id
        """,
        (accession_number,),
    ).fetchall()
    if not rows:
        raise LookupError(f"No parsed filing chunks: {accession_number}")
    return tuple(FilingChunk(**_row_as_dict(row)) for row in rows)


def fetch_parsed_earnings_call(connection, call_id: str) -> ParsedEarningsCall:
    """Reconstruct the persisted extract call, including all attributable turns."""
    header = connection.execute(
        """
        SELECT p.call_id, p.parser_version, p.qa_boundary_segment_index,
               e.company_quarter_id
        FROM parsed_earnings_call AS p
        JOIN earnings_call AS e ON e.call_id = p.call_id
        WHERE p.call_id = ?
        """,
        (call_id,),
    ).fetchone()
    if header is None:
        raise LookupError(f"No parsed earnings-call view: {call_id}")
    header_values = _row_as_dict(header)

    remark_rows = connection.execute(
        """
        SELECT rc.remark_chunk_id, pr.segment_index, rc.chunk_index, pr.speaker,
               rc.text, rc.eligible_for_synthesis, rc.exclusion_reason
        FROM remark_chunk AS rc
        JOIN prepared_remark AS pr ON pr.remark_id = rc.remark_id
        WHERE pr.call_id = ?
        ORDER BY pr.segment_index, rc.chunk_index
        """,
        (call_id,),
    ).fetchall()
    prepared_units = tuple(
        PreparedRemarkUnit(
            unit_id=row["remark_chunk_id"],
            segment_index=row["segment_index"],
            unit_index=row["chunk_index"],
            speaker=row["speaker"],
            text=row["text"],
            eligible_for_synthesis=bool(row["eligible_for_synthesis"]),
            exclusion_reason=row["exclusion_reason"],
        )
        for row in remark_rows
    )

    turn_rows = connection.execute(
        """
        SELECT r.round_index, r.announced_analyst, r.analyst,
               t.turn_id, t.segment_index, t.speaker, t.role, t.text,
               t.role_evidence
        FROM qa_round AS r
        JOIN qa_turn AS t ON t.round_id = r.round_id
        WHERE r.call_id = ?
        ORDER BY r.round_index,
                 CASE t.role WHEN 'question' THEN 0 ELSE 1 END,
                 t.turn_index
        """,
        (call_id,),
    ).fetchall()
    rounds: dict[int, dict] = {}
    for row in turn_rows:
        round_data = rounds.setdefault(
            row["round_index"],
            {
                "announced_analyst": row["announced_analyst"],
                "analyst": row["analyst"],
                "question_turns": [],
                "answer_turns": [],
            },
        )
        turn = QATurn(
            turn_id=row["turn_id"],
            segment_index=row["segment_index"],
            speaker=row["speaker"],
            role=row["role"],
            text=row["text"],
            role_evidence=row["role_evidence"],
        )
        round_data[f"{turn.role}_turns"].append(turn)
    qa_rounds = tuple(
        QARound(
            round_index=round_index,
            announced_analyst=data["announced_analyst"],
            analyst=data["analyst"],
            question_turns=tuple(data["question_turns"]),
            answer_turns=tuple(data["answer_turns"]),
        )
        for round_index, data in sorted(rounds.items())
    )

    excluded_segments = tuple(
        ExcludedTranscriptSegment(
            segment_index=row["segment_index"],
            speaker=row["speaker"],
            text=row["text"],
            reason=row["reason"],
        )
        for row in connection.execute(
            """
            SELECT segment_index, speaker, text, reason
            FROM excluded_transcript_segment
            WHERE call_id = ?
            ORDER BY segment_index, reason
            """,
            (call_id,),
        ).fetchall()
    )
    issues = tuple(
        CallParseIssue(
            code=row["code"],
            segment_index=row["segment_index"],
            detail=row["detail"],
        )
        for row in connection.execute(
            """
            SELECT code, segment_index, detail
            FROM call_parse_issue
            WHERE call_id = ?
            ORDER BY issue_index
            """,
            (call_id,),
        ).fetchall()
    )
    return ParsedEarningsCall(
        call_id=header_values["call_id"],
        parser_version=header_values["parser_version"],
        qa_boundary_segment_index=header_values["qa_boundary_segment_index"],
        prepared_units=prepared_units,
        qa_rounds=qa_rounds,
        excluded_segments=excluded_segments,
        issues=issues,
        company_quarter_id=header_values["company_quarter_id"],
    )


def fetch_valid_extract_scopes(connection) -> tuple[ExtractScope, ...]:
    """Return only unambiguous, fully parsed company-quarter source bundles."""
    rows = connection.execute(
        """
        WITH call_scope AS (
            SELECT company_quarter_id, MIN(call_id) AS call_id, MIN(cik) AS cik
            FROM earnings_call
            WHERE company_quarter_id IS NOT NULL
            GROUP BY company_quarter_id
            HAVING COUNT(*) = 1
        ), filing_scope AS (
            SELECT company_quarter_id, MIN(accession_number) AS accession_number
            FROM filing
            WHERE company_quarter_id IS NOT NULL AND parse_complete = 1
            GROUP BY company_quarter_id
            HAVING COUNT(*) = 1
        )
        SELECT c.company_quarter_id, c.cik, c.call_id, f.accession_number
        FROM call_scope AS c
        JOIN filing_scope AS f ON f.company_quarter_id = c.company_quarter_id
        JOIN parsed_earnings_call AS p ON p.call_id = c.call_id
        WHERE EXISTS (
            SELECT 1 FROM filing_chunk AS fc
            WHERE fc.accession_number = f.accession_number
        )
          AND EXISTS (
            SELECT 1 FROM remark_chunk AS rc
            JOIN prepared_remark AS pr ON pr.remark_id = rc.remark_id
            WHERE pr.call_id = c.call_id
        )
          AND EXISTS (
            SELECT 1 FROM qa_turn AS qt
            JOIN qa_round AS qr ON qr.round_id = qt.round_id
            WHERE qr.call_id = c.call_id
        )
        ORDER BY c.company_quarter_id
        """
    ).fetchall()
    return tuple(ExtractScope(**_row_as_dict(row)) for row in rows)


def fetch_extract_source(connection, company_quarter_id: str) -> ExtractSource:
    """Load one complete tri-view source after enforcing the scope invariant."""
    matching = [
        scope
        for scope in fetch_valid_extract_scopes(connection)
        if scope.company_quarter_id == company_quarter_id
    ]
    if not matching:
        raise LookupError(
            f"Company quarter is not a valid extract scope: {company_quarter_id}"
        )
    scope = matching[0]
    return ExtractSource(
        scope=scope,
        earnings_call=fetch_parsed_earnings_call(connection, scope.call_id),
        filing=fetch_company_filing(connection, scope.accession_number),
        filing_chunks=fetch_filing_chunks(connection, scope.accession_number),
    )
