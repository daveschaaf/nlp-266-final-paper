from datetime import date
import logging
import sqlite3

from ..constants import DATA_ROOT
from .company import (
    CompanyQuarter,
    FISCAL_PERIOD_TO_QUARTER,
    create_table_company,
    create_table_company_quarter,
    create_table_filing,
    create_table_filing_chunk,
    make_company_quarter_id,
)
from .transcript import setup_transcript_tables

logger = logging.getLogger(__name__)


def sqlite_connection():
    connection = sqlite3.connect(DATA_ROOT / "corpus.sqlite")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row
    return connection


def migrate_schema(connection) -> None:
    """Add columns introduced after an existing corpus database was created."""
    connection.execute(create_table_company_quarter)

    filing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(filing)")
    }

    if "parser_version" not in filing_columns:
        connection.execute(
            "ALTER TABLE filing ADD COLUMN parser_version TEXT NOT NULL DEFAULT ''"
        )
    if "parse_complete" not in filing_columns:
        connection.execute(
            "ALTER TABLE filing ADD COLUMN parse_complete INTEGER NOT NULL DEFAULT 0"
        )
    if "company_quarter_id" not in filing_columns:
        connection.execute(
            "ALTER TABLE filing ADD COLUMN company_quarter_id TEXT"
        )
    call_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(earnings_call)")
    }
    if "company_quarter_id" not in call_columns:
        connection.execute(
            "ALTER TABLE earnings_call ADD COLUMN company_quarter_id TEXT"
        )

MAX_CALL_LAG_DAYS = 120


def _parse_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _retrieval_hint(call) -> dict:
    return {
        "call_id": call["call_id"],
        "cik": call["cik"],
        "call_date": call["call_date"],
        "calendar_year": call["year"],
        "calendar_quarter": call["quarter"],
    }


def backfill_company_quarters(
    connection,
    *,
    excluded_call_ids: set[str] | None = None,
) -> list[dict]:
    """Link earnings calls to filings by fiscal-period end date.

    For each call, the candidate filings are those for the same CIK whose
    fiscal_period_end falls within MAX_CALL_LAG_DAYS before the call date.
    The most recent such period wins; the filing's own fiscal metadata mints
    the canonical company_quarter_id. Calls with no candidate, or with an
    ambiguous tie, are returned as retrieval hints and left unlinked.
    ``excluded_call_ids`` are also left unlinked: a newer resolved filing for
    those calls failed to save, so linking to an older local filing is invalid.
    """
    excluded_call_ids = excluded_call_ids or set()
    filings_by_cik: dict[str, list] = {}
    for filing in connection.execute(
        """
        SELECT accession_number, cik, form, fiscal_year, fiscal_period,
               fiscal_period_start, fiscal_period_end
        FROM filing
        """
    ).fetchall():
        period_end = _parse_date(filing["fiscal_period_end"])
        if period_end is None:
            continue
        filings_by_cik.setdefault(str(filing["cik"]).zfill(10), []).append(
            (period_end, filing)
        )

    hints = []
    for call in connection.execute(
        "SELECT call_id, cik, year, quarter, call_date FROM earnings_call"
    ).fetchall():
        if call["call_id"] in excluded_call_ids:
            connection.execute(
                "UPDATE earnings_call SET company_quarter_id = NULL WHERE call_id = ?",
                (call["call_id"],),
            )
            hints.append(_retrieval_hint(call))
            continue
        call_date = _parse_date(call["call_date"])
        if call_date is None:
            hints.append(_retrieval_hint(call))
            continue

        candidates = [
            (period_end, filing)
            for period_end, filing in filings_by_cik.get(
                str(call["cik"]).zfill(10), []
            )
            if 0 <= (call_date - period_end).days <= MAX_CALL_LAG_DAYS
        ]
        if not candidates:
            hints.append(_retrieval_hint(call))
            continue

        latest_end = max(period_end for period_end, _ in candidates)
        winners = [filing for period_end, filing in candidates if period_end == latest_end]
        if len(winners) != 1:
            logger.warning(
                "Ambiguous filing match for call_id=%s: %s",
                call["call_id"],
                [filing["accession_number"] for filing in winners],
            )
            hints.append(_retrieval_hint(call))
            continue

        filing = winners[0]
        quarter_id = make_company_quarter_id(
            cik=filing["cik"],
            fiscal_year=filing["fiscal_year"],
            fiscal_period=filing["fiscal_period"],
        )
        CompanyQuarter(
            cik=filing["cik"],
            company_quarter_id=quarter_id,
            fiscal_year=filing["fiscal_year"],
            fiscal_quarter=FISCAL_PERIOD_TO_QUARTER[
                filing["fiscal_period"].upper().strip()
            ],
            fiscal_quarter_start=filing["fiscal_period_start"],
            fiscal_quarter_end=filing["fiscal_period_end"],
        ).save(connection)

        connection.execute(
            "UPDATE filing SET company_quarter_id = ? WHERE accession_number = ?",
            (quarter_id, filing["accession_number"]),
        )
        connection.execute(
            "UPDATE earnings_call SET company_quarter_id = ? WHERE call_id = ?",
            (quarter_id, call["call_id"]),
        )

    return hints


def setup_db() -> None:

    conn = sqlite_connection()
    conn.execute(create_table_company)
    conn.execute(create_table_company_quarter)
    conn.execute(create_table_filing)
    conn.execute(create_table_filing_chunk)
    setup_transcript_tables(conn)
    migrate_schema(conn)
    backfill_company_quarters(conn)
    conn.commit()


if __name__ == "__main__":
    setup_db()
