from dataclasses import dataclass
import logging

from .transcript_repository import create_table_prepared_remark, create_table_remark_chunk
from .call_alignment_repository import setup_call_alignment_tables

logger = logging.getLogger(__name__)


create_table_earnings_call = """
CREATE TABLE IF NOT EXISTS earnings_call (
    call_id TEXT PRIMARY KEY,
    cik TEXT NOT NULL REFERENCES company(cik),
    company_quarter_id TEXT REFERENCES company_quarter(company_quarter_id),
    ticker TEXT NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    call_date TEXT NOT NULL,
    company_name TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    candidate_pair_count INTEGER NOT NULL CHECK (candidate_pair_count >= 0),
    clean_pair_count INTEGER NOT NULL CHECK (
        clean_pair_count >= 0 AND clean_pair_count <= candidate_pair_count
    ),
    UNIQUE (cik, year, quarter)
) STRICT;
"""


create_table_question_answer_pair = """
CREATE TABLE IF NOT EXISTS question_answer_pair (
    pair_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES earnings_call(call_id),
    pair_index INTEGER NOT NULL CHECK (pair_index >= 0),
    analyst TEXT NOT NULL,
    announced_analyst TEXT,
    answer_speaker TEXT NOT NULL,
    question_text TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    question_segment_index INTEGER NOT NULL CHECK (question_segment_index >= 0),
    answer_segment_index INTEGER NOT NULL CHECK (answer_segment_index >= 0),
    UNIQUE (call_id, pair_index)
) STRICT;
"""


create_table_transcript_ingestion_status = """
CREATE TABLE IF NOT EXISTS transcript_ingestion_status (
    cik TEXT NOT NULL REFERENCES company(cik),
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    status TEXT NOT NULL CHECK (
        status IN ('saved', 'saved_zero_clean', 'unavailable', 'failed')
    ),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    last_error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (cik, year, quarter)
) STRICT;
"""


@dataclass
class EarningsCall:
    call_id: str
    cik: str
    ticker: str
    year: int
    quarter: int
    call_date: str
    company_name: str
    raw_path: str
    source_dataset: str
    parser_version: str
    candidate_pair_count: int
    clean_pair_count: int
    company_quarter_id: str | None = None

    @classmethod
    def make_call_id(cls, cik: str, year: int, quarter: int) -> str:
        return f"{cik}:{year}:Q{quarter}"

    def save(self, connection, *, commit: bool = True) -> str:
        connection.execute(
            """
            INSERT INTO earnings_call (
                call_id, cik, ticker, year, quarter, call_date, company_name,
                raw_path, source_dataset, parser_version, candidate_pair_count,
                clean_pair_count, company_quarter_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET
                cik = excluded.cik,
                ticker = excluded.ticker,
                year = excluded.year,
                quarter = excluded.quarter,
                call_date = excluded.call_date,
                company_name = excluded.company_name,
                raw_path = excluded.raw_path,
                source_dataset = excluded.source_dataset,
                parser_version = excluded.parser_version,
                candidate_pair_count = excluded.candidate_pair_count,
                clean_pair_count = excluded.clean_pair_count,
                company_quarter_id = excluded.company_quarter_id
            """,
            (
                self.call_id,
                self.cik,
                self.ticker,
                self.year,
                self.quarter,
                self.call_date,
                self.company_name,
                self.raw_path,
                self.source_dataset,
                self.parser_version,
                self.candidate_pair_count,
                self.clean_pair_count,
                self.company_quarter_id
            ),
        )
        if commit:
            connection.commit()
        logger.info("Saved earnings call call_id=%s", self.call_id)
        return self.call_id


@dataclass
class QuestionAnswerPair:
    pair_id: str
    call_id: str
    pair_index: int
    analyst: str
    announced_analyst: str | None
    answer_speaker: str
    question_text: str
    answer_text: str
    question_segment_index: int
    answer_segment_index: int

    @classmethod
    def from_parsed_pair(
        cls,
        *,
        call_id: str,
        pair_index: int,
        # String annotation: ParsedQuestionAnswerPair lives in
        # parsing/transcript_parser.py. corpus/ doesn't import parsing/
        # (only ingestion/ is meant to import both layers), so this stays
        # a documentation-only forward reference, never resolved at runtime.
        parsed_pair: "ParsedQuestionAnswerPair",
    ) -> "QuestionAnswerPair":
        if not parsed_pair.is_clean:
            raise ValueError("Only clean parsed question-answer pairs can be saved.")

        return cls(
            pair_id=f"{call_id}:{pair_index}",
            call_id=call_id,
            pair_index=pair_index,
            analyst=parsed_pair.analyst,
            announced_analyst=parsed_pair.announced_analyst,
            answer_speaker=parsed_pair.answer_speakers[0],
            question_text=parsed_pair.question,
            answer_text=parsed_pair.answer,
            question_segment_index=parsed_pair.question_segment_indexes[0],
            answer_segment_index=parsed_pair.answer_segment_indexes[0],
        )

    def save(self, connection, *, commit: bool = True) -> str:
        connection.execute(
            """
            INSERT INTO question_answer_pair (
                pair_id, call_id, pair_index, analyst, announced_analyst,
                answer_speaker, question_text, answer_text,
                question_segment_index, answer_segment_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair_id) DO UPDATE SET
                call_id = excluded.call_id,
                pair_index = excluded.pair_index,
                analyst = excluded.analyst,
                announced_analyst = excluded.announced_analyst,
                answer_speaker = excluded.answer_speaker,
                question_text = excluded.question_text,
                answer_text = excluded.answer_text,
                question_segment_index = excluded.question_segment_index,
                answer_segment_index = excluded.answer_segment_index
            """,
            (
                self.pair_id,
                self.call_id,
                self.pair_index,
                self.analyst,
                self.announced_analyst,
                self.answer_speaker,
                self.question_text,
                self.answer_text,
                self.question_segment_index,
                self.answer_segment_index,
            ),
        )
        if commit:
            connection.commit()
        logger.info("Saved question-answer pair pair_id=%s", self.pair_id)
        return self.pair_id


@dataclass
class TranscriptIngestionStatus:
    cik: str
    year: int
    quarter: int
    status: str
    attempt_count: int
    last_error: str | None
    updated_at: str

    def save(self, connection, *, commit: bool = True) -> None:
        connection.execute(
            """
            INSERT INTO transcript_ingestion_status (
                cik, year, quarter, status, attempt_count, last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cik, year, quarter) DO UPDATE SET
                status = excluded.status,
                attempt_count = excluded.attempt_count,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (
                self.cik,
                self.year,
                self.quarter,
                self.status,
                self.attempt_count,
                self.last_error,
                self.updated_at,
            ),
        )
        if commit:
            connection.commit()
        logger.info(
            "Saved transcript ingestion status cik=%s year=%s quarter=%s status=%s",
            self.cik,
            self.year,
            self.quarter,
            self.status,
        )


def setup_transcript_tables(connection) -> None:
    connection.execute(create_table_earnings_call)
    connection.execute(create_table_question_answer_pair)
    connection.execute(create_table_prepared_remark)
    connection.execute(create_table_remark_chunk)
    connection.execute(create_table_transcript_ingestion_status)
    setup_call_alignment_tables(connection)
