"""Persisted records for prepared remarks and their retained chunks.

Split out of the old transcript_data.py so that parsing/prepared_remarks.py
(the pure parser) has no reason to import anything with a .save() method.
PreparedRemark/RemarkChunk are built from the parser's plain ParsedRemark/
ParsedRemarkChunk output via from_parsed_remark()/from_parsed_chunk(),
mirroring corpus.transcript.QuestionAnswerPair.from_parsed_pair(). The
replace_for_call() diff/persist logic that used to live as a method on
PreparedRemarksParser lives here as a plain function.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


create_table_prepared_remark = """
CREATE TABLE IF NOT EXISTS prepared_remark (
    remark_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES earnings_call(call_id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL CHECK (word_count >= 1),
    parser_version TEXT NOT NULL,
    UNIQUE (call_id, segment_index)
) STRICT;
"""


create_table_remark_chunk = """
CREATE TABLE IF NOT EXISTS remark_chunk (
    remark_chunk_id TEXT PRIMARY KEY,
    remark_id TEXT NOT NULL REFERENCES prepared_remark(remark_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL CHECK (word_count >= 1),
    source_start_char INTEGER NOT NULL CHECK (source_start_char >= 0),
    source_end_char INTEGER NOT NULL CHECK (source_end_char > source_start_char),
    sentence_index INTEGER NOT NULL CHECK (sentence_index >= 0),
    eligible_for_synthesis INTEGER NOT NULL CHECK (eligible_for_synthesis IN (0, 1)),
    exclusion_reason TEXT,
    parser_version TEXT NOT NULL,
    UNIQUE (remark_id, chunk_index)
) STRICT;
"""


@dataclass(frozen=True)
class PreparedRemark:
    """One original non-operator prepared-remarks transcript segment."""

    remark_id: str
    call_id: str
    segment_index: int
    speaker: str
    text: str
    word_count: int
    parser_version: str

    @classmethod
    def make_remark_id(cls, call_id: str, segment_index: int) -> str:
        return f"{call_id}:remark:{segment_index}"

    @classmethod
    def from_parsed_remark(cls, parsed) -> "PreparedRemark":
        """Build the persisted record from parsing.prepared_remarks.ParsedRemark."""
        return cls(
            remark_id=parsed.remark_id,
            call_id=parsed.call_id,
            segment_index=parsed.segment_index,
            speaker=parsed.speaker,
            text=parsed.text,
            word_count=parsed.word_count,
            parser_version=parsed.parser_version,
        )

    def save(self, connection, *, commit: bool = True) -> str:
        connection.execute(
            """
            INSERT INTO prepared_remark (
                remark_id, call_id, segment_index, speaker, text, word_count,
                parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(remark_id) DO UPDATE SET
                call_id = excluded.call_id,
                segment_index = excluded.segment_index,
                speaker = excluded.speaker,
                text = excluded.text,
                word_count = excluded.word_count,
                parser_version = excluded.parser_version
            """,
            (
                self.remark_id,
                self.call_id,
                self.segment_index,
                self.speaker,
                self.text,
                self.word_count,
                self.parser_version,
            ),
        )
        if commit:
            connection.commit()
        logger.info("Saved prepared remark remark_id=%s", self.remark_id)
        return self.remark_id


@dataclass(frozen=True)
class RemarkChunk:
    """One short, provenance-preserving sentence or clause from a remark."""

    remark_chunk_id: str
    remark_id: str
    chunk_index: int
    text: str
    word_count: int
    source_start_char: int
    source_end_char: int
    sentence_index: int
    eligible_for_synthesis: bool
    exclusion_reason: str | None
    parser_version: str

    @classmethod
    def make_remark_chunk_id(cls, remark_id: str, chunk_index: int) -> str:
        return f"{remark_id}:chunk:{chunk_index}"

    @classmethod
    def from_parsed_chunk(cls, parsed) -> "RemarkChunk":
        """Build the persisted record from parsing.prepared_remarks.ParsedRemarkChunk."""
        return cls(
            remark_chunk_id=parsed.remark_chunk_id,
            remark_id=parsed.remark_id,
            chunk_index=parsed.chunk_index,
            text=parsed.text,
            word_count=parsed.word_count,
            source_start_char=parsed.source_start_char,
            source_end_char=parsed.source_end_char,
            sentence_index=parsed.sentence_index,
            eligible_for_synthesis=parsed.eligible_for_synthesis,
            exclusion_reason=parsed.exclusion_reason,
            parser_version=parsed.parser_version,
        )

    def save(self, connection, *, commit: bool = True) -> str:
        connection.execute(
            """
            INSERT INTO remark_chunk (
                remark_chunk_id, remark_id, chunk_index, text, word_count,
                source_start_char, source_end_char, sentence_index,
                eligible_for_synthesis, exclusion_reason, parser_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(remark_chunk_id) DO UPDATE SET
                remark_id = excluded.remark_id,
                chunk_index = excluded.chunk_index,
                text = excluded.text,
                word_count = excluded.word_count,
                source_start_char = excluded.source_start_char,
                source_end_char = excluded.source_end_char,
                sentence_index = excluded.sentence_index,
                eligible_for_synthesis = excluded.eligible_for_synthesis,
                exclusion_reason = excluded.exclusion_reason,
                parser_version = excluded.parser_version
            """,
            (
                self.remark_chunk_id,
                self.remark_id,
                self.chunk_index,
                self.text,
                self.word_count,
                self.source_start_char,
                self.source_end_char,
                self.sentence_index,
                int(self.eligible_for_synthesis),
                self.exclusion_reason,
                self.parser_version,
            ),
        )
        if commit:
            connection.commit()
        logger.info("Saved remark chunk remark_chunk_id=%s", self.remark_chunk_id)
        return self.remark_chunk_id


def replace_for_call(connection, result) -> None:
    """Atomically replace one call's generated remarks and chunks.

    ``result`` is a parsing.prepared_remarks.PreparedRemarksParseResult (duck
    typed here, not imported, to keep corpus/ free of a dependency on
    parsing/ — only ingestion/ is meant to import both layers).

    Existing chunk IDs are updated in place. Removing a chunk that is
    already used by retrieval or audit data is refused rather than allowed
    to cascade-delete human review records.
    """
    if any(remark.call_id != result.call_id for remark in result.remarks):
        raise ValueError("A prepared-remarks parse result must contain one call.")
    call_id = result.call_id
    current_chunk_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT rc.remark_chunk_id
            FROM remark_chunk AS rc
            JOIN prepared_remark AS pr ON pr.remark_id = rc.remark_id
            WHERE pr.call_id = ?
            """,
            (call_id,),
        )
    }
    next_chunk_ids = {chunk.remark_chunk_id for chunk in result.chunks}
    removed_chunk_ids = current_chunk_ids - next_chunk_ids
    if removed_chunk_ids:
        placeholders = ", ".join("?" for _ in removed_chunk_ids)
        candidate_count = 0
        legacy_schema_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'remark_filing_candidate'"
        ).fetchone()
        if legacy_schema_exists:
            candidate_count = connection.execute(
                f"SELECT COUNT(*) FROM remark_filing_candidate "
                f"WHERE remark_chunk_id IN ({placeholders})",
                tuple(removed_chunk_ids),
            ).fetchone()[0]
        retrieval_schema_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'remark_filing_retrieval_candidate'"
        ).fetchone()
        if retrieval_schema_exists:
            candidate_count += connection.execute(
                f"SELECT COUNT(*) FROM remark_filing_retrieval_candidate "
                f"WHERE remark_chunk_id IN ({placeholders})",
                tuple(removed_chunk_ids),
            ).fetchone()[0]
        if candidate_count:
            raise ValueError(
                f"Reparse would remove {candidate_count} retrieval candidates for "
                f"{call_id}; use versioned chunk identities instead."
            )
    for remark in result.remarks:
        PreparedRemark.from_parsed_remark(remark).save(connection, commit=False)
    for chunk in result.chunks:
        RemarkChunk.from_parsed_chunk(chunk).save(connection, commit=False)
    if removed_chunk_ids:
        connection.execute(
            f"DELETE FROM remark_chunk WHERE remark_chunk_id IN ({placeholders})",
            tuple(removed_chunk_ids),
        )
    next_remark_ids = {remark.remark_id for remark in result.remarks}
    stale_remarks = connection.execute(
        "SELECT remark_id FROM prepared_remark WHERE call_id = ?", (call_id,)
    ).fetchall()
    for (remark_id,) in stale_remarks:
        if remark_id not in next_remark_ids:
            connection.execute("DELETE FROM prepared_remark WHERE remark_id = ?", (remark_id,))
