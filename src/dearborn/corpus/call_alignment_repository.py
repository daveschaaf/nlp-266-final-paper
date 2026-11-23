"""Persistence for complete parsed Q&A turns used by ``dearborn.extract``."""

from __future__ import annotations


create_table_parsed_earnings_call = """
CREATE TABLE IF NOT EXISTS parsed_earnings_call (
    call_id TEXT PRIMARY KEY REFERENCES earnings_call(call_id) ON DELETE CASCADE,
    parser_version TEXT NOT NULL,
    qa_boundary_segment_index INTEGER NOT NULL
) STRICT;
"""

create_table_qa_round = """
CREATE TABLE IF NOT EXISTS qa_round (
    round_id TEXT PRIMARY KEY,
    call_id TEXT NOT NULL REFERENCES parsed_earnings_call(call_id) ON DELETE CASCADE,
    round_index INTEGER NOT NULL CHECK (round_index >= 0),
    announced_analyst TEXT NOT NULL,
    analyst TEXT NOT NULL,
    UNIQUE (call_id, round_index)
) STRICT;
"""

create_table_qa_turn = """
CREATE TABLE IF NOT EXISTS qa_turn (
    turn_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES qa_round(round_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('question', 'answer')),
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    role_evidence TEXT NOT NULL,
    UNIQUE (round_id, role, turn_index)
) STRICT;
"""

create_table_excluded_transcript_segment = """
CREATE TABLE IF NOT EXISTS excluded_transcript_segment (
    call_id TEXT NOT NULL REFERENCES parsed_earnings_call(call_id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (call_id, segment_index, reason)
) STRICT;
"""

create_table_call_parse_issue = """
CREATE TABLE IF NOT EXISTS call_parse_issue (
    call_id TEXT NOT NULL REFERENCES parsed_earnings_call(call_id) ON DELETE CASCADE,
    issue_index INTEGER NOT NULL CHECK (issue_index >= 0),
    code TEXT NOT NULL,
    segment_index INTEGER,
    detail TEXT NOT NULL,
    PRIMARY KEY (call_id, issue_index)
) STRICT;
"""


def setup_call_alignment_tables(connection) -> None:
    connection.execute(create_table_parsed_earnings_call)
    connection.execute(create_table_qa_round)
    connection.execute(create_table_qa_turn)
    connection.execute(create_table_excluded_transcript_segment)
    connection.execute(create_table_call_parse_issue)


def replace_parsed_earnings_call(connection, parsed_call) -> None:
    """Replace one complete parsed-call view atomically within a caller transaction."""
    connection.execute(
        """
        INSERT INTO parsed_earnings_call (
            call_id, parser_version, qa_boundary_segment_index
        ) VALUES (?, ?, ?)
        ON CONFLICT(call_id) DO UPDATE SET
            parser_version = excluded.parser_version,
            qa_boundary_segment_index = excluded.qa_boundary_segment_index
        """,
        (
            parsed_call.call_id,
            parsed_call.parser_version,
            parsed_call.qa_boundary_segment_index,
        ),
    )
    connection.execute("DELETE FROM qa_round WHERE call_id = ?", (parsed_call.call_id,))
    connection.execute(
        "DELETE FROM excluded_transcript_segment WHERE call_id = ?",
        (parsed_call.call_id,),
    )
    connection.execute("DELETE FROM call_parse_issue WHERE call_id = ?", (parsed_call.call_id,))

    for qa_round in parsed_call.qa_rounds:
        round_id = f"{parsed_call.call_id}:qa_round:{qa_round.round_index}"
        connection.execute(
            """
            INSERT INTO qa_round (round_id, call_id, round_index, announced_analyst, analyst)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                round_id,
                parsed_call.call_id,
                qa_round.round_index,
                qa_round.announced_analyst,
                qa_round.analyst,
            ),
        )
        for role, turns in (("question", qa_round.question_turns), ("answer", qa_round.answer_turns)):
            for turn_index, turn in enumerate(turns):
                connection.execute(
                    """
                    INSERT INTO qa_turn (
                        turn_id, round_id, role, turn_index, segment_index, speaker, text,
                        role_evidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.turn_id,
                        round_id,
                        role,
                        turn_index,
                        turn.segment_index,
                        turn.speaker,
                        turn.text,
                        turn.role_evidence,
                    ),
                )
    for segment in parsed_call.excluded_segments:
        connection.execute(
            """
            INSERT INTO excluded_transcript_segment (call_id, segment_index, speaker, text, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                parsed_call.call_id,
                segment.segment_index,
                segment.speaker,
                segment.text,
                segment.reason,
            ),
        )
    for issue_index, issue in enumerate(parsed_call.issues):
        connection.execute(
            """
            INSERT INTO call_parse_issue (call_id, issue_index, code, segment_index, detail)
            VALUES (?, ?, ?, ?, ?)
            """,
            (parsed_call.call_id, issue_index, issue.code, issue.segment_index, issue.detail),
        )
