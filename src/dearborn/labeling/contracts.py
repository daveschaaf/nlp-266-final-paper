"""Shared contracts for human and model-assigned prepared-remark labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


PreparedRemarkLabel: TypeAlias = Literal[
    "boilerplate",
    "administrative",
    "repeated_substantive",
    "novel_substantive",
]

ALLOWED_LABELS: frozenset[str] = frozenset(
    {
        "boilerplate",
        "administrative",
        "repeated_substantive",
        "novel_substantive",
    }
)


@dataclass(frozen=True)
class LabelingInput:
    """One prepared remark and its prior-company comparison."""

    sentence_id: str
    ticker: str
    current_statement: str
    previous_call_match: str
    same_company_similarity: float


@dataclass(frozen=True)
class LabelingOutput:
    """One structured label returned by a human or language model."""

    sentence_id: str
    label: PreparedRemarkLabel
    confidence: float
    rationale: str
    model_id: str
    prompt_version: str
