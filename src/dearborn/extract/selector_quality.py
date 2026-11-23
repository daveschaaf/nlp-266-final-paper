"""Intrinsic, label-free diagnostics for prepared-remark selection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

from .contracts import ExtractTrainingResult
from .embedding_store import ExtractEmbeddingStore


WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
NUMERIC_PATTERN = re.compile(r"(?<!\w)\d[\d,.:/%-]*(?!\w)")


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def numeric_density(text: str) -> float:
    words = word_count(text)
    return len(NUMERIC_PATTERN.findall(text)) / words if words else 0.0


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def selector_quality_report(
    result: ExtractTrainingResult,
    embedding_store: ExtractEmbeddingStore,
) -> dict[str, Any]:
    """Measure compression, duplicate selection, length, and numeric density.

    These are intrinsic diagnostics, not semantic relevance labels.  The
    subsequent blinded audit and FinRank experiment establish semantic quality.
    """
    budgets = [
        budget
        for selector_output in result.selector_outputs
        for budget in selector_output.selection_budget.detach().cpu().tolist()
    ]
    if len(budgets) != len(result.company_quarters):
        raise ValueError("Selector outputs do not align with evaluated company quarters.")

    rows = []
    for company_quarter, selection_budget in zip(result.company_quarters, budgets):
        remark_ids = company_quarter.metadata.remark_ids
        records = embedding_store.fetch(list(remark_ids))
        texts_by_id = {record.input.unit_id: record.input.text for record in records}
        selected_ids = company_quarter.selected_ids
        selected_texts = [texts_by_id[remark_id] for remark_id in selected_ids]
        all_texts = [texts_by_id[remark_id] for remark_id in remark_ids]

        eligible_count = len(remark_ids)
        selected_count = len(selected_ids)
        selected_slot_count = selection_budget
        all_word_count = sum(word_count(text) for text in all_texts)
        selected_word_count = sum(word_count(text) for text in selected_texts)
        selected_positions = [remark_ids.index(remark_id) for remark_id in selected_ids]
        normalized_positions = [
            position / max(eligible_count - 1, 1) for position in selected_positions
        ]

        rows.append(
            {
                "company_quarter_id": company_quarter.company_quarter_id,
                "call_id": company_quarter.call_id,
                "eligible_remark_count": eligible_count,
                "selection_budget": selection_budget,
                "selected_slot_count": selected_slot_count,
                "unique_selected_remark_count": selected_count,
                "unique_selection_rate": selected_count / eligible_count if eligible_count else 0.0,
                "duplicate_slot_rate": (
                    1.0 - (selected_count / selected_slot_count)
                    if selected_slot_count else 0.0
                ),
                "all_remark_word_count": all_word_count,
                "selected_remark_word_count": selected_word_count,
                "word_compression_rate": (
                    selected_word_count / all_word_count if all_word_count else 0.0
                ),
                "mean_all_remark_words": mean([word_count(text) for text in all_texts]),
                "mean_selected_remark_words": (
                    mean([word_count(text) for text in selected_texts]) if selected_texts else 0.0
                ),
                "all_numeric_token_density": numeric_density(" ".join(all_texts)),
                "selected_numeric_token_density": numeric_density(" ".join(selected_texts)),
                "mean_selected_relative_position": (
                    mean(normalized_positions) if normalized_positions else 0.0
                ),
            }
        )

    total_eligible = sum(row["eligible_remark_count"] for row in rows)
    total_selected = sum(row["unique_selected_remark_count"] for row in rows)
    total_slots = sum(row["selected_slot_count"] for row in rows)
    total_all_words = sum(row["all_remark_word_count"] for row in rows)
    total_selected_words = sum(row["selected_remark_word_count"] for row in rows)
    return {
        "report_type": "selector_intrinsic_quality_v1",
        "scope_count": len(rows),
        "summary": {
            "total_eligible_remarks": total_eligible,
            "total_unique_selected_remarks": total_selected,
            "weighted_unique_selection_rate": total_selected / total_eligible if total_eligible else 0.0,
            "weighted_duplicate_slot_rate": 1.0 - (total_selected / total_slots) if total_slots else 0.0,
            "weighted_word_compression_rate": (
                total_selected_words / total_all_words if total_all_words else 0.0
            ),
            "unique_selection_rate": describe([row["unique_selection_rate"] for row in rows]),
            "word_compression_rate": describe([row["word_compression_rate"] for row in rows]),
            "mean_selected_remark_words": describe(
                [row["mean_selected_remark_words"] for row in rows]
            ),
            "selected_numeric_token_density": describe(
                [row["selected_numeric_token_density"] for row in rows]
            ),
            "mean_selected_relative_position": describe(
                [row["mean_selected_relative_position"] for row in rows]
            ),
        },
        "per_company_quarter": rows,
    }


def write_selector_quality_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")
