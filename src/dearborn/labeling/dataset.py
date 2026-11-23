"""Load the fixed demonstrations and holdout data for label-model evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dearborn.constants import DATA_ROOT

from .audit_app import current_labels, load_audit, load_label_events
from .contracts import ALLOWED_LABELS, LabelingInput, PreparedRemarkLabel


DEFAULT_RUN_DIRECTORY = (
    DATA_ROOT / "experiments" / "boilerplate_similarity" / "pilot-10"
)
DEMONSTRATION_AUDIT_IDS = (15, 28, 66, 7, 64, 96, 38, 61, 92, 4, 22, 53)
TEST_TICKER = "EW"


@dataclass(frozen=True)
class LabeledLabelingExample:
    """One model input paired with its human audit label."""

    audit_id: int
    input: LabelingInput
    human_label: PreparedRemarkLabel


@dataclass(frozen=True)
class LabelingEvaluationData:
    """The fixed few-shot demonstrations and locked transcript holdout."""

    demonstrations: tuple[LabeledLabelingExample, ...]
    test_examples: tuple[LabeledLabelingExample, ...]


class LabelingEvaluationDataLoader:
    """Join the blinded audit, label ledger, and similarity-score records."""

    def __init__(self, run_directory: Path = DEFAULT_RUN_DIRECTORY) -> None:
        self.run_directory = Path(run_directory)
        self.audit_path = self.run_directory / "blinded_audit.jsonl"
        self.labels_path = self.run_directory / "blinded_audit_labels.jsonl"
        self.scores_path = self.run_directory / "sentence_similarity_scores.jsonl"

    def load(self) -> LabelingEvaluationData:
        audit_records = load_audit(self.audit_path)
        audit_by_id = {int(record["audit_id"]): record for record in audit_records}
        labels = current_labels(load_label_events(self.labels_path))
        scores_by_sentence_id = self._load_scores()

        examples = {
            audit_id: self._build_example(
                audit_id=audit_id,
                audit_record=audit_record,
                human_label=labels.get(audit_id),
                scores_by_sentence_id=scores_by_sentence_id,
            )
            for audit_id, audit_record in audit_by_id.items()
        }

        missing_demonstrations = [
            audit_id
            for audit_id in DEMONSTRATION_AUDIT_IDS
            if audit_id not in examples
        ]
        if missing_demonstrations:
            raise ValueError(
                f"Missing demonstration audit IDs: {missing_demonstrations}"
            )

        demonstrations = tuple(
            examples[audit_id] for audit_id in DEMONSTRATION_AUDIT_IDS
        )
        test_examples = tuple(
            examples[int(record["audit_id"])]
            for record in audit_records
            if record["ticker"] == TEST_TICKER
        )

        self._validate(demonstrations, test_examples)
        return LabelingEvaluationData(
            demonstrations=demonstrations,
            test_examples=test_examples,
        )

    def _load_scores(self) -> dict[str, dict]:
        records: dict[str, dict] = {}
        with self.scores_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                sentence_id = record.get("sentence_id")
                if not sentence_id:
                    raise ValueError(
                        f"Missing sentence_id at {self.scores_path}:{line_number}"
                    )
                if sentence_id in records:
                    raise ValueError(
                        f"Duplicate sentence_id {sentence_id!r} in {self.scores_path}"
                    )
                records[sentence_id] = record
        return records

    def _build_example(
        self,
        audit_id: int,
        audit_record: dict,
        human_label: str | None,
        scores_by_sentence_id: dict[str, dict],
    ) -> LabeledLabelingExample:
        if human_label not in ALLOWED_LABELS:
            raise ValueError(f"Audit ID {audit_id} has no valid human label.")

        sentence_id = str(audit_record["sentence_id"])
        score_record = scores_by_sentence_id.get(sentence_id)
        if score_record is None:
            raise ValueError(f"No similarity score found for {sentence_id!r}.")

        return LabeledLabelingExample(
            audit_id=audit_id,
            input=LabelingInput(
                sentence_id=sentence_id,
                ticker=str(audit_record["ticker"]),
                current_statement=str(audit_record["current_sentence"]),
                previous_call_match=str(audit_record["previous_call_match"]),
                same_company_similarity=float(
                    score_record["same_company_similarity"]
                ),
            ),
            human_label=cast(PreparedRemarkLabel, human_label),
        )

    @staticmethod
    def _validate(
        demonstrations: tuple[LabeledLabelingExample, ...],
        test_examples: tuple[LabeledLabelingExample, ...],
    ) -> None:
        if len(demonstrations) != 12:
            raise ValueError(f"Expected 12 demonstrations; found {len(demonstrations)}.")
        if len(test_examples) != 17:
            raise ValueError(f"Expected 17 EW test examples; found {len(test_examples)}.")

        demonstration_ids = {example.audit_id for example in demonstrations}
        test_ids = {example.audit_id for example in test_examples}
        if demonstration_ids & test_ids:
            raise ValueError("Demonstration and test audit IDs overlap.")

        demonstration_counts = {
            label: sum(example.human_label == label for example in demonstrations)
            for label in ALLOWED_LABELS
        }
        if any(count != 3 for count in demonstration_counts.values()):
            raise ValueError(
                "Expected three demonstrations per label; "
                f"found {demonstration_counts}."
            )

        expected_test_counts = {
            "boilerplate": 3,
            "administrative": 2,
            "repeated_substantive": 5,
            "novel_substantive": 7,
        }
        test_counts = {
            label: sum(example.human_label == label for example in test_examples)
            for label in ALLOWED_LABELS
        }
        if test_counts != expected_test_counts:
            raise ValueError(
                f"Unexpected EW test-label distribution: {test_counts}."
            )
