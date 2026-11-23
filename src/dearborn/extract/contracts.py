"""Shared, dependency-free records at the extract data boundary.

They deliberately contain no parser implementation and no database methods, so
both ``parsing/`` and ``corpus/`` can use the same model-facing contract.
"""

from numpy.typing import NDArray
from typing import Literal
from dataclasses import dataclass, replace
import torch
import numpy as np


@dataclass(frozen=True)
class PreparedRemarkUnit:
    """One short, semantically coherent unit from prepared management remarks."""
    unit_id: str
    segment_index: int
    unit_index: int
    speaker: str
    text: str
    eligible_for_synthesis: bool
    exclusion_reason: str | None


@dataclass(frozen=True)
class QATurn:
    """One speaker turn within a parsed analyst-question/management-answer round."""
    turn_id: str
    segment_index: int
    speaker: str
    role: str
    text: str
    role_evidence: str


@dataclass(frozen=True)
class QARound:
    """All question and answer turns belonging to one analyst exchange."""
    round_index: int
    announced_analyst: str
    analyst: str
    question_turns: tuple[QATurn, ...]
    answer_turns: tuple[QATurn, ...]

    @property
    def turns(self) -> tuple[QATurn, ...]:
        return self.question_turns + self.answer_turns


@dataclass(frozen=True)
class ExcludedTranscriptSegment:
    segment_index: int
    speaker: str
    text: str
    reason: str


@dataclass(frozen=True)
class CallParseIssue:
    code: str
    segment_index: int | None
    detail: str


@dataclass(frozen=True)
class ParsedEarningsCall:
    call_id: str
    parser_version: str
    qa_boundary_segment_index: int
    prepared_units: tuple[PreparedRemarkUnit, ...]
    qa_rounds: tuple[QARound, ...]
    excluded_segments: tuple[ExcludedTranscriptSegment, ...]
    issues: tuple[CallParseIssue, ...]
    company_quarter_id: str | None = None

    @property
    def qa_turns(self) -> tuple[QATurn, ...]:
        """Flatten rounds while preserving their stored question-then-answer order."""
        return tuple(turn for qa_round in self.qa_rounds for turn in qa_round.turns)

@dataclass(frozen=True)
class EmbeddingInput():
    """Text and provenance sent to the frozen embedding encoder and Qdrant."""
    unit_id: str
    source_scope_id: str
    modality: Literal["remark", "qa_question", "qa_answer", "filing"]
    text: str

@dataclass(frozen=True)
class EmbeddingRecord():
    """A cached vector plus enough metadata to recover its source text."""
    input: EmbeddingInput
    encoder_id: str
    embedding_dimension: int
    normalization: str | None
    vector: NDArray[np.float32]

@dataclass(frozen=True)
class AlignmentCompanyQuarter:
    """One unpadded company-quarter example before PyTorch batch collation.

    Positive views belong to this company-quarter. Filing negatives are sampled
    chunks; Q&A negatives are the full call from another scope in the split.
    """
    company_quarter_id: str
    call_id: str
    accession_number: str
    split: Literal['test', 'val', 'train']
    remark_ids: tuple[str, ...]
    remark_vectors: NDArray[np.float32]
    qa_turn_ids: tuple[str, ...]
    qa_roles: tuple[str, ...]
    qa_vectors: NDArray[np.float32]
    filing_chunk_ids: tuple[str, ...]
    filing_vectors: NDArray[np.float32]
    negative_filing_scope_ids: tuple[str, ...]
    negative_filing_chunk_ids: tuple[str, ...]
    negative_filing_vectors: NDArray[np.float32]
    negative_call_company_quarter_id: str | None
    negative_remark_ids: tuple[str, ...]
    negative_remark_vectors: NDArray[np.float32]
    negative_qa_turn_ids: tuple[str, ...]
    negative_qa_roles: tuple[str, ...]
    negative_qa_vectors: NDArray[np.float32]

@dataclass(frozen=True)
class AlignmentMetadata:
    """Non-tensor IDs retained to trace a model decision back to source text."""
    company_quarter_id: str
    call_id: str
    accession_number: str
    split: str

    remark_ids: tuple[str, ...]
    qa_turn_ids: tuple[str, ...]
    filing_chunk_ids: tuple[str, ...]

    negative_filing_scope_ids: tuple[str, ...]
    negative_filing_chunk_ids: tuple[str, ...]
    negative_call_company_quarter_id: str | None
    negative_remark_ids: tuple[str, ...]
    negative_qa_turn_ids: tuple[str, ...]

@dataclass(frozen=True)
class AlignmentBatch:
    """Padded tensors and provenance for a batch of company-quarter examples."""
    remark_vectors: torch.Tensor  # [B, max_remarks, 768]
    remark_mask: torch.BoolTensor # [B, max_remarks]

    qa_vectors: torch.Tensor      # [B, max_qa_turns, 768]
    qa_mask: torch.BoolTensor     # [B, max_qa_turns]
    qa_role_ids: torch.LongTensor # question=0, answer=1, pad=-1

    filing_vectors: torch.Tensor  # [B, max_chunks, 768]
    filing_mask: torch.BoolTensor # [B, max_chunks]

    negative_filing_vectors: torch.Tensor   # [B, max_negative_chunks, 768]
    negative_filing_mask: torch.BoolTensor  # [B, max_negative_chunks]

    negative_remark_vectors: torch.Tensor
    negative_remark_mask: torch.BoolTensor
    negative_qa_vectors: torch.Tensor
    negative_qa_mask: torch.BoolTensor
    negative_qa_role_ids: torch.LongTensor # question=0, answer=1, pad=-1

    metadata: tuple[AlignmentMetadata, ...]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AlignmentBatch":
        return replace(
            self,
            remark_vectors=self.remark_vectors.to(device, non_blocking=non_blocking),
            remark_mask=self.remark_mask.to(device, non_blocking=non_blocking),
            qa_vectors=self.qa_vectors.to(device, non_blocking=non_blocking),
            qa_mask=self.qa_mask.to(device, non_blocking=non_blocking),
            qa_role_ids=self.qa_role_ids.to(device, non_blocking=non_blocking),
            filing_vectors=self.filing_vectors.to(device, non_blocking=non_blocking),
            filing_mask=self.filing_mask.to(device, non_blocking=non_blocking),
            negative_filing_vectors=self.negative_filing_vectors.to(
                device, non_blocking=non_blocking
            ),
            negative_filing_mask=self.negative_filing_mask.to(
                device, non_blocking=non_blocking
            ),
            negative_remark_vectors=self.negative_remark_vectors.to(
                device, non_blocking=non_blocking
            ),
            negative_remark_mask=self.negative_remark_mask.to(
                device, non_blocking=non_blocking
            ),
            negative_qa_vectors=self.negative_qa_vectors.to(
                device, non_blocking=non_blocking
            ),
            negative_qa_mask=self.negative_qa_mask.to(
                device, non_blocking=non_blocking
            ),
            negative_qa_role_ids=self.negative_qa_role_ids.to(
                device, non_blocking=non_blocking
            ),
        )


@dataclass(frozen=True)
class CrossViewAlignmentOutput:
    """Projected vectors, diagnostic scores, and negative-availability flags."""
    remark_representation: torch.Tensor
    qa_positive_representation: torch.Tensor
    qa_negative_representation: torch.Tensor
    filing_positive_representation: torch.Tensor
    filing_negative_representation: torch.Tensor
    qa_mask: torch.Tensor
    qa_negative_mask: torch.BoolTensor
    filing_mask: torch.Tensor
    filing_negative_mask: torch.BoolTensor

@dataclass(frozen=True)
class ExtractModelLoss():
    total_loss: torch.Tensor
    qa_loss: torch.Tensor
    filing_loss: torch.Tensor
    kl_loss: torch.Tensor

@dataclass(frozen=True)
class ExtractTrainerLoss():
    total_loss: float
    qa_loss: float
    filing_loss: float
    kl_loss: float

@dataclass(frozen=True)
class SelectorOutput():
    remark_representations: torch.Tensor       # [B, N, D]
    remark_selection_weights: torch.Tensor     # [B, N]
    remark_selected_mask: torch.BoolTensor     # [B, N]
    selection_budget: torch.LongTensor
    kl_loss: torch.Tensor

@dataclass(frozen=True)
class ExtractModelOutput():
    model_loss: ExtractModelLoss
    selector_output: SelectorOutput

@dataclass(frozen=True)
class EmbeddingRecordLoader:
    unit_id: str
    text: str
    vector: NDArray[np.float32]

@dataclass(frozen=True)
class EvaluatedCompanyQuarter:
    company_quarter_id: str
    call_id: str
    accession_number: str
    metadata: AlignmentMetadata
    selected_ids: tuple[str, ...]
    selected_weights: tuple[float, ...]
    embeddings: tuple[EmbeddingRecordLoader, ...]


@dataclass(frozen=True)
class ExtractTrainingResult:
    losses: tuple[ExtractTrainerLoss, ...]
    selector_outputs: tuple[SelectorOutput, ...]
    company_quarters: tuple[EvaluatedCompanyQuarter, ...]


@dataclass(frozen=True)
class ExtractTrainerHistory:
    train: list[ExtractTrainerLoss]
    val: list[ExtractTrainerLoss]
