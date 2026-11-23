"""Cross-view supervision for the prepared-remarks selector.

This adapts structure-aware contrastive learning from Huang et al. (2024),
"Explainable Earnings Call Representation Learning," AAAI-24 Student Abstract:
https://ojs.aaai.org/index.php/AAAI/article/view/30454

The original method aligns selected presentation content with transcript Q&A.
Here, fused prepared-remarks/Q&A evidence additionally aligns with the linked
company-quarter filing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .contracts import CrossViewAlignmentOutput

ANSWER_ID = 1

class ProjectionEncoder(nn.Module):
    """Map one modality's frozen embedding into the shared alignment space."""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.ReLU(),
            nn.Linear(output_dimension, output_dimension),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class CrossViewAlignment(nn.Module):
    """Align selected prepared remarks with management answers and filings.

    The Q&A objective uses the selected remark representation.  The filing
    objective uses a learned fusion of that representation and its own Q&A.
    """

    def __init__(self, dimension: int):
        super().__init__()
        # Modality-specific projections let frozen embedding vectors adapt to
        # a shared task space.  The call encoder fuses remarks + Q&A.
        self.remark_encoder = ProjectionEncoder(dimension, dimension)
        self.filing_encoder = ProjectionEncoder(dimension, dimension)
        self.qa_encoder = ProjectionEncoder(dimension, dimension)
        
    def forward(
        self,
        remark_representations: torch.Tensor,       # [B, N, D]
        qa_vectors,
        qa_mask,
        qa_role_ids,
        negative_qa_vectors,
        negative_qa_mask,
        negative_qa_role_ids,
        filing_vectors,
        filing_mask,
        negative_filing_vectors,
        negative_filing_mask,
    ) -> CrossViewAlignmentOutput:
        # Retain management answers only.  Questions and padding do not
        # contribute to the Q&A representation used by the alignment loss.
        answer_mask = qa_mask & qa_role_ids.eq(ANSWER_ID)
        negative_answer_mask = negative_qa_mask & negative_qa_role_ids.eq(ANSWER_ID)

        # Project positive and negative Q&A bags with the same encoder so
        # their similarity scores are directly comparable.
        qa_positive_representation = self.qa_encoder(qa_vectors)
        qa_negative_representation = self.qa_encoder(negative_qa_vectors)
        remark_representation = self.remark_encoder(remark_representations)

        # Positive and negative filing bags likewise share one projection.
        filing_positive_representation = self.filing_encoder(filing_vectors)
        filing_negative_representation = self.filing_encoder(negative_filing_vectors)

        return CrossViewAlignmentOutput(
            remark_representation=remark_representation,
            qa_positive_representation=qa_positive_representation,
            qa_negative_representation=qa_negative_representation,
            filing_positive_representation=filing_positive_representation,
            filing_negative_representation=filing_negative_representation,
            qa_mask=answer_mask,
            qa_negative_mask=negative_answer_mask,
            filing_mask=filing_mask,
            filing_negative_mask=negative_filing_mask,
        )

