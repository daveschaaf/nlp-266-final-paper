# tri-view selector and alignment model

import torch
import torch.nn as nn
from .selector import PreparedRemarksSelector
from .losses import PairedInfoNCELoss
from .contracts import AlignmentBatch, ExtractModelLoss, ExtractModelOutput
from .alignment import CrossViewAlignment

class ExtractModel(nn.Module):
    def __init__(self,
         dimension: int=768,
         slots: int=200,
         temperature: float=0.1,
         tau: float=0.1,
         beta: float = 0.0,
         compression_rate: float = 0.15,
         top_k_lower_bound: int=5,
         use_qa_alignment: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.use_qa_alignment = use_qa_alignment
        self.selector = PreparedRemarksSelector(
            dimension=dimension,
            slots=slots,
            temperature=temperature,
            tau=tau,
            compression_rate=compression_rate,
            top_k_lower_bound=top_k_lower_bound,
        )
        self.alignment = CrossViewAlignment(dimension=dimension)
        self.qa_loss_fn = PairedInfoNCELoss(temperature=temperature)
        self.filing_loss_fn = PairedInfoNCELoss(temperature=temperature)

    def forward(self, x: AlignmentBatch):
        selector_output = self.selector(x.remark_vectors, x.remark_mask)
        alignment_output = self.alignment(
            remark_representations=selector_output.remark_representations,
            qa_vectors=x.qa_vectors,
            qa_mask=x.qa_mask,
            qa_role_ids=x.qa_role_ids,
            negative_qa_vectors=x.negative_qa_vectors,
            negative_qa_mask=x.negative_qa_mask,
            negative_qa_role_ids=x.negative_qa_role_ids,
            filing_vectors=x.filing_vectors,
            filing_mask=x.filing_mask,
            negative_filing_vectors=x.negative_filing_vectors,
            negative_filing_mask=x.negative_filing_mask,
        )
        if self.use_qa_alignment:
            qa_loss = self.qa_loss_fn(
                query=alignment_output.remark_representation,
                query_weights=selector_output.remark_selection_weights,
                positive_key=alignment_output.qa_positive_representation,
                positive_mask=alignment_output.qa_mask,
                negative_key=alignment_output.qa_negative_representation,
                negative_mask=alignment_output.qa_negative_mask,
            )
        else:
            qa_loss = alignment_output.remark_representation.new_zeros(())
        filing_loss = self.filing_loss_fn(
            query=alignment_output.remark_representation,
            query_weights=selector_output.remark_selection_weights,
            positive_key=alignment_output.filing_positive_representation,
            positive_mask=alignment_output.filing_mask,
            negative_key=alignment_output.filing_negative_representation,
            negative_mask=alignment_output.filing_negative_mask,
        )
        total_loss = qa_loss + filing_loss + self.beta * (selector_output.kl_loss)
        loss_output = ExtractModelLoss(
            total_loss=total_loss,
            qa_loss=qa_loss,
            filing_loss=filing_loss,
            kl_loss=selector_output.kl_loss,
        )
        return ExtractModelOutput(
            model_loss=loss_output, 
            selector_output=selector_output, 
        )


