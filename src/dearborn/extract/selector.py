"""Prepared-remarks information-bottleneck selector.

Architecture adapted from Huang et al. (2024), "Explainable Earnings Call
Representation Learning," AAAI-24 Student Abstract:
https://ojs.aaai.org/index.php/AAAI/article/view/30454

This project keeps the selector's two-stage, length-bounded extraction
mechanism but changes its supervision: selected prepared remarks must align
with management Q&A answers and the linked filing, rather than predict market
volatility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .contracts import SelectorOutput
import math

class ImportanceScorer(nn.Module):
    """Score how useful each slot-selected remark is for cross-view alignment."""

    def __init__(self, dimension: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2*dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class PreparedRemarksSelector(nn.Module):
    """Compress a bag of prepared-remark embeddings into one call representation.

    Each learned container slot first chooses one valid remark.  A second,
    learned score then selects a length-bounded subset of those slot choices.
    """

    def __init__(
        self,
        dimension: int,
        slots: int = 200,
        temperature: float = 0.1,
        tau: float = 0.1,
        compression_rate: float = 0.15,
        top_k_lower_bound: int = 5,
    ):
        super().__init__()
        self.slots = slots
        self.temperature = temperature
        self.tau = tau
        self.compression_rate = compression_rate
        self.top_k_lower_bound = top_k_lower_bound
        # K trainable query vectors.  They learn distinct kinds of useful
        # prepared-remark content during cross-view alignment training.
        self.container = nn.Parameter(
            torch.randn(slots, dimension)
        )
        self.importance_scorer = ImportanceScorer(dimension)

    def forward(
        self,
        remark_vectors: torch.Tensor,
        remark_mask: torch.BoolTensor,
    ) -> SelectorOutput:
        ##########
        ##########

        # Count real remarks, excluding zero-padding added by the collator.
        remark_count = remark_mask.sum(dim=1)

        # Select roughly 15% of the available remarks, but never fewer than
        # the lower bound or more than the number of learned slots.
        selection_budget = (
            torch.ceil(remark_count * self.compression_rate).to(torch.long)
            + self.top_k_lower_bound
        ).clamp(max=self.slots)

        # use normalized representations to avoid growth of dot products
        normalized_container = F.normalize(self.container, dim=-1)
        normalized_remark_vectors = F.normalize(remark_vectors, dim=-1)

        # Share the same K learned slots across every example in the batch:
        # [K, D] -> [B, K, D].  expand does not copy the learned parameters.
        container = self.container.unsqueeze(0).expand(
            normalized_remark_vectors.shape[0], -1, -1
        )

        # Compare every slot to every remark:
        # [B, K, D] @ [B, D, N] -> [B, K, N].
        similarity_score = normalized_container @ normalized_remark_vectors.permute(0, 2, 1)
        # A padded remark must never be chosen by any slot.
        similarity_score = similarity_score.masked_fill(
            ~remark_mask.unsqueeze(1),
            -1e9 # avoid -inf gradient errors
        )

        # Each slot chooses exactly one remark.  The forward pass is one-hot;
        # the backward pass uses the soft probabilities (straight-through).
        if self.training:
            selected_position_weights = self.gumbel_softmax_topK(
                logits=similarity_score,
                top_k=1,
                tau=self.tau,
                hard=True,
                dim=-1,
            )
        else:
            selected_remark_positions = similarity_score.argmax(dim=-1)
            selected_position_weights = F.one_hot(
                selected_remark_positions,
                num_classes=remark_vectors.shape[1],
            ).to(remark_vectors.dtype)

        # Convert slot-to-remark positions into K selected remark vectors:
        # [B, K, N] @ [B, N, D] -> [B, K, D].
        selected_remarks = selected_position_weights @ remark_vectors

        # Score each (slot, selected remark) pair.  These are the logits used
        # to decide which slot selections survive the information bottleneck.
        importance_logits = self.importance_scorer(
            torch.cat([container, selected_remarks], dim=-1)
        ).squeeze(-1)


        # KL loss
        kl_loss = self.kl_loss(importance_logits)

        # The selection budget differs by example, so choose the top M slots
        # one batch row at a time.  The result is K-hot in the forward pass
        # while retaining soft gradients to ImportanceScorer.
        if self.training:
            selected_slot_mask = torch.stack([
                self.gumbel_softmax_topK(
                    logits=logits,
                    top_k=int(budget.item()),
                    tau=self.tau,
                    hard=True,
                    dim=-1,
                )
                for logits, budget, in zip(importance_logits, selection_budget)
            ])
        else:
            selected_slot_mask = torch.stack([
                torch.zeros_like(logits).scatter_(
                    0,
                    logits.topk(int(budget.item())).indices,
                    1.0
                )
                for logits, budget in zip(importance_logits, selection_budget)
            ])

        # Divide K-hot selections by M to form weights that average the M
        # retained selected remarks instead of summing their magnitudes.
        slot_weights = selected_slot_mask / selection_budget.unsqueeze(1)

        # Sum the total weight assigned to each remark across all slots
        remark_selection_weights = (
            selected_position_weights * slot_weights.unsqueeze(-1)
        ).sum(dim=1)

        # Select remarks with positive weight
        remark_selected_mask = remark_selection_weights > 0

        return SelectorOutput(
            remark_representations=remark_vectors,
            remark_selection_weights=remark_selection_weights,
            remark_selected_mask=remark_selected_mask,
            selection_budget=selection_budget.long(),
            kl_loss=kl_loss,
        )
    def kl_loss(self, logits: torch.Tensor) -> torch.Tensor:
        logits_log_p = F.log_softmax(logits, dim=-1)
        logits_p = torch.exp(logits_log_p)
        logits_kl = (
            logits_p * (
                logits_log_p + math.log(self.slots)
            )
        ).sum(dim=-1)
        return logits_kl.mean()

    def gumbel_noise_like(self,logits: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(logits.dtype).eps
        # Keep the uniform draw strictly inside (0, 1), so nested logarithms
        # cannot evaluate log(0).
        uniform = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
        return -torch.log(-torch.log(uniform))

    def gumbel_softmax_topK(self, logits, top_k, tau:float = 1.0, hard:bool = False, dim: int = -1):
        """Select top_k entries with a straight-through Gumbel-softmax path."""

        # Gumbel noise samples a discrete choice while the softmax supplies a
        # differentiable approximation for backpropagation.
        gumbel_noise = self.gumbel_noise_like(logits)
        soft_selection = ((logits + gumbel_noise) / tau).softmax(dim)

        if hard:
            _, selected_indices = soft_selection.topk(top_k, dim=dim)

            # K-hot forward values; the final expression preserves the soft
            # gradient so the importance scorer can learn from the loss.
            hard_selection = torch.zeros_like(logits).scatter_(
                dim,
                selected_indices,
                1.0,
            )

            return hard_selection - soft_selection.detach() + soft_selection

        return soft_selection
