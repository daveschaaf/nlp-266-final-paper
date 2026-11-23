import torch
import torch.nn as nn
import torch.nn.functional as F

class PairedInfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
    def forward(
        self,
        query,          # [B, N, D]: one projected representation per remark
        query_weights,  # [B, N]: selector weight assigned to each remark
        positive_key,   # [B, P, D]: positive answer turns or filing chunks
        positive_mask,  # [B, P]: True for real positive evidence
        negative_key,   # [B, Q, D]: negative answer turns or filing chunks
        negative_mask,  # [B, Q]: True for real negative evidence
    ) -> torch.Tensor:
        """Return one selector-weighted contrastive loss scalar.

        B is batch size, N is remark count, P and Q are positive and
        negative evidence counts, and D is embedding dimension.
        """
        assert positive_mask.any(dim=1).all()
        assert negative_mask.any(dim=1).all()
        # Shapes remain [B, N, D], [B, P, D], and [B, Q, D].
        # Only each final D-dimensional vector is normalized.
        query = F.normalize(query, dim=-1)
        positive_key = F.normalize(positive_key, dim=-1)
        negative_key = F.normalize(negative_key, dim=-1)

        # [B, N, D] @ [B, D, P] -> [B, N, P]
        positive_similarity = query @ positive_key.transpose(1,2)
        # [B, N, D] @ [B, D, Q] -> [B, N, Q]
        negative_similarity = query @ negative_key.transpose(1,2)

        # [B, P] -> [B, 1, P], broadcast over N; result [B, N, P].
        positive_scores = positive_similarity.masked_fill(
            ~positive_mask.unsqueeze(1),
            -1e9
        )
        # [B, Q] -> [B, 1, Q], broadcast over N; result [B, N, Q].
        negative_scores = negative_similarity.masked_fill(
            ~negative_mask.unsqueeze(1),
            -1e9
        )
        # Reduce the evidence dimensions P and Q: [B, N, P/Q] -> [B, N].
        positive_support = positive_scores.max(dim=-1).values
        negative_support = negative_scores.max(dim=-1).values

        # Two candidates per remark, positive at index 0: [B, N, 2].
        logits = torch.stack(
            [positive_support, negative_support],
            dim=-1
        )

        # One class target per remark: [B, N], always positive class 0.
        targets = torch.zeros(
            logits.shape[:2],
            dtype=torch.long,
            device=logits.device,
        )

        loss_per_remark = self.loss_fn(
            (logits / self.temperature).reshape(-1,2),
            targets.reshape(-1),
        ).reshape(query.shape[0], query.shape[1])
        # loss_per_remark [B, N] * query_weights [B, N] -> [B, N].
        # Sum remarks N -> one loss per call [B].
        loss_per_call = (loss_per_remark * query_weights).sum(dim=1)
        # Mean calls B -> scalar.
        loss = loss_per_call.mean()
        return loss
