# per-view padding and masks for

from .contracts import AlignmentCompanyQuarter, AlignmentBatch, AlignmentMetadata
import torch
import numpy as np

class ExtractCollator(object):
    """Turn variable-length company-quarter bags into one padded PyTorch batch."""

    def pad_vectors(self, sequences: list[np.ndarray]) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Pad [N_i, D] arrays to [B, max(N_i), D] and return a valid-row mask."""

        # Dataset records remain NumPy on CPU. Convert only at the model-facing
        # batching boundary, after vectors from each modality have been grouped.
        tensors = [torch.from_numpy(seq).to(torch.float32) for seq in sequences]
        lengths = torch.tensor([t.shape[0] for t in tensors])
        max_length = int(lengths.max())
        padded = torch.zeros([len(tensors), max_length, tensors[0].shape[1]])
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        # Row j is valid when j is less than that example's unpadded length.
        padding_mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, padding_mask

    def metadata( self, batch: list[AlignmentCompanyQuarter]) -> tuple[AlignmentMetadata, ...]:
        """Keep non-tensor provenance alongside model tensors for later audit."""
        metadata = [
            AlignmentMetadata(
                company_quarter_id=cq.company_quarter_id,
                call_id=cq.call_id,
                accession_number=cq.accession_number,
                split=cq.split,
                remark_ids=cq.remark_ids,
                qa_turn_ids=cq.qa_turn_ids,
                filing_chunk_ids=cq.filing_chunk_ids,
                negative_filing_scope_ids=cq.negative_filing_scope_ids,
                negative_filing_chunk_ids=cq.negative_filing_chunk_ids,
                negative_call_company_quarter_id=cq.negative_call_company_quarter_id,
                negative_remark_ids=cq.negative_remark_ids,
                negative_qa_turn_ids=cq.negative_qa_turn_ids,
            )
            for cq in batch
        ]
        return tuple(metadata)

    def qa_role_ids(self, batch: list[AlignmentCompanyQuarter]) -> torch.LongTensor:
        """Encode Q&A role strings as question=0, answer=1, padding=-1."""
        conversion = {"question": 0, "answer": 1}
        batch_size = len(batch)
        lengths = torch.tensor([len(cq.qa_roles) for cq in batch])
        max_length = int(lengths.max())

        # -1 is distinct from valid role IDs and is ignored by the Q&A mask.
        padded = torch.full((batch_size, max_length),-1, dtype=torch.long)
        
        for i, cq in enumerate(batch):
            roles = torch.tensor(
                [conversion[role] for role in cq.qa_roles],
                dtype=torch.long,
            )
            padded[i, :roles.shape[0]] = roles

        return padded

    def negative_qa_role_ids(self, batch: list[AlignmentCompanyQuarter]) -> torch.LongTensor:
        """Encode roles for the separately sampled negative call's Q&A bag."""
        conversion = {"question": 0, "answer": 1}
        batch_size = len(batch)
        lengths = torch.tensor([len(cq.negative_qa_roles) for cq in batch])
        max_length = int(lengths.max())

        padded = torch.full((batch_size, max_length),-1, dtype=torch.long)
        
        for i, cq in enumerate(batch):
            roles = torch.tensor(
                [conversion[role] for role in cq.negative_qa_roles],
                dtype=torch.long,
            )
            padded[i, :roles.shape[0]] = roles

        return padded


    def __call__(self, batch: list[AlignmentCompanyQuarter]):
        # Check row-level Q&A/vector alignment before padding obscures a
        # data-construction error.
        assert all(
            len(cq.qa_roles) == cq.qa_vectors.shape[0]
            for cq in batch
        )
        assert all(
            len(cq.negative_qa_roles) == cq.negative_qa_vectors.shape[0]
            for cq in batch
        )

        # Pad each modality independently because their bag lengths differ.
        remark_vectors,remark_mask=self.pad_vectors([cq.remark_vectors for cq in batch])
        qa_vectors,qa_mask=self.pad_vectors([cq.qa_vectors for cq in batch])
        filing_vectors, filing_mask=self.pad_vectors([cq.filing_vectors for cq in batch])
        negative_filing_vectors,negative_filing_mask=self.pad_vectors(
            [cq.negative_filing_vectors for cq in batch]
        )
        negative_remark_vectors,negative_remark_mask=self.pad_vectors(
            [cq.negative_remark_vectors for cq in batch]
        )
        negative_qa_vectors, negative_qa_mask = self.pad_vectors(
            [cq.negative_qa_vectors for cq in batch]
        )

        return AlignmentBatch(
            metadata=self.metadata(batch),
            remark_vectors=remark_vectors,
            remark_mask=remark_mask,
            qa_vectors=qa_vectors,
            qa_mask=qa_mask,
            qa_role_ids=self.qa_role_ids(batch),
            filing_vectors=filing_vectors,
            filing_mask=filing_mask,
            negative_filing_vectors=negative_filing_vectors,
            negative_filing_mask=negative_filing_mask,

            negative_remark_vectors=negative_remark_vectors,
            negative_remark_mask=negative_remark_mask,
            negative_qa_vectors=negative_qa_vectors,
            negative_qa_mask=negative_qa_mask,
            negative_qa_role_ids=self.negative_qa_role_ids(batch),
        )
