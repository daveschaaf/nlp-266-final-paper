"""Build split-safe, unpadded company-quarter alignment examples.

This is the boundary between persisted corpus/Qdrant records and PyTorch's
Dataset interface. Negatives are built only from examples in the same split to
prevent evaluation leakage.
"""
from torch.utils.data import Dataset
from dataclasses import dataclass, replace
import torch
import numpy as np
import logging
from dearborn.logging_config import setup_logging
from torch.utils.data.dataset import random_split
from dearborn.corpus.extract_repository import fetch_extract_source, fetch_valid_extract_scopes
from .negative import sample_negative_filing_ids, sample_negative_call_scope
from .embedding_store import QDRANT_COLLECTION, ExtractEmbeddingStore
from dearborn.company_data import sqlite_connection
from dearborn.finrank_policy import fetch_finrank_ciks
from .contracts import AlignmentCompanyQuarter
logger = logging.getLogger(__name__)
setup_logging()


class AlignmentDataset(Dataset):
    """Thin indexable wrapper around immutable AlignmentCompanyQuarter records."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_val_test_split_by_company_quarter(scopes, lengths, generator=None):
    """Randomly split whole company-quarter scopes, never individual vectors."""
    train_scopes, val_scopes, test_scopes = random_split(
        scopes,
        lengths=lengths,
        generator=generator,
    )
    return train_scopes, val_scopes, test_scopes

def build_alignment_company_quarter(client, connection, scope, split):
    """Fetch one scope's positive vectors and initialize empty negative bags."""
    source = fetch_extract_source(connection, scope.company_quarter_id)
    earnings_call = source.earnings_call
    filing_chunks = source.filing_chunks
    # Only deterministic-parser units eligible for synthesis can be selected.
    remarks = [
        unit for unit in earnings_call.prepared_units
        if unit.eligible_for_synthesis
    ]
    remark_ids = [u.unit_id for u in remarks]
    qa_turn_ids=[u.turn_id for u in earnings_call.qa_turns]
    filing_chunk_ids=[u.chunk_id for u in filing_chunks]
    # fetch() restores requested ID order, so vectors align with IDs above.
    filing_vectors=np.stack([
        record.vector
        for record in client.fetch(filing_chunk_ids)
    ])
    remark_vectors=np.stack([
        record.vector
        for record in client.fetch(remark_ids)
    ])
    qa_vectors=np.stack([
        record.vector
        for record in client.fetch(qa_turn_ids)
    ])

    def empty_vectors(vectors):
        # Retain D for safe padding even when this scope has no negatives yet.
        return np.empty((0, vectors.shape[1]), dtype=np.float32)

    return AlignmentCompanyQuarter(
        company_quarter_id=scope.company_quarter_id,
        call_id=scope.call_id,
        accession_number= scope.accession_number,
        split=split,
        remark_ids=tuple(remark_ids),
        qa_turn_ids=tuple(qa_turn_ids),
        filing_chunk_ids=tuple(filing_chunk_ids),
        qa_roles=tuple([u.role for u in earnings_call.qa_turns]),
        remark_vectors=remark_vectors,
        qa_vectors=qa_vectors,
        filing_vectors=filing_vectors,
        negative_filing_scope_ids=(),
        negative_filing_chunk_ids=(),
        negative_filing_vectors=empty_vectors(filing_vectors),
        negative_call_company_quarter_id=None,
        negative_remark_ids=(),
        negative_remark_vectors=empty_vectors(remark_vectors),
        negative_qa_turn_ids=(),
        negative_qa_roles=(),
        negative_qa_vectors=empty_vectors(qa_vectors),
    )

def build_extract_datasets(
        scope_ids: tuple[str, ...],
        lengths: tuple[float, float, float]=(1.0, 0.0, 0.0),
        generator: torch.Generator | None = None,
    ):
    """Build train/validation/test datasets and attach split-safe negatives."""
    client = ExtractEmbeddingStore(QDRANT_COLLECTION)
    connection = sqlite_connection()
    try:
        available = {
            scope.company_quarter_id: scope
            for scope in fetch_valid_extract_scopes(connection)
        }
        finrank_ciks = fetch_finrank_ciks(connection)
        requested_scopes = [available[scope_id] for scope_id in scope_ids]
        scopes = [scope for scope in requested_scopes if scope.cik not in finrank_ciks]
        excluded = len(requested_scopes) - len(scopes)
        if excluded:
            logger.info("Excluded %d FinRank benchmark scope(s) from extract datasets.", excluded)

        train_scopes, val_scopes, test_scopes = train_val_test_split_by_company_quarter(
            scopes,
            lengths=lengths,
            generator=generator,
        )
        def build(scopes, split, rng):
            data = []
            # Build all positive examples first. This complete split is the
            # only pool from which its negatives may be sampled.
            alignments = [
                build_alignment_company_quarter(client, connection, scope, split)
                for scope in scopes
            ]

            for cq in alignments:
                # Filing negatives are sampled chunks; Q&A negatives retain a
                # complete other call and therefore its answer-bag structure.
                negative_filings = sample_negative_filing_ids(cq, alignments, rng)
                negative_scope_ids = tuple([scope_id for scope_id, _ in negative_filings])
                negative_chunk_ids = tuple([chunk_id for _, chunk_id in negative_filings])
                if negative_chunk_ids:
                    records = client.fetch(list(negative_chunk_ids))
                    negative_filing_vectors = np.stack([record.vector for record in records])
                else:
                    negative_filing_vectors = np.empty((0, cq.filing_vectors.shape[1]), dtype=np.float32)
                negative_call_scope = sample_negative_call_scope(cq, alignments, rng)
                
                # replace() returns a new immutable record; it never mutates
                # the positive alignment shared by other sampling iterations.
                negatives = {
                    "negative_filing_scope_ids": negative_scope_ids,
                    "negative_filing_chunk_ids": negative_chunk_ids,
                    "negative_filing_vectors": negative_filing_vectors,
                }

                if negative_call_scope is not None:
                    negatives.update({
                        "negative_call_company_quarter_id":
                            negative_call_scope.company_quarter_id,
                        "negative_remark_ids": negative_call_scope.remark_ids,
                        "negative_remark_vectors": negative_call_scope.remark_vectors,
                        "negative_qa_turn_ids": negative_call_scope.qa_turn_ids,
                        "negative_qa_roles": negative_call_scope.qa_roles,
                        "negative_qa_vectors": negative_call_scope.qa_vectors,
                    })

                data.append(replace(cq, **negatives))

            return AlignmentDataset(data)

        rng = np.random.default_rng(seed=42)
        train_data = build(train_scopes, 'train', rng)
        val_data = build(val_scopes, 'val', rng)
        test_data =build(test_scopes, 'test', rng)

        return train_data, val_data, test_data


    finally:
        connection.close()
