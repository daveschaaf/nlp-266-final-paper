"""Split-safe negative sampling for filing chunks and full earnings calls."""

from .contracts import AlignmentCompanyQuarter

def sample_negative_filing_ids(
    anchor: AlignmentCompanyQuarter,
    scopes: list[AlignmentCompanyQuarter],
    rng,
) -> tuple[tuple[str, str], ...]:
    """Sample up to one anchor-filing-bag's worth of chunks from other scopes.

    Sampling is scope-first, then chunk-without-replacement within that scope,
    so an unusually long filing cannot supply repeated chunks.
    """
    available = {
        cq.company_quarter_id: list(cq.filing_chunk_ids)
        for cq in scopes
        if (
            cq.company_quarter_id != anchor.company_quarter_id
            and cq.split == anchor.split
        )
    }
    # A 1:1 cap matches the anchor's own filing-bag size when possible.
    sample_size = min(
        len(anchor.filing_chunk_ids),
        sum(len(ids) for ids in available.values()),
    )
    negatives = []

    while len(negatives) < sample_size:
        # Exclude exhausted source filings after all of their chunks are drawn.
        eligible_scope_ids = [
            scope_id for scope_id, ids in available.items() if ids
        ]
        cq_id = rng.choice(eligible_scope_ids)
        filing_ids = available[cq_id]
        negatives.append(
            (
                cq_id,
                filing_ids.pop(rng.integers(len(filing_ids)))
            )
        )

    return tuple(negatives)

def sample_negative_call_scope(
    anchor: AlignmentCompanyQuarter,
    scopes: list[AlignmentCompanyQuarter],
    rng
) -> AlignmentCompanyQuarter | None:
    """Return one other full call in the split, or None if no call is available."""
    available = [
        scope
        for scope in scopes
        if scope.company_quarter_id != anchor.company_quarter_id
        and scope.split == anchor.split
    ]
    if not available:
        return None
    return available[int(rng.integers(len(available)))]
