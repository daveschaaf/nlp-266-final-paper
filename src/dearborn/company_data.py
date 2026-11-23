"""Compatibility shim: company_data.py moved to corpus/company.py + corpus/db.py.

Kept so untouched consumers (cross_similarity.py, inspect_prepared_remarks.py,
the ECT-SKIE modules) resolve without edits during the data-pipeline
refactor. Remove this file once those are updated to import the corpus/
paths directly (DATA_PIPELINE_REFACTOR_PROPOSAL.md, step 7).
"""
from .corpus.company import (
    Company,
    CompanyFiling,
    CompanyQuarter,
    FilingChunk,
    FISCAL_PERIOD_TO_QUARTER,
    create_table_company,
    create_table_company_quarter,
    create_table_filing,
    create_table_filing_chunk,
    make_company_quarter_id,
)
from .corpus.db import (
    MAX_CALL_LAG_DAYS,
    backfill_company_quarters,
    migrate_schema,
    setup_db,
    sqlite_connection,
)

__all__ = [
    "Company",
    "CompanyFiling",
    "CompanyQuarter",
    "FilingChunk",
    "FISCAL_PERIOD_TO_QUARTER",
    "MAX_CALL_LAG_DAYS",
    "backfill_company_quarters",
    "create_table_company",
    "create_table_company_quarter",
    "create_table_filing",
    "create_table_filing_chunk",
    "make_company_quarter_id",
    "migrate_schema",
    "setup_db",
    "sqlite_connection",
]

if __name__ == "__main__":
    setup_db()
