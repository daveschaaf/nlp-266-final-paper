from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Company:
    """Company identifier"""
    cik: str # primary key
    name: str
    ticker: str
    sic: int
    industry: str


    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO company (cik, name, ticker, sic, industry)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cik) DO UPDATE SET
                name = excluded.name,
                ticker = excluded.ticker,
                sic = excluded.sic,
                industry = excluded.industry
            """,
            (self.cik, self.name, self.ticker, self.sic, self.industry)
        )
        if commit:
            connection.commit()
        logger.info(f"Saved Company cik={self.cik}, ticker={self.ticker}, name={self.name}")
        return self.cik


create_table_company = """
CREATE TABLE IF NOT EXISTS company (
    cik TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ticker TEXT NOT NULL UNIQUE,
    sic INTEGER,
    industry TEXT
) STRICT;
"""

@dataclass
class CompanyQuarter:
    """A single SEC 10-K or 10-Q quarter"""
    cik: str # foreign key
    company_quarter_id: str # cik-fiscal_year-
    fiscal_year: str # "FY2024"
    fiscal_quarter: int
    fiscal_quarter_start: str
    fiscal_quarter_end: str


    def save(self, connection, *, commit: bool= True):
        connection.execute(
            """
            INSERT INTO company_quarter (
                cik, company_quarter_id, fiscal_year, fiscal_quarter, fiscal_quarter_start, fiscal_quarter_end
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_quarter_id) DO UPDATE SET
                cik=EXCLUDED.cik,
                company_quarter_id=EXCLUDED.company_quarter_id,
                fiscal_year=EXCLUDED.fiscal_year,
                fiscal_quarter=EXCLUDED.fiscal_quarter,
                fiscal_quarter_start=EXCLUDED.fiscal_quarter_start,
                fiscal_quarter_end=EXCLUDED.fiscal_quarter_end
            """,
            (
                self.cik,
                self.company_quarter_id,
                self.fiscal_year,
                self.fiscal_quarter,
                self.fiscal_quarter_start,
                self.fiscal_quarter_end
            ),
        )
        if commit:
            connection.commit()
        logger.info(f"Saved CompanyQuarter company_quarter_id=%s", self.company_quarter_id)
        return self.company_quarter_id
create_table_company_quarter = """
CREATE TABLE IF NOT EXISTS company_quarter (
    company_quarter_id TEXT PRIMARY KEY,
    cik TEXT NOT NULL REFERENCES company(cik),
    fiscal_year TEXT NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    fiscal_quarter_start TEXT NOT NULL,
    fiscal_quarter_end TEXT NOT NULL
) STRICT ;
"""
FISCAL_PERIOD_TO_QUARTER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "FY": 4,
}
def make_company_quarter_id(cik: str, fiscal_year: int | str, fiscal_period: str) -> str:
    """Make a primary key for this quarter"""
    normalized_cik = str(cik).zfill(10)
    normalized_year = int(str(fiscal_year).removeprefix("FY"))
    normalized_quarter = fiscal_period.upper().strip()
    try:
        return f"{normalized_cik}:{normalized_year}:Q{FISCAL_PERIOD_TO_QUARTER[normalized_quarter]}"
    except KeyError:
        raise ValueError(f"Invalid fiscal period {fiscal_period}")

@dataclass
class CompanyFiling:
    """A single SEC 10-K or 10-Q report"""
    cik: str # foreign key
    accession_number: str # primary key
    form: str
    filing_date: str
    filing_year: int
    source_url: str
    fiscal_year: str # "FY2024"
    fiscal_period: str
    fiscal_period_start: str
    fiscal_period_end: str
    parser_version: str = ""
    parse_complete: bool = False
    company_quarter_id: str | None = None

    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO filing (cik, company_quarter_id,accession_number, form, filing_date,
            filing_year, fiscal_year, fiscal_period, fiscal_period_start,
            fiscal_period_end, source_url, parser_version, parse_complete)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.cik, self.company_quarter_id, self.accession_number, self.form, self.filing_date,
            self.filing_year, self.fiscal_year, self.fiscal_period,
            self.fiscal_period_start, self.fiscal_period_end, self.source_url,
            self.parser_version, self.parse_complete)
        )
        if commit:
            connection.commit()
        logger.info(f"Saved CompanyFiling accession_number={self.accession_number}, cik={self.cik}")
        return self.accession_number

create_table_filing = """
CREATE TABLE IF NOT EXISTS filing (
    accession_number TEXT PRIMARY KEY,
    cik TEXT NOT NULL REFERENCES company(cik),
    company_quarter_id TEXT REFERENCES company_quarter(company_quarter_id),
    form TEXT NOT NULL CHECK (form IN ('10-K', '10-Q')),
    filing_date TEXT NOT NULL,
    filing_year INTEGER NOT NULL,
    fiscal_year TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,
    fiscal_period_start TEXT NOT NULL,
    fiscal_period_end TEXT NOT NULL,
    source_url TEXT NOT NULL,
    parser_version TEXT NOT NULL DEFAULT '',
    parse_complete INTEGER NOT NULL DEFAULT 0 CHECK (parse_complete IN (0, 1))
) STRICT;
"""

@dataclass
class FilingChunk:
    """A chunk of text from a company filing"""
    chunk_id: str
    accession_number: str
    part: str              # "PART I" or "PART II"
    item: str              # "ITEM 7"
    item_title: str        # "MANAGEMENT'S DISCUSSION..."
    chunk_index: int
    text: str
    parser_version: str
    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO filing_chunk (accession_number, chunk_id, part, item,
            item_title, chunk_index, text, parser_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.accession_number, self.chunk_id, self.part, self.item,
             self.item_title, self.chunk_index, self.text, self.parser_version)
        )
        if commit:
            connection.commit()
        logger.debug(f"Saved FilingChunk chunk_id={self.chunk_id}")
        return self.chunk_id

create_table_filing_chunk = """
CREATE TABLE IF NOT EXISTS filing_chunk (
    chunk_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL REFERENCES filing(accession_number),
    part TEXT NOT NULL,
    item TEXT NOT NULL,
    item_title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    parser_version TEXT NOT NULL
) STRICT;
"""
