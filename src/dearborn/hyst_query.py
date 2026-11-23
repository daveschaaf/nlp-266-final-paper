import outlines
from pydantic import BaseModel, ConfigDict
from typing import Literal
from .query_filters import QueryFilters
from .constants import CONTEXT_DIR
from .labeling.models import LabelingModelLoader, LabelingModelName
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

Ticker = Literal['XOM', 'MRK', 'GM', 'TSLA', 'LLY', 'AMGN', 'MRNA', 'JNJ', 'EPD', 'ET', 'WMB', 'F', 'LNG', 'KMI', 'ABBV', 'EOG', 'COP', 'CTRA', 'CVX', 'MPC', 'DVN', 'PFE']


class QueryExtraction(BaseModel):
    """What the question states, nothing inferred. Literal types keep constrained decoding finite."""
    model_config = ConfigDict(extra='forbid')
    ticker: Ticker | None = None
    fiscal_year: Literal[2021, 2022, 2023, 2024, 2025] | None = None
    fiscal_period: Literal['FY', 'Q1', 'Q2', 'Q3', 'Q4'] | None = None
    named_filing_year: Literal[2021, 2022, 2023, 2024, 2025] | None = None
    named_form: Literal['10-K', '10-Q'] | None = None


def period_filters(extraction: QueryExtraction) -> QueryFilters:
    """Map a stated fiscal period to the filing that reports it.

    A filing the question names ("Amgen's 2024 Form 10-K") is used as named;
    FinRank labels such filings by the named year.
    Otherwise, all 22 FinRank issuers end their fiscal year in late December
    (J&J on the 52/53-week Sunday nearest Dec 31), so:
      FY N or Q4 N -> 10-K labelled N or N+1 (Q4 has no 10-Q)
      Q1-Q3 N      -> 10-Q filed in N
      year only    -> either filing year, form unknown
    """
    ticker = {'$eq': extraction.ticker} if extraction.ticker else {}
    if extraction.named_filing_year:
        form = {'$eq': extraction.named_form} if extraction.named_form else {}
        return QueryFilters(ticker=ticker, year={'$eq': extraction.named_filing_year}, doc_type=form)
    year, period = extraction.fiscal_year, extraction.fiscal_period
    if year is None:
        return QueryFilters(ticker=ticker)
    if period in ('FY', 'Q4'):
        # FinRank labels a 10-K by its filing year or its fiscal year, inconsistently, so allow both.
        return QueryFilters(ticker=ticker, year={'$in': [year, year + 1]}, doc_type={'$eq': '10-K'})
    if period in ('Q1', 'Q2', 'Q3'):
        return QueryFilters(ticker=ticker, year={'$eq': year}, doc_type={'$eq': '10-Q'})
    return QueryFilters(ticker=ticker, year={'$in': [year, year + 1]})


@dataclass
class HySTResult:
    query: str
    extraction: QueryExtraction
    filters: QueryFilters


class HySTQuery:
    """
    HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data
    https://arxiv.org/abs/2508.18048

    The LLM extracts the stated issuer and fiscal period; period_filters maps them to filing metadata.
    The model is one of the vetted local MLX models; Outlines constrains output to QueryExtraction.
    """
    def __init__(self, model_name: LabelingModelName):
        self.prompt = (CONTEXT_DIR/"hyst_query.md").read_text()
        logger.info(f"Loading model {model_name}")
        loaded = LabelingModelLoader(local_files_only=True).load(model_name)
        self.tokenizer = loaded.input_processor
        model = outlines.from_mlxlm(loaded.model, self.tokenizer)
        self.generator = outlines.Generator(model, QueryExtraction)
        logger.info(f"HySTQuery initialized with {loaded.model_id}")

    def query_filters(self, question: str) -> HySTResult:
        # Nemotron ignores enable_thinking; "/no_think" in the system turn closes the think block.
        messages = [{"role": "system", "content": "/no_think"},
                    {"role": "user", "content": f"{self.prompt}\n\nQuestion:\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            raw = self.generator(prompt, max_tokens=100)
            logger.info(f"HyST Generator output: {raw}")
            extraction = QueryExtraction.model_validate_json(raw)
        except ValueError as e:
            logger.warning(f"HyST extraction failed for {question}: {e}")
            extraction = QueryExtraction()
        return HySTResult(query=question, extraction=extraction, filters=period_filters(extraction))
