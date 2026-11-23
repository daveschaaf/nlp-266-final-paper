import logging
from .constants import FINRANK_DIR
logger = logging.getLogger(__name__)

from .upstream.run_baselines import load_records, build_corpus, per_record_metrics, aggregate, tokenize

aggregate = aggregate
per_record_metrics = per_record_metrics
tokenize = tokenize

def gold_records(records: list[dict]) -> list[set]:
    golds = []
    for rec in records:
        gold_passages = set(INDEX[p['text']] for p in rec['passages'] if p['text'].strip() )
        golds.append(gold_passages)
    return golds

RECORDS: list[dict] = load_records(FINRANK_DIR / "FinRank.jsonl")
TEXT, METADATA, INDEX = build_corpus(RECORDS)
QUESTIONS: list[dict] = [rec['question'] for rec in RECORDS]
GOLDS: list[set] = gold_records(RECORDS)
