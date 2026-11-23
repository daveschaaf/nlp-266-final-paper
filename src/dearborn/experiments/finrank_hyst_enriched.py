"""HyST filters combined with passage metadata enrichment on the 376 FinRank quantitative questions.

Reuses the enrichment run's dense MPNet rankings over metadata-prefixed passages, drops passages the
filter excludes (order unchanged), then reranks the top 20 survivors with MS MARCO on the enriched text.
  enriched            enrichment only, no filter (reference)
  hyst_enriched       HyST v2 filters (cached extractions; no LLM calls)
  oracle_enriched     the record's own (ticker, year, doc_type), an upper bound
  hyst_raw            HyST v2 filters on the original passages (baseline dense rankings), for attribution
Filters use the pool's first-occurrence metadata, as in finrank_hyst_filter. Filtered-out gold is a miss.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from dearborn.finrank import METADATA, TEXT
from dearborn.hyst_query import QueryExtraction, period_filters
from .finrank_generation_run import ROOT
from .finrank_quantitative_baseline import metrics_for, CROSS
from .finrank_hyst_filter import records_and_golds, oracle

ENRICHED = ROOT/'data/experiments/finrank_metadata_enrichment/v1'
BASELINE = ROOT/'data/experiments/finrank_quantitative_baseline'
HYST = ROOT/'data/experiments/finrank_hyst_filter/v2'
DEFAULT = ROOT/'data/experiments/finrank_hyst_filter/v2_enriched'


def matches(condition, value):
    if not condition: return True
    return value == condition['$eq'] if '$eq' in condition else value in condition['$in']


def keep(filters, metadata):
    ticker, year, doc_type = metadata
    year = int(year) if str(year).isdigit() else None
    return matches(filters.ticker, ticker) and matches(filters.year, year) and matches(filters.doc_type, doc_type)


def rerank(name, rankings, records, texts, model, out):
    """Rerank each ranking's top 20 on enriched text; cached per condition and question."""
    path = out/f'rerank_{name}.jsonl'
    done = {r['id']: r['head'] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    with path.open('a') as handle:
        for r, ranking in zip(records, rankings):
            if r['id'] in done: continue
            head = ranking[:20]
            scores = model.predict([(r['question'], texts[j]) for j in head], show_progress_bar=False) if head else []
            done[r['id']] = [head[k] for k in np.argsort(-np.asarray(scores), kind='stable')]
            handle.write(json.dumps(dict(id=r['id'], head=done[r['id']])) + '\n'); handle.flush()
    return [done[r['id']] + ranking[20:] for r, ranking in zip(records, rankings)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT); parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    records, golds = records_and_golds()
    plan = json.loads((ENRICHED/'plan.json').read_text())
    if plan['baseline']['query_ids'] != [r['id'] for r in records]: raise ValueError('Enrichment run covers different questions')
    with gzip.open(ENRICHED/'enriched_corpus.jsonl.gz', 'rt') as handle:
        texts = [json.loads(line)['text'] for line in handle]
    dense = np.load(ENRICHED/'dense_rankings.npy')
    extractions = {row['id']: row['extraction'] for row in map(json.loads, (HYST/'filters.jsonl').read_text().splitlines())}
    makers = dict(hyst_enriched=lambda r: period_filters(QueryExtraction(**extractions[r['id']])), oracle_enriched=oracle)
    from sentence_transformers import CrossEncoder
    kwargs = dict(local_files_only=True, **({'device': args.device} if args.device else {}))
    model = CrossEncoder(CROSS, **kwargs)
    results = {}
    reranked = np.load(ENRICHED/'reranked_rankings.npy')
    results['enriched_dense'] = metrics_for([list(map(int, x)) for x in dense], records, golds)
    results['enriched_rerank'] = metrics_for([list(map(int, x)) for x in reranked], records, golds)
    for name, make in makers.items():
        filtered = []
        for i, r in enumerate(records):
            f = make(r); filtered.append([int(j) for j in dense[i] if keep(f, METADATA[j])])
        results[f'{name}_dense'] = metrics_for(filtered, records, golds)
        results[f'{name}_rerank'] = metrics_for(rerank(name, filtered, records, texts, model, out), records, golds)
    # Attribution: the same HyST filters and reranker on the original (unenriched) passages.
    base_dense = np.load(BASELINE/'dense_rankings.npy')
    raw = [[int(j) for j in base_dense[i] if keep(makers['hyst_enriched'](r), METADATA[j])] for i, r in enumerate(records)]
    results['hyst_raw_dense'] = metrics_for(raw, records, golds)
    results['hyst_raw_rerank'] = metrics_for(rerank('hyst_raw', raw, records, TEXT, model, out), records, golds)
    summary = {k: v['metrics'] for k, v in results.items()}
    (out/'results.json').write_text(json.dumps(dict(summary=summary, detail=results), indent=1))
    for k, v in summary.items():
        print(f"{k:24}" + ' '.join(f"{m}={v[m]:.1f}" for m in ['recall@1', 'recall@10', 'recall@20', 'mrr', 'ndcg@10']))


if __name__ == '__main__': main()
