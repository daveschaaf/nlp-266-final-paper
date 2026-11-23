"""Re-score saved rankings for Quantitative AND Financials; no inference or training."""
import argparse
import json
from pathlib import Path
import numpy as np
from dearborn.finrank import RECORDS,TEXT,INDEX
from .finrank_generation_run import ROOT
from .finrank_quantitative_baseline import digest,metrics_for
from .finrank_source_recovery import source_labels,assess
from .finrank_eda import table
from dearborn.training.finrank_metadata_training import sha
from dearborn.training.finrank_pilot import atomic_json

DEFAULT=ROOT/'data/experiments/finrank_financials_quantitative'


def financial_subset(rankings,records,corpus_count):
    """Subset question rows only; retain every candidate and its original order."""
    if rankings.shape!=(len(records),corpus_count):raise ValueError('Ranking shape differs from frozen benchmark')
    positions=[i for i,r in enumerate(records) if r['topic']=='Financials' and r['reasoning_type']=='Quantitative']
    if not positions:raise ValueError('No eligible questions')
    selected=rankings[positions]
    if not np.all(np.sort(selected,axis=1)==np.arange(corpus_count)):raise ValueError('Rankings are not full corpus permutations')
    return selected,[records[i] for i in positions]


def load_saved_rankings():
    experiments=ROOT/'data/experiments';base=experiments/'finrank_quantitative_baseline'
    manifest=json.loads((base/'run_manifest.json').read_text())
    records=[r for r in RECORDS if r['reasoning_type']=='Quantitative']
    if manifest['query_ids']!=[r['id'] for r in records] or manifest['query_sha256']!=digest([r['question'] for r in records]) or manifest['corpus_sha256']!=digest(TEXT):raise ValueError('Frozen benchmark changed')
    sources={
        'TF-IDF':base/'tfidf_rankings.npy',
        'BM25':base/'bm25_rankings.npy',
        'Original MPNet':base/'dense_rankings.npy',
        'Metadata only':experiments/'finrank_metadata_enrichment/v1/reranked_rankings.npy',
        'Question training only':experiments/'finrank_training/hard-negatives-ablation-v1/evaluation/training_only/rankings.npy',
        'Metadata + question training':experiments/'finrank_training/hard-negatives-ablation-v1/evaluation/metadata_training/rankings.npy',
        'Metadata + remark training':experiments/'finrank_training/remark561-metadata-lr2e7-v1/evaluation/remark_training/rankings.npy',
    }
    dense=np.load(sources['Original MPNet'])
    trace=base/'cross_encoder.jsonl'
    completed={r['id']:r for r in map(json.loads,trace.read_text().splitlines()) if r['status']=='completed'}
    combined=np.stack([np.concatenate([completed[r['id']]['head'],dense[i][20:]]) for i,r in enumerate(records)])
    rankings={'TF-IDF':np.load(sources['TF-IDF']),'BM25':np.load(sources['BM25']),'Original MPNet':dense,'Original MPNet + MS MARCO':combined}
    rankings.update({name:np.load(path) for name,path in sources.items() if name not in rankings})
    # Verify training/evaluation sources inherit the same frozen question/corpus protocol.
    protocols=[experiments/'finrank_metadata_enrichment/v1/plan.json',experiments/'finrank_training/hard-negatives-ablation-v1/evaluation/plan.json',experiments/'finrank_training/remark561-metadata-lr2e7-v1/evaluation/plan.json']
    for path in protocols:
        protocol=json.loads(path.read_text());inherited=protocol.get('benchmark',protocol)
        if inherited['baseline']!=manifest:raise ValueError(f'Incompatible source benchmark: {path}')
    return records,rankings,manifest,sources,trace,protocols


def run(out):
    records,rankings,manifest,sources,trace,protocols=load_saved_rankings()
    selected=[r for r in records if r['topic']=='Financials']
    plan=dict(filter=dict(reasoning_type='Quantitative',topic='Financials'),query_ids=[r['id'] for r in selected],query_count=len(selected),corpus_count=len(TEXT),baseline_manifest=manifest,source_sha256={str(p.resolve()):sha(p) for p in [*sources.values(),trace,*protocols]},note='Exploratory existing-label subgroup after full-benchmark inspection. No selection based on model scores; unchanged full corpus and rankings. Training regimes and sample sizes differ; see original experiment plans.')
    out.mkdir(parents=True,exist_ok=True);path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Subgroup inputs changed; use a new output directory')
    atomic_json(path,plan)
    labels=source_labels(RECORDS,INDEX);models={}
    for name,ranking in rankings.items():
        subset,rows=financial_subset(ranking,records,len(TEXT))
        golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in rows]
        metrics=metrics_for(subset,rows,golds);diagnostics=assess(subset,rows,golds,labels)
        models[name]=dict(**metrics,source_metrics=diagnostics['source_metrics'])
    result=dict(plan=plan,models=models)
    atomic_json(out/'results.json',result)
    (out/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    headers=['Configuration','R@1','R@5','R@10','R@20','MRR','nDCG@10'];rows=[]
    for name,m in models.items():rows.append([name,*[f'{m["metrics"][key]:.2f}' for key in ['recall@1','recall@5','recall@10','recall@20','mrr','ndcg@10']]])
    comparison=table(headers,rows)
    note='54 questions selected only by existing Financials + Quantitative labels; unchanged 5,230-passage search corpus. All metrics use the original macro gold-passage protocol, scaled to 0–100. No model inference, retraining, question rewriting, or candidate filtering. 20 questions have one gold passage and 34 have multiple. This is an exploratory subgroup analysis; it does not replace the full 376-question result or establish statistical significance. Training sample sizes/rates differ between question and remark training; this is not a controlled query-form ablation.'
    (out/'REPORT.md').write_text('# FinRank: Quantitative and Financials\n\n'+comparison+'\n'+note+'\n')
    print(comparison);print(f'DONE: {len(selected)} questions re-scored; results saved to {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();run(args.output_dir.resolve())


if __name__=='__main__':main()
