"""Train original remarks directly against their saved passage matches, without synthesis."""
import argparse
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from . import finrank_ablation as engine
from .finrank_hard_negatives import read_pool,sample_records,validation_pool
from .finrank_metadata_training import sha
from .finrank_reranker import issuer_split
from .finrank_pilot import atomic_json
from dearborn.finrank_policy import assert_not_finrank_tickers
from dearborn.experiments.finrank_generation_run import ROOT

SOURCE=engine.SOURCE
DEFAULT=SOURCE.parent/'remark561-metadata-v1'


def recover(source,connection,seed=42):
    """Recover original input remarks; generated summaries/questions never become queries."""
    connection.row_factory=sqlite3.Row
    accepted_path=source/'accepted.jsonl'
    sources=sorted({r['source_run'] for r in map(json.loads,accepted_path.read_text().splitlines())})
    hashes={str(accepted_path.resolve()):sha(accepted_path)};pairs={}
    calls={r['call_id']:dict(r) for r in connection.execute('SELECT * FROM earnings_call')}
    filings={r['accession_number']:dict(r) for r in connection.execute('SELECT * FROM filing')}
    for directory in map(Path,sources):
        plan_path=directory/'plan.json';summary_path=directory/'summary.json'
        hashes[str(plan_path.resolve())]=sha(plan_path);hashes[str(summary_path.resolve())]=sha(summary_path)
        cases={r['id']:r for r in json.loads(plan_path.read_text())['cases']}
        for result in json.loads(summary_path.read_text())['results']:
            if result['status']!='accepted':continue
            case=cases[result['id']];values=case['values']
            ticker=values['ticker'].upper();assert_not_finrank_tickers([ticker])
            call=calls.get(values['call_period']);filing=filings.get(case['passage_id'].split(':',1)[0])
            if not call or not filing:raise ValueError(f'Missing call or filing for {result["id"]}')
            if not call['company_quarter_id'] or call['company_quarter_id']!=filing['company_quarter_id']:
                raise ValueError(f'Missing or unequal company-quarter links for {result["id"]}')
            fiscal_year=int(re.sub(r'\D','',str(filing['fiscal_year'])))
            fiscal_quarter=4 if filing['form']=='10-K' and filing['fiscal_period']=='FY' else int(re.sub(r'\D','',str(filing['fiscal_period'])))
            if int(call['cik'])!=int(filing['cik']) or call['ticker'].upper()!=ticker or (call['year'],call['quarter'])!=(fiscal_year,fiscal_quarter):
                raise ValueError(f'Issuer or fiscal-period mismatch for {result["id"]}')
            # This is the original remark input, not claim_statement, summary, or question.
            remark=values['remark_text']
            if not remark.strip():raise ValueError('Empty source remark')
            key=(ticker,call['call_id'],' '.join(remark.split()),case['passage_id'])
            group=hashlib.sha256(json.dumps(key[:3],ensure_ascii=False).encode()).hexdigest()
            provenance=dict(source_run=str(directory),case_id=case['id'],call_id=call['call_id'],company_quarter_id=call['company_quarter_id'],accession_number=filing['accession_number'],acceptance='Existing pipeline acceptance; not a new full-remark verification')
            if key in pairs:
                pairs[key]['provenance'].append(provenance);continue
            identifier=hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()
            pairs[key]=dict(id=identifier,question=remark,query_group=group,ticker=ticker,split=issuer_split(ticker,seed),positive_text=values['passage_text'],passage_id=case['passage_id'],provenance=[provenance])
    return sorted(pairs.values(),key=lambda r:r['id']),hashes


def prepare(source,out,database,seed=42,learning_rate=2e-6):
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError("Learning rate must be finite and positive")
    out.mkdir(parents=True,exist_ok=True);path=out/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text());request=plan['request']
        if request['source']!=str(source.resolve()) or request['database']!=str(database.resolve()) or request['seed']!=seed or request['rates']!=[learning_rate]:raise ValueError('Frozen configuration differs; use a separate output directory for a new learning rate')
        for filename,value in plan['files'].items():
            if sha(out/filename)!=value:raise ValueError(f'Frozen artifact changed: {filename}')
        for filename,value in plan['source_hashes'].items():
            if sha(Path(filename))!=value:raise ValueError(f'Source artifact changed: {filename}')
        return plan
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        records,hashes=recover(source,connection,seed)
    if not records:raise ValueError('No accepted source pairs')
    pool=read_pool(database,{r['ticker'] for r in records})
    frozen,stats=sample_records(records,pool,seed)
    held,validation,golds=validation_pool(frozen,pool)
    index={p['text']:i for i,p in enumerate(validation)}
    multiple=defaultdict(set)
    for r in held:multiple[r['query_group']].add(index[r['positive']['text']])
    # Credit any saved positive for a remark; never count another known match as wrong.
    golds=[sorted(multiple[r['query_group']]) for r in held]
    engine.write_rows(out/'records.jsonl.gz',frozen);engine.write_rows(out/'validation_pool.jsonl.gz',validation)
    atomic_json(out/'validation.json',dict(query_ids=[r['id'] for r in held],golds=golds,query_count=len(held),candidate_count=len(validation),candidate_ids=[p['passage_id'] for p in validation]))
    request=dict(version='original-remark-training-v1',source=str(source.resolve()),database=str(database.resolve()),seed=seed,rates=[learning_rate],epochs=3,evaluate_every_steps=25,temperature=.05,retriever_batch_size=2,reranker_batch_size=8,reranker_max_length=512,query_mode='Original remark text, no synthesis or query prefix',metadata=True,selection='Highest validation MRR per component, including step 0; earliest tie. Fixed CLI learning rate; starts from original pretrained weights.',validation='All held-out issuers parsed passages; fixed original MPNet top20 for reranker; any known positive credited for a remark.',negative_rules='Same four categories as the previous ablation; exclude all known positive texts for the same issuer/remark; skip unavailable categories.',period_guard='Same nonnull company-quarter ID, issuer CIK, and fiscal year/quarter required. Q4 maps to the fiscal-year 10-K.',labels='Recovered accepted-source pairs; no new assertion of complete original-remark factual support.',baseline=str(Path(json.loads((source/'pilot_plan.json').read_text())['baseline']).resolve()),enriched_reference=str((ROOT/'data/experiments/finrank_metadata_enrichment/v1').resolve()))
    counts=dict(pairs=len(frozen),unique_remarks=len({r['query_group'] for r in frozen}),unique_passages=len({r['positive']['passage_id'] for r in frozen}),issuers=len({r['ticker'] for r in frozen}),split_counts=dict(Counter(r['split'] for r in frozen)),validation_passages=len(validation),total_negatives=sum(len(r['negatives']) for r in frozen),negative_counts=stats,verified_exact_period_links=len(records))
    files={name:sha(out/name) for name in ['records.jsonl.gz','validation_pool.jsonl.gz','validation.json']}
    plan=dict(request=request,counts=counts,files=files,source_hashes=hashes)
    atomic_json(path,plan);return plan


def selection(out):
    chosen={}
    rate=json.loads((out/'plan.json').read_text())['request']['rates'][0]
    for kind in ['retriever','reranker']:
        directory=engine.run_dir(out,True,kind,rate);status=json.loads((directory/'status.json').read_text())
        if status['phase']!='trained':raise ValueError('Complete both models first')
        chosen[kind]=dict(rate=rate,best_step=status['best_step'],best_validation_mrr=status['best_validation_mrr'],checkpoints={'enriched':str((directory/'best').resolve())})
    atomic_json(out/'selection.json',chosen);return chosen


def evaluate(out,device=None):
    from dearborn.finrank import TEXT,RECORDS,INDEX
    from dearborn.experiments.finrank_metadata_enrichment import prepare as prepare_enriched
    from dearborn.experiments.finrank_eda import table
    plan=json.loads((out/'plan.json').read_text())
    _,texts,_,_=prepare_enriched(Path(plan['request']['baseline']),Path(plan['request']['enriched_reference']))
    engine.evaluate(out,device,chosen=selection(out),training_cases=[('remark_training','enriched',texts)])
    path=out/'evaluation/results.json';result=json.loads(path.read_text())
    prior=SOURCE.parent/'hard-negatives-ablation-v1/evaluation/results.json'
    if prior.exists():
        result['models']['question_training_500']=json.loads(prior.read_text())['models']['metadata_training']
        result['question_reference_sha256']=sha(prior)
    rows=[]
    order=['baseline','metadata_only','question_training_500','remark_training']
    for name in order:
        if name not in result['models']:continue
        m=result['models'][name];sm=m['source_metrics'];gm=m['gold_passage_metrics']
        rows.append([name,*[f'{v:.2f}' for v in [sm['issuer_hit@1'],sm['source_hit@1'],gm['recall@5'],gm['recall@10'],gm['recall@20'],gm['mrr'],gm['ndcg@10']]]])
    report=table(['Case','Issuer Hit@1','Source Hit@1','R@5','R@10','R@20','MRR','nDCG@10'],rows)
    result['note']='Original-remark queries for training, unchanged human FinRank questions for evaluation. Both models start from original weights and use metadata-enriched passages. Question-training reference used 500 pairs; this run uses all recovered pairs, so it is not a pure query-form ablation. Source pair acceptance originally verified generated summaries, not necessarily every claim in the source remark.'
    atomic_json(path,result)
    (out/'evaluation/REPORT.md').write_text('# Original remark-to-filing training\n\n'+report+'\n'+result['note']+'\n')
    print(report);print('DONE: direct remark training and FinRank comparison saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','run'])
    parser.add_argument('--source-dir',type=Path,default=SOURCE);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    parser.add_argument('--database',type=Path,default=ROOT/'data/corpus.sqlite');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    parser.add_argument('--learning-rate',type=float,default=2e-6)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args.source_dir.resolve(),out,args.database,args.seed,args.learning_rate);print(json.dumps(plan['counts'],indent=2),flush=True)
        if args.action=='prepare':print('DONE: original remark pairs frozen; no model loaded.');return
        if args.action in {'train','run'}:
            engine.freeze_reranker_candidates(out,args.device)
            for kind in ['retriever','reranker']:
                print(f'STAGE: enriched {kind}, original remarks, lr={args.learning_rate}',flush=True)
                engine.train_one(out,True,kind,args.learning_rate,args.device)
        if args.action in {'evaluate','run'}:evaluate(out,args.device)


if __name__=='__main__':main()
