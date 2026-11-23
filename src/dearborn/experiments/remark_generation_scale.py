"""Seeded own-corpus generation expansion; JEV ungated, retrieval diagnostic only."""
import argparse
from collections import defaultdict,Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from . import remark_generation_five as base
from .remark_filing_trial import atomic,load,sha,ROOT

DEFAULT=ROOT/'data/experiments/finrank_synthesis/reviewed-generation-100-v1'
RECORDS=ROOT/'data/experiments/finrank_training/remark561-metadata-lr2e7-v1/records.jsonl.gz'

def select_records(records,excluded,seed,limit):
    buckets=defaultdict(list);seen=set()
    for r in sorted(records,key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()):
        key=(r['ticker'],' '.join(r['question'].casefold().split()),' '.join(r['positive']['text'].split()))
        if r['ticker'] in excluded or key in seen:continue
        seen.add(key);buckets[r['ticker']].append(r)
    order=sorted(buckets,key=lambda t:hashlib.sha256(f'{seed}:{t}'.encode()).hexdigest())
    selected=[];used_passages=set()
    while len(selected)<limit and any(buckets.values()):
        for ticker in order:
            if len(selected)>=limit:break
            while buckets[ticker]:
                r=buckets[ticker].pop(0);pid=r['positive']['passage_id']
                if pid in used_passages:continue
                used_passages.add(pid);selected.append(r);break
    return selected

def prepare(out,target=100,seed=42):
    if (out/'plan.json').exists():
        plan=base.prepare(out)
        if (plan.get('target_questions'),plan.get('selection_seed'))!=(target,seed):raise ValueError('Changed settings; use original options or a new directory')
        return plan
    examples=load(base.EXAMPLES)['examples'];pilot=load(base.DEFAULT/'inputs.json')
    excluded={r['ticker'] for r in examples+pilot}
    from dearborn.finrank_policy import FINRANK_TICKERS,assert_not_finrank_tickers
    excluded.update(FINRANK_TICKERS)
    with gzip.open(RECORDS,'rt') as f:records=[json.loads(line) for line in f]
    chosen=select_records(records,excluded,seed,target)
    if len(chosen)<target:raise ValueError(f'Only {len(chosen)} unique eligible inputs; requested reserve {target}')
    inputs=[];pool={}
    with sqlite3.connect((ROOT/'data/corpus.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for i,r in enumerate(chosen,1):
            p=r['positive'];a=p['accession_number'];prov=r['provenance'][0]
            if not prov.get('company_quarter_id'):raise ValueError('Missing exact-quarter linkage')
            assert_not_finrank_tickers([r['ticker']])
            if a not in pool:pool[a]=[dict(x) for x in db.execute('SELECT chunk_id,text FROM filing_chunk WHERE accession_number=? ORDER BY chunk_id',(a,))]
            frozen=next((x for x in pool[a] if x['chunk_id']==p['passage_id']),None)
            if not frozen or frozen['text']!=p['text']:raise ValueError('Saved positive no longer matches parsed filing')
            quarter=int(prov['company_quarter_id'].split(':')[-1][1:]);year=int(p['fiscal_year'].replace('FY',''))
            inputs.append(dict(case_id=f'scale-{i:03}',ticker=r['ticker'],company_name=p['name'],year=year,quarter=quarter,fiscal_period_end=p['fiscal_period_end'],form=p['form'],accession_number=a,
                original_remark=r['question'],bounded_remark=r['question'],correction='None; no new manual narrowing performed.',selection_status='prior_pipeline_silver_match_not_newly_human_verified',provenance=r['provenance'],
                passages=[dict(passage_id=p['passage_id'],text=p['text'])]))
    atomic(out/'queue_inputs.json',inputs)
    with gzip.open(out/'queue_filing_pool.json.gz','wt') as f:json.dump(pool,f)
    plan=base.prepare(out,out/'queue_inputs.json',out/'queue_filing_pool.json.gz')
    plan.update(target_questions=target,selection_seed=seed,source_records_sha256=sha(RECORDS),source_records=str(RECORDS),issuer_count=len({r['ticker'] for r in inputs}),excluded_issuers=sorted(excluded),
        selection='Seeded hash ordering; issuer round-robin; unique source passage per input. Up to 100 reserve inputs, stop generation at 100 valid-format questions. Not 100 JEV accepts. No selection by retrieval or new model outputs.',
        evaluation_mode='JEV on every question without retrieval gate. Retrieval diagnostics only. Judge proposed accepts require human audit.',
        note='Expanded development pilot from existing silver remark-filing matches, not newly hand-reviewed inputs. Original remarks unchanged; generator must ground each question in its supplied passage. No training started.')
    atomic(out/'plan.json',plan);return plan

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','status']);parser.add_argument('--output-dir',type=Path,default=DEFAULT);parser.add_argument('--target',type=int,default=100);parser.add_argument('--seed',type=int,default=42)
    a=parser.parse_args(argv)
    if a.target<1:parser.error('Target must be positive')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'scale.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(out,a.target,a.seed)
        print(f"Target={a.target} questions; reserve_inputs={len(load(out/'inputs.json'))}; issuers={plan['issuer_count']}; JEV ungated.",flush=True)
        if a.action=='prepare':print('READY: no inference started.');return 0
        if a.action=='status':base.report(out);return int(base.report_ungated(out))
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        try:
            for stage in ['generate','retrieve','judge-all']:
                print('STAGE: '+stage,flush=True)
                subprocess.run([sys.executable,'-u','-m',base.__name__,stage,'--output-dir',str(out)],cwd=ROOT,env=env,check=True)
        except (KeyboardInterrupt,subprocess.CalledProcessError):
            base.report_ungated(out);print('PAUSED/INCOMPLETE: rerun the same command to resume.');return 1
        generated=len(base.candidates(out));pending=base.report_ungated(out)
        print(f"Generated={generated}/{a.target}; all JEV results in ungated_jev_review.json; retrieval diagnostics in retrieval/retrieval.jsonl.",flush=True)
        if generated<a.target or pending:print('INCOMPLETE: rerun to retry failures; reserve exhaustion is not success.');return 1
        print('DONE: 100-question development evaluation complete; no training started.' if a.target==100 else 'DONE: development evaluation complete; no training started.');return 0
if __name__=='__main__':raise SystemExit(main())
