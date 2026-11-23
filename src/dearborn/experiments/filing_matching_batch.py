"""Freeze fresh retained remarks and run the fixed matching development trial."""
import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from dearborn.constants import PROJECT_ROOT
from dearborn.experiments.filing_matching_development import main as run_trial


CRITERIA = dict(support_precision_minimum=0.95,minimum_accepted_pairs=50,
                annotation_execution_success_minimum=0.95,rejected_pairs_to_review=20,
                maximum_clear_false_rejections_in_sample=2,
                difficult_derived_support='Report separately',traceability='Required for every accepted pair')


def diverse_sample(rows, count, seed):
    rng=random.Random(seed);groups=defaultdict(list)
    for row in sorted(rows,key=lambda x:x['remark_id']):groups[row['ticker']].append(row)
    for group in groups.values():rng.shuffle(group)
    selected=[]
    while groups and len(selected)<count:
        tickers=sorted(groups);rng.shuffle(tickers)
        for ticker in tickers:
            selected.append(groups[ticker].pop())
            if not groups[ticker]:del groups[ticker]
            if len(selected)==count:break
    return selected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--label-run-id',default='qwen-corpus-v1')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--run-planned',action='store_true',help='Use the frozen input in an existing planned directory.')
    args=parser.parse_args(argv)
    if args.count<1 or (args.plan_only and args.run_planned):parser.error('Invalid count or mode.')
    reference=PROJECT_ROOT/'data/experiments/remark_filing_validation/development_reference.jsonl'
    frozen=args.output_dir/'fresh_remarks.jsonl'
    if not args.run_planned:
        excluded={x['remark_id'] for x in map(json.loads,reference.read_text().splitlines())}
        labels=PROJECT_ROOT/'data/qwen_corpus_labels.sqlite'
        with closing(sqlite3.connect(labels.resolve().as_uri()+'?mode=ro',uri=True)) as c:
            c.row_factory=sqlite3.Row
            c.execute('ATTACH DATABASE ? AS corpus',((PROJECT_ROOT/'data/corpus.sqlite').resolve().as_uri()+'?mode=ro',))
            rows=[dict(x) for x in c.execute('''SELECT r.remark_chunk_id AS remark_id,r.remark_text,r.ticker,r.call_id,
                r.remark_hash,f.accession_number AS filing_accession,f.parser_version
                FROM remark_label r JOIN corpus.earnings_call e ON e.call_id=r.call_id
                JOIN corpus.filing f ON f.company_quarter_id=e.company_quarter_id
                WHERE r.run_id=? AND r.status='completed' AND r.label='yes'
                AND f.parse_complete=1 AND EXISTS(SELECT 1 FROM corpus.filing_chunk fc WHERE fc.accession_number=f.accession_number)
                ORDER BY r.remark_chunk_id''',(args.label_run_id,)) if x['remark_id'] not in excluded]
        counts=defaultdict(int)
        for row in rows:counts[row['remark_id']]+=1
        rows=[r for r in rows if counts[r['remark_id']]==1]
        if len(rows)<args.count:parser.error(f'Only {len(rows)} eligible fresh remarks.')
        selected=diverse_sample(rows,args.count,args.seed)
        for row in selected:
            assert hashlib.sha256(row['remark_text'].encode()).hexdigest()==row['remark_hash']
            row.update(case_id='fresh:'+row['remark_id'],evidence=[],dataset_role='fresh_development')
        args.output_dir.mkdir(parents=True,exist_ok=False)
        frozen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        (args.output_dir/'plan.json').write_text(json.dumps(dict(count=len(selected),seed=args.seed,
            issuer_count=len({r['ticker'] for r in selected}),call_count=len({r['call_id'] for r in selected}),
            label_run_id=args.label_run_id,input_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
            acceptance_criteria=CRITERIA,model='qwen14b',top_k=4,numeric_gate_enabled=False),indent=2))
    else:
        plan=json.loads((args.output_dir/'plan.json').read_text())
        if hashlib.sha256(frozen.read_bytes()).hexdigest()!=plan['input_sha256']:
            parser.error('Frozen input checksum mismatch.')
    print(f'Frozen inputs: {frozen}',flush=True)
    if args.plan_only:
        print('DONE: plan saved; no model loaded.');return 0
    started=time.perf_counter()
    try:
        return run_trial(['--reference',str(frozen),'--output-dir',str(args.output_dir/'run'),
                          '--model','qwen14b','--top-k','4','--annotate','--comparison-only'])
    finally:
        (args.output_dir/'runtime.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':raise SystemExit(main())
