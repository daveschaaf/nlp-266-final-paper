"""Five reviewed remark/evidence bundles → questions → retrieval → JEV, resumable."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pydantic import BaseModel, ConfigDict, Field
from .remark_filing_trial import ROOT, atomic, load, sha, generate_backend
from .finrank_summary_roundtrip import MPNET, RERANKER

DEFAULT=ROOT/'data/experiments/finrank_synthesis/reviewed-five-v1'
SOURCE=ROOT/'data/experiments/remark_filing_validation/fresh-topic30-reviewed-v4'
INPUTS=ROOT/'prompts/remark_generation_five_inputs_v1.json'
EXAMPLES=ROOT/'prompts/remark_generation_examples_v1.json'
GENERATOR=ROOT/'prompts/remark_generation_v1.md'
JUDGE=ROOT/'prompts/remark_generation_judge_v1.md'
JEV_RUNTIME='git+https://github.com/internet-projects/autotrust-jev27b-mlx@5ef540537b4fc38e6b1586f321d234b465040bc9'
SNAPSHOT=Path.home()/'.cache/huggingface/hub/models--netprojects--AutoTrust-JEV-27B-v1.1-mlx-4bit/snapshots/39bc135bf354aac7a12a8a8328d2d6f7d6039a11'

class Question(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question:str
    answer:str
    evidence_indices:list[int]=Field(min_length=1,max_length=2)
class Questions(BaseModel):
    model_config=ConfigDict(extra='forbid')
    questions:list[Question]=Field(max_length=3)

def validate(raw,case):
    value=Questions.model_validate_json(raw).model_dump();seen=set()
    for q in value['questions']:
        key=' '.join(q['question'].casefold().split())
        if not key or not q['answer'].strip():raise ValueError('Empty question or diagnostic answer')
        if key in seen:raise ValueError('Duplicate question')
        seen.add(key)
        indices=q['evidence_indices']
        if len(indices)!=len(set(indices)) or any(i<0 or i>=len(case['passages']) for i in indices):raise ValueError('Invalid evidence index')
    return value

def prepare(out,inputs_path=None,corpus_path=None):
    if (out/'plan.json').exists():
        plan=load(out/'plan.json')
        for name,digest in plan['files'].items():
            if sha(out/name)!=digest:raise ValueError('Frozen input changed: '+name)
        return plan
    out.mkdir(parents=True,exist_ok=True)
    inputs_path=Path(inputs_path or INPUTS);corpus_path=Path(corpus_path or SOURCE/'filing_pool.json.gz')
    cases=load(inputs_path);examples=load(EXAMPLES)
    from dearborn.finrank_policy import assert_not_finrank_tickers
    if not cases:raise ValueError("No inputs")
    assert_not_finrank_tickers([c['ticker'] for c in cases])
    if {c['ticker'] for c in cases}&{e['ticker'] for e in examples['examples']}:raise ValueError('Example/test issuer overlap')
    with gzip.open(corpus_path,'rt') as f:pool=json.load(f)
    selected={c['accession_number'] for c in cases}
    docs={p['chunk_id']:dict(id=p['chunk_id'],text=p['text']) for a in selected for p in pool[a]}
    for c in cases:
        for p in c['passages']:
            if docs[p['passage_id']]['text']!=p['text']:raise ValueError('Evidence changed')
    atomic(out/'inputs.json',cases);atomic(out/'examples.json',examples)
    (out/'generator.md').write_text(GENERATOR.read_text());(out/'judge.md').write_text(JUDGE.read_text())
    with gzip.open(out/'corpus.jsonl.gz','wt') as f:
        for key in sorted(docs):f.write(json.dumps(docs[key],ensure_ascii=False)+'\n')
    plan=dict(version='reviewed-five-generation-v1',model='nemotron6bit',max_tokens=1800,snapshot=str(SNAPSHOT),retrieve_k=50,rerank_k=5,
              jev_runtime=JEV_RUNTIME,corpus_count=len(docs),files={n:sha(out/n) for n in ['inputs.json','examples.json','generator.md','judge.md','corpus.jsonl.gz']},
              source_hashes={str(f):sha(f) for f in [inputs_path,EXAMPLES,GENERATOR,JUDGE,corpus_path]},
              note='Development pilot, no training. Assistant-bounded inputs authorized by user; new questions and corrections require human review. No source labels overwritten.')
    atomic(out/'plan.json',plan);return plan

def metrics(mx):
    # Runtime telemetry is diagnostic; avoid turning a ps/sysctl permission issue into inference failure.
    import resource
    return dict(active_mlx_gb=mx.get_active_memory()/1e9,peak_mlx_gb=mx.get_peak_memory()/1e9,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6))

def generate(out,plan,backend=None):
    cases=load(out/'inputs.json');todo=[c for c in cases if load(out/'generation'/f"{c['case_id']}.json",{}).get('status')!='completed']
    target=plan.get('target_questions')
    if not todo or (target and len(candidates(out))>=target):return
    mx=None
    if backend is None:
        import mlx.core as mx
        backend,_=generate_backend(plan['model'],plan['max_tokens'])
    print(f"Loading {plan['model']}: {len(todo)} unfinished inputs; up to three questions each.",flush=True)
    demos=json.dumps(load(out/'examples.json')['examples'],ensure_ascii=False)
    for c in todo:
        if target and len(candidates(out))>=target:break
        path=out/'generation'/f"{c['case_id']}.json";old=load(path,{})
        if old:atomic(out/'attempts'/f"{c['case_id']}-generation-{old.get('attempt',0)}.json",old)
        # Original remark and audit labels are provenance only, not generator input.
        current={k:c[k] for k in ['company_name','ticker','year','quarter','fiscal_period_end','form','bounded_remark']}
        current['passages']=[dict(index=i,text=p['text']) for i,p in enumerate(c['passages'])]
        prompt=(out/'generator.md').read_text()+'\nDEMONSTRATIONS:\n'+demos+'\nCURRENT INPUT:\n'+json.dumps(current,ensure_ascii=False)
        row=dict(case_id=c['case_id'],attempt=old.get('attempt',0)+1,status='running',prompt=prompt);atomic(path,row)
        tick=time.perf_counter()
        if mx:mx.reset_peak_memory()
        try:
            raw=backend(prompt,Questions);row['raw_response']=raw;atomic(path,row)
            row.update(status='completed',output=validate(raw,c))
        except Exception as e:row.update(status='failed',error=f'{type(e).__name__}: {e}')
        finally:
            row['seconds']=time.perf_counter()-tick
            if mx:row['memory']=metrics(mx)
            atomic(path,row)
        print(f"{c['case_id']} {c['ticker']}: {row['status']} questions={len(row.get('output',{}).get('questions',[]))} {row['seconds']:.1f}s",flush=True)
        if row.get('error'):print(row['error'],flush=True)
        if mx:mx.clear_cache()

def candidates(out):
    results=[]
    for c in load(out/'inputs.json'):
        g=load(out/'generation'/f"{c['case_id']}.json",{})
        if g.get('status')!='completed':continue
        for i,q in enumerate(g['output']['questions']):
            key=hashlib.sha256((c['case_id']+'\0'+q['question']).encode()).hexdigest()
            results.append(dict(id=key,case_id=c['case_id'],ticker=c['ticker'],company_name=c['company_name'],fiscal_period_end=c['fiscal_period_end'],
                                year=c['year'],quarter=c['quarter'],form=c['form'],question=q['question'],answer=q['answer'],
                                evidence=[c['passages'][j] for j in q['evidence_indices']]))
    target=load(out/'plan.json',{}).get('target_questions')
    if target:results=results[:target]
    atomic(out/'candidates.json',results);return results

def retrieve(out,plan):
    from .filing_window_retrieval import run_stage
    tasks=[];questions=[]
    for q in candidates(out):
        for i,p in enumerate(q['evidence']):
            key=f"{q['id']}:{i}";tasks.append(dict(id=key,passage_id=p['passage_id']))
            questions.append(dict(id=key,status='completed',output=dict(status='ready',question=q['question'])))
    if not tasks:return
    dest=out/'retrieval';dest.mkdir(exist_ok=True)
    # Compatible adapter to existing cached MPNet window and MS MARCO implementation.
    import shutil
    if not (dest/'corpus.jsonl.gz').exists():shutil.copyfile(out/'corpus.jsonl.gz',dest/'corpus.jsonl.gz')
    digest=hashlib.sha256()
    with gzip.open(dest/'corpus.jsonl.gz','rt') as f:
        for line in f:digest.update(line.encode())
    adapter=dict(cases=tasks,mpnet=MPNET,reranker=RERANKER,max_length=512,retrieve_k=50,rerank_k=5,
                 retrieval_encoding='token-windows-parent-max-v1',window_overlap_tokens=32,corpus_count=plan['corpus_count'],corpus_sha256=digest.hexdigest())
    atomic(dest/'plan.json',adapter)
    (dest/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
    run_stage(dest,adapter)

def rankings(out):
    from .finrank_summary_roundtrip import done
    return done(out/'retrieval/retrieval.jsonl')

def gate(q,ranks):
    rows=[ranks.get(f"{q['id']}:{i}") for i in range(len(q['evidence']))]
    if not all(rows):return 'pending_retrieval'
    if not all(r['retrieval_pass'] for r in rows):return 'retrieval_miss'
    if not all(r['rerank_pass'] for r in rows):return 'rerank_miss'
    return 'judge_ready'

def judge_tasks(out,ungated=False):
    ranks={} if ungated else rankings(out)
    folder='judgments_ungated' if ungated else 'judgments'
    return [q for q in candidates(out) if (ungated or gate(q,ranks)=='judge_ready') and load(out/folder/f"{q['id']}.json",{}).get('status')!='completed']

def judge(out,plan,ungated=False):
    todo=judge_tasks(out,ungated)
    folder='judgments_ungated' if ungated else 'judgments'
    if not todo:return
    if not Path(plan['snapshot']).exists():raise ValueError('Local JEV snapshot missing: '+plan['snapshot'])
    import mlx.core as mx
    import jev27b_mlx
    print(f'Loading JEV for {len(todo)} question/evidence judgments.',flush=True)
    model=jev27b_mlx.load(plan['snapshot']);rubric=model.question('choice',(out/'judge.md').read_text(),['pass','fail'])
    for q in todo:
        path=out/folder/f"{q['id']}.json";old=load(path,{})
        if old:atomic(out/'attempts'/f"{q['id']}-{folder}-{old.get('attempt',0)}.json",old)
        row=dict(id=q['id'],status='running',attempt=old.get('attempt',0)+1,input=q);atomic(path,row)
        tick=time.perf_counter();mx.reset_peak_memory()
        try:
            result=model.decide(json.dumps(q,ensure_ascii=False),rubric)
            if result['choice'] not in {'pass','fail'}:raise ValueError('Invalid JEV choice')
            row.update(status='completed',result=result)
        except Exception as e:row.update(status='failed',error=f'{type(e).__name__}: {e}')
        finally:
            row.update(seconds=time.perf_counter()-tick,memory=metrics(mx));atomic(path,row)
        print(f"{q['case_id']} JEV: {row.get('result',{}).get('choice',row['status'])} {row['seconds']:.1f}s",flush=True);mx.clear_cache()

def report_ungated(out):
    rows=[];counts=Counter()
    for q in candidates(out):
        j=load(out/'judgments_ungated'/f"{q['id']}.json",{})
        state=j.get('result',{}).get('choice',j.get('status','pending'))
        counts[state]+=1;rows.append(dict(candidate=q,judgment=j))
    value=dict(mode='all questions; no retrieval/reranker gate',counts=dict(counts),results=rows)
    atomic(out/'ungated_jev_review.json',value)
    print(json.dumps(dict(mode=value['mode'],counts=value['counts'])),flush=True)
    return any(k not in {'pass','fail'} for k in counts)

def report(out):
    qs=candidates(out);ranks=rankings(out);counts=Counter();rows=[]
    generation=Counter(load(out/'generation'/f"{c['case_id']}.json",{}).get('status','pending') for c in load(out/'inputs.json'))
    for q in qs:
        state=gate(q,ranks);j=load(out/'judgments'/f"{q['id']}.json",{})
        if state=='judge_ready':state=('proposed_accept' if j['result']['choice']=='pass' else 'judge_reject') if j.get('status')=='completed' else 'pending_judge'
        counts[state]+=1;rows.append(dict(q,state=state,source_ranks=[ranks.get(f"{q['id']}:{i}",{}) for i in range(len(q['evidence']))]))
    target=load(out/'plan.json',{}).get('target_questions')
    summary=dict(input_count=len(load(out/'inputs.json')),target_questions=target,generation=dict(generation),candidate_questions=len(qs),counts=dict(counts),note='Retrieval misses are separate from factual rejection. Proposed acceptances require human review; no training export.')
    atomic(out/'summary.json',summary);atomic(out/'review.json',rows);print(json.dumps(summary),flush=True)
    generation_pending=(max(0,target-len(qs)) if target else generation.get('pending',0)+generation.get('failed',0)+generation.get('running',0))
    return generation_pending+sum(v for k,v in counts.items() if k.startswith('pending'))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','generate','retrieve','judge','judge-all','status']);p.add_argument('--output-dir',type=Path,default=DEFAULT)
    p.add_argument('--ungated',action='store_true',help='Judge every saved question; save separate judgments_ungated outputs')
    a=p.parse_args(argv)
    if a.ungated and a.action!='judge':p.error('--ungated applies to judge only')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        # Child stages have isolated processes; the parent holds the full-run lock.
        if a.action in {'run','judge-all','prepare','status'}:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            stage_lock=(out/'stage.lock').open('a')
            fcntl.flock(stage_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(out)
        if a.action=='prepare':print(f"READY: five inputs, ten examples, {plan['corpus_count']} corpus passages. No inference started.");return 0
        if a.action=='status':report(out);return 0
        if a.action=='judge-all':
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
            command=['uv','run','--no-project','--with',plan.get('jev_runtime',JEV_RUNTIME),'--with','pydantic','python','-u','-m',__spec__.name,'judge','--ungated','--output-dir',str(out)]
            try:subprocess.run(command,cwd=ROOT,env=env,check=True)
            except (KeyboardInterrupt,subprocess.CalledProcessError):
                report_ungated(out);print('PAUSED: rerun judge-all to resume.');return 1
            return int(report_ungated(out))
        if a.action=='judge' and a.ungated:
            judge(out,plan,ungated=True);return int(report_ungated(out))
        if a.action=='run':
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
            try:
                for stage in ['generate','retrieve','judge']:
                    print('STAGE: '+stage,flush=True)
                    command=[sys.executable,'-u','-m',__spec__.name,stage,'--output-dir',str(out)]
                    if stage=='judge':command=['uv','run','--no-project','--with',plan.get('jev_runtime',JEV_RUNTIME),'--with','pydantic','python',*command[1:]]
                    subprocess.run(command,cwd=ROOT,env=env,check=True)
            except (KeyboardInterrupt,subprocess.CalledProcessError):report(out);print('PAUSED: rerun the same command to resume.');return 1
        else:globals()[a.action](out,plan)
        pending=report(out)
        print('INCOMPLETE: rerun the same command to retry failures.' if pending else 'DONE: candidates, retrieval diagnostics and proposed judgments saved; no training started.')
        return int(bool(pending)) if a.action=='run' else 0
if __name__=='__main__':raise SystemExit(main())
