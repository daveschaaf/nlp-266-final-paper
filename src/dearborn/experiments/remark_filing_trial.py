"""Frozen own-corpus remark matching trial. prepare/prompts/run/status; no training."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import json
from pathlib import Path
import re
import resource
import sqlite3
import sys
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'data/experiments/remark_filing_validation/fresh-topic30-summary-v3'
INSTRUCTIONS=ROOT/'prompts/remark_filing_matching_v2.md'
EXAMPLES=ROOT/'prompts/remark_filing_matching_examples_v1.json'
QUOTAS={'Financials':6,'Company Overview':4,'Risk':4,'Accounting':4,'Legal':4,'Shareholder Return':4,'Footnotes':2,'Governance':2}
PATTERNS={
 'Financials':r'net income|net revenue|operating cash flow|gross margin|operating margin',
 'Company Overview':r'completed.*acquisition|acquired|business model|reportable segments|distribution centers|manufacturing facilities',
 'Risk':r'tariff|cybersecurity|supply.chain|uncertain|exposure',
 'Accounting':r'accounting|tax rate|reclassif|GAAP|recognition',
 'Legal':r'litigation|settlement|lawsuit|consent|regulatory approv|warning letter',
 'Shareholder Return':r'dividend|repurchas|buyback',
 'Footnotes':r'\bpension\b|\bimpairment\b|\bleases\b|\bmaturities\b|restructuring charge',
 'Governance':r'board of directors|\bCEO\b|chairman|appointed|succession|prepare to retire',
}
class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')
class Request(Strict):
    eligibility:Literal['eligible','personal_opinion','too_general','unclear']
    information_request:str
    required_evidence:list[str]
    reason:str
class Evidence(Strict):
    passage_id:str
    supplies:list[str]
    summary:str
class Judgment(Strict):
    decision:Literal['accept','review','reject']
    reason:str
    request_coverage:Literal['complete','partial','none']
    evidence:list[Evidence]
    original_remark_coverage:Literal['complete','partial','not_established']
    uncovered_original_assertions:list[str]

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(path):return sha_bytes(path.read_bytes())
def normalized(s):return ' '.join(s.casefold().split())
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)
def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def select_cases(rows,examples,excluded_tickers,seed):
    demo_ids={e['input']['original_remark'] for e in examples}
    demo_tickers={e['input']['ticker'] for e in examples}
    rows=[r for r in rows if r['ticker'] not in demo_tickers|set(excluded_tickers)
          and normalized(r['text']) not in {normalized(t) for t in demo_ids}
          and r['ticker']!='VLO' # Previously identified prepared/Q&A parse contamination.
          and 15<=len(r['text'].split())<=95
          and not re.search(r'forward.looking statements|safe harbor|joining me|on today.s call|with us today|joined by|joining us|\?',r['text'],re.I)]
    unique={r['remark_chunk_id']:r for r in rows}
    rows=sorted(unique.values(),key=lambda r:hashlib.sha256(f'{seed}:{r["remark_chunk_id"]}'.encode()).hexdigest())
    selected=[];seen=set();texts=set();issuers=Counter()
    # Fill rare buckets first, independent of passage retrieval or model judgments.
    order=sorted(QUOTAS,key=lambda t:sum(bool(re.search(PATTERNS[t],r['text'],re.I)) for r in rows))
    counts=Counter()
    for topic in order:
        for r in rows:
            if counts[topic]>=QUOTAS[topic]:break
            if r['remark_chunk_id'] in seen or normalized(r['text']) in texts or issuers[r['ticker']]>=2:continue
            if not re.search(PATTERNS[topic],r['text'],re.I):continue
            if topic in {'Financials','Accounting','Shareholder Return','Footnotes'}:
                if not re.search(r'\d',r['text']):continue
                if re.search(r'guidance|we expect|we anticipate|forecast|we plan|we intend|expected|reaffirm|outlook|we will|we.ve assumed|we assume',r['text'],re.I):continue
            selected.append(dict(r,sampling_topic=topic));seen.add(r['remark_chunk_id']);texts.add(normalized(r['text']));issuers[r['ticker']]+=1;counts[topic]+=1
        if counts[topic]!=QUOTAS[topic]:raise ValueError(f'Not enough eligible rows for {topic}; got {counts[topic]} of {QUOTAS[topic]}')
    # Round-robin order by topic, retaining seeded within-bucket order.
    buckets={t:[r for r in selected if r['sampling_topic']==t] for t in QUOTAS}
    result=[]
    while any(buckets.values()):
        for topic in QUOTAS:
            if buckets[topic]:result.append(buckets[topic].pop(0))
    return [dict(r,case_id=f'trial-{i:02}') for i,r in enumerate(result,1)]

def prepare(out,database,seed=42,model='nemotron6bit',top_k=5,max_tokens=1400,instructions_path=None,examples_path=None,exclude_cases=()):
    existing=load(out/'plan.json')
    config=dict(seed=seed,model=model,top_k=top_k,max_tokens=max_tokens,runner_sha256=sha(Path(__file__)))
    instructions_path=Path(instructions_path or INSTRUCTIONS)
    examples_path=Path(examples_path or EXAMPLES)
    exclude_cases=tuple(Path(x).resolve() for x in exclude_cases)
    if instructions_path!=INSTRUCTIONS or examples_path!=EXAMPLES or exclude_cases:
        config.update(instructions_path=str(instructions_path.resolve()),examples_path=str(examples_path.resolve()),exclude_cases=[str(x) for x in exclude_cases])
    if existing:
        if existing['config']!=config:raise ValueError('Resume settings changed; use original settings or a new output directory')
        verify(out,existing);return existing
    instructions=instructions_path.read_text();examples=json.loads(examples_path.read_text())['examples']
    benchmark=ROOT/'data/FinRank/FinRank.jsonl'
    excluded={json.loads(l)['ticker'] for l in benchmark.read_text().splitlines()}
    exclusion_records=[dict(path=str(x),sha256=sha(x),cases=load(x)) for x in exclude_cases]
    # Exclude whole prior issuers, not just individual remarks, from the fresh trial.
    excluded.update(c['ticker'] for record in exclusion_records for c in record['cases'])
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        rows=[dict(r) for r in c.execute("""
SELECT e.ticker,e.company_name,e.call_id,e.call_date,e.company_quarter_id,e.year,e.quarter,
 rc.remark_chunk_id,rc.text,pr.segment_index,f.accession_number,f.form,f.fiscal_period_end,f.source_url
FROM remark_chunk rc JOIN prepared_remark pr ON pr.remark_id=rc.remark_id
JOIN earnings_call e ON e.call_id=pr.call_id JOIN parsed_earnings_call pe ON pe.call_id=e.call_id
JOIN company_quarter cq ON cq.company_quarter_id=e.company_quarter_id
JOIN filing f ON f.company_quarter_id=cq.company_quarter_id AND f.cik=e.cik
WHERE rc.eligible_for_synthesis=1 AND pr.segment_index < pe.qa_boundary_segment_index
AND f.parse_complete=1 AND e.year=2025 AND e.quarter=1 AND cq.cik=e.cik
AND cq.fiscal_quarter=e.quarter AND cq.fiscal_year='FY'||e.year
AND f.fiscal_year=cq.fiscal_year AND f.fiscal_period='Q1' AND f.form='10-Q'
AND (SELECT COUNT(*) FROM filing f2 WHERE f2.company_quarter_id=cq.company_quarter_id)=1
""")]
        cases=select_cases(rows,examples,excluded,seed)
        pool={}
        for case in cases:
            a=case['accession_number']
            if a not in pool:
                pool[a]=[dict(r) for r in c.execute('SELECT chunk_id,accession_number,part,item,item_title,chunk_index,text FROM filing_chunk WHERE accession_number=? ORDER BY chunk_id',(a,))]
            if not pool[a]:raise ValueError(f'Missing parsed filing {a}')
    out.mkdir(parents=True,exist_ok=True)
    atomic(out/'cases.json',cases);(out/'instructions.md').write_text(instructions);atomic(out/'examples.json',examples)
    with gzip.open(out/'filing_pool.json.gz','wt') as f:json.dump(pool,f,ensure_ascii=False)
    files=['cases.json','instructions.md','examples.json','filing_pool.json.gz']
    plan=dict(version='remark-filing-trial-v1',config=config,files={name:sha(out/name) for name in files},source_database=str(database.resolve()),source_hashes={'instructions':sha(instructions_path),'examples':sha(examples_path),'benchmark_issuer_exclusions':sha(benchmark)},query_count=len(cases),issuer_count=len({c['ticker'] for c in cases}),sampling_topics=dict(Counter(c['sampling_topic'] for c in cases)),selection='Seeded hash order within keyword buckets, rare buckets first; up to two remarks per issuer. Financials, Accounting, Shareholder Return and Footnotes require a digit and exclude listed forward-looking/guidance cues. Word-boundary matching avoids treating releases as leases. All example issuers and FinRank issuers excluded. Fiscal Q1 2025 only, exact company-quarter/CIK/year/quarter link, unique filing, before parsed Q&A boundary. VLO excluded for previously identified parser contamination. No candidate evidence inspected for selection.',topic_warning='Keyword sampling hints, not human labels or measured FinRank topic/type coverage.',retrieval='BM25 of generated request + required evidence, top5 within frozen exact-quarter filing; no remark numeric answer appended. No adjacent chunks automatically added.',purpose='Development trial. Human review required; model acceptances are not training labels.',created_at=datetime.now(timezone.utc).isoformat())
    if exclusion_records:
        atomic(out/'excluded_cases.json',exclusion_records)
        plan['files']['excluded_cases.json']=sha(out/'excluded_cases.json')
        plan['selection']+=' Prior trial issuers explicitly excluded; see excluded_cases.json.'
    atomic(out/'plan.json',plan);return plan

def verify(out,plan):
    for name,digest in plan['files'].items():
        if sha(out/name)!=digest:raise ValueError('Frozen input changed: '+name)

def request_input(case):
    return {k:case[k] for k in ['ticker','company_name','call_id','year','quarter','fiscal_period_end','text']}

def request_prompt(out,case):
    demos=[]
    for e in load(out/'examples.json'):
        h=e['human_review'];id=e['case_id']
        if id not in ['case-01','case-02','case-05','case-08','case-10']:continue
        eligibility='personal_opinion' if id=='case-08' else 'too_general' if id=='case-10' else 'eligible'
        demos.append(dict(remark=e['input']['original_remark'],ticker=e['input']['ticker'],output=dict(eligibility=eligibility,information_request=h.get('information_request',''),required_evidence=h.get('required_inputs',h.get('required_evidence',[])),reason=h.get('notes','Specific factual request grounded in the remark.'))))
    return (out/'instructions.md').read_text()+'\nExecute STAGE 1 ONLY. Output the Request JSON schema. Examples are demonstrations, not current evidence.\n'+json.dumps(demos,ensure_ascii=False)+'\nCURRENT INPUT:\n'+json.dumps(request_input(case),ensure_ascii=False)

def judge_prompt(out,case,request,candidates):
    # Keep examples small; use human-approved source passages, never another case's evidence as current evidence.
    demos=[]
    for e in load(out/'examples.json'):
        if e['case_id'] not in ['case-01','case-03','case-04','case-05','case-06']:continue
        h=e['human_review'];ids={p['chunk_id'] for p in h.get('passages',[]) if p.get('coverage')=='complete'}
        candidates_demo=[p for p in e['input']['candidate_passages'] if p['chunk_id'] in ids]
        if not candidates_demo:candidates_demo=e['input']['candidate_passages'][:2] if e['case_id']=='case-04' else e['input']['candidate_passages'][3:4]
        demos.append(dict(original_remark=e['input']['original_remark'],information_request=h.get('information_request',h.get('proposed_information_request','Find evidence for the specific named acquisition.')),candidate_passages=[dict(passage_id=p['chunk_id'],text=p['text']) for p in candidates_demo],human_judgment=h))
    reviewed=[e['stage2_example'] for e in load(out/'examples.json') if 'stage2_example' in e]
    if reviewed:demos=reviewed
    current=dict(metadata=request_input(case),request=request,candidate_passages=[dict(passage_id=p['chunk_id'],section=p['item_title'],text=p['text']) for p in candidates])
    return (out/'instructions.md').read_text()+'\nExecute STAGE 2 ONLY. Do not change the frozen request. Output the Judgment JSON schema.\nDEVELOPMENT EXAMPLES:\n'+json.dumps(demos,ensure_ascii=False)+'\nCURRENT INPUT:\n'+json.dumps(current,ensure_ascii=False)

def validate_request(raw):
    r=Request.model_validate_json(raw)
    if not r.reason.strip():raise ValueError('Empty request reason')
    if r.eligibility=='eligible':
        if not r.information_request.strip() or not r.required_evidence or any(not s.strip() for s in r.required_evidence):raise ValueError('Eligible request needs question and evidence requirements')
    elif r.information_request or r.required_evidence:raise ValueError('Ineligible remark must not generate a request')
    return r.model_dump()

def validate_judgment(raw,candidates):
    r=Judgment.model_validate_json(raw);by_id={p['chunk_id']:p['text'] for p in candidates}
    if not r.reason.strip():raise ValueError('Empty judgment reason')
    for e in r.evidence:
        if e.passage_id not in by_id:raise ValueError('Unknown passage ID')
        if not e.summary.strip():raise ValueError('Evidence summary must not be empty')
        if not e.supplies or any(not x.strip() for x in e.supplies):raise ValueError('Evidence must name supplied facts')
    if r.decision=='accept' and (r.request_coverage!='complete' or not r.evidence):raise ValueError('Acceptance requires complete request coverage and sourced evidence')
    if r.original_remark_coverage=='partial' and not r.uncovered_original_assertions:raise ValueError('Partial remark coverage must state what is uncovered')
    return r.model_dump()

def generate_backend(name,max_tokens):
    import outlines
    from mlx_lm.sample_utils import make_sampler
    from dearborn.labeling.models import LabelingModelLoader
    loaded=LabelingModelLoader(local_files_only=True).load(name)
    wrapped=outlines.from_mlxlm(loaded.model,loaded.input_processor);wrapped.type_adapter.has_chat_template=False
    generators={}
    def call(prompt,schema):
        if schema not in generators:generators[schema]=outlines.Generator(wrapped,schema)
        text=loaded.input_processor.apply_chat_template([{'role':'system','content':'/no_think\nReturn only the requested JSON object.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return generators[schema](text,max_tokens=max_tokens,sampler=make_sampler(temp=0),verbose=False)
    return call,loaded.model_id

def stage_path(out,id,stage):return out/'calls'/f'{id}-{stage}.json'
def completed(out,id,stage):
    r=load(stage_path(out,id,stage),{})
    return r if r.get('status')=='completed' else None

def report(out):
    cases=load(out/'cases.json');counts=Counter();rows=[]
    for c in cases:
        request=load(stage_path(out,c['case_id'],'request'),{})
        evidence=load(stage_path(out,c['case_id'],'evidence'),{})
        if request.get('status')=='failed':state='failed_request'
        elif request.get('status')!='completed':state='pending_request'
        elif request['output']['eligibility']!='eligible':state='filtered_'+request['output']['eligibility']
        elif evidence.get('status')=='failed':state='failed_evidence'
        elif evidence.get('status')!='completed':state='pending_evidence'
        else:state='model_'+evidence['output']['decision']
        counts[state]+=1;rows.append(dict(case_id=c['case_id'],ticker=c['ticker'],state=state,request=request.get('output'),judgment=evidence.get('output')))
    value=dict(planned=len(cases),counts=dict(counts),results=rows,note='Model decisions only; no human accuracy calculated and no training started.')
    atomic(out/'summary.json',value);atomic(out/'review_candidates.json',[dict(case=c,**r) for c,r in zip(cases,rows)])
    print(json.dumps(dict(planned=len(cases),counts=dict(counts))),flush=True);return value

def run(out,plan,limit=None,backend=None):
    verify(out,plan);cases=load(out/'cases.json')
    with gzip.open(out/'filing_pool.json.gz','rt') as f:pool=json.load(f)
    todo=[c for c in cases if not completed(out,c['case_id'],'request') or (completed(out,c['case_id'],'request')['output']['eligibility']=='eligible' and not completed(out,c['case_id'],'evidence'))]
    todo=todo[:limit] if limit is not None else todo
    if not todo:return report(out)
    print(f"Loading {plan['config']['model']} for {len(todo)} unfinished cases; one model, sequential calls. Failed calls retry on resume.",flush=True)
    mx=None
    if backend is None:
        import mlx.core as mx
        backend,model_id=generate_backend(plan['config']['model'],plan['config']['max_tokens'])
        atomic(out/'runtime.json',dict(model_id=model_id,backend='mlx-outlines',temperature=0))
    from .filing_matching_development import retrieve
    def call(c,stage,prompt,schema,validator):
        path=stage_path(out,c['case_id'],stage);prior=load(path,{})
        row=dict(case_id=c['case_id'],stage=stage,status='running',attempt=prior.get('attempt',0)+1,prompt=prompt,started_at=datetime.now(timezone.utc).isoformat());atomic(path,row)
        start=time.perf_counter()
        try:
            raw=backend(prompt,schema);row['raw_response']=raw;atomic(path,row)
            row.update(output=validator(raw),status='completed')
        except Exception as e:row.update(status='failed',error=f'{type(e).__name__}: {e}')
        except KeyboardInterrupt:
            row.update(status='interrupted');atomic(path,row);raise
        finally:
            row['seconds']=time.perf_counter()-start;row['peak_rss_gb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6)
            if mx:row['memory']=dict(active_gb=mx.get_active_memory()/1e9,cache_gb=mx.get_cache_memory()/1e9,peak_gb=mx.get_peak_memory()/1e9)
            atomic(path,row);gc.collect()
            if mx:mx.clear_cache()
        print(f"{c['case_id']} {c['ticker']} {stage}: {row['status']} {row['seconds']:.1f}s"+(f" {row['output'].get('decision',row['output'].get('eligibility'))}" if row.get('output') else ''),flush=True)
        if row.get('error'):print(row['error'],flush=True)
        return row if row['status']=='completed' else None
    for c in todo:
        request=completed(out,c['case_id'],'request') or call(c,'request',request_prompt(out,c),Request,validate_request)
        if not request or request['output']['eligibility']!='eligible':report(out);continue
        # Freeze retrieval before evidence inference, including exact source text.
        candidate_path=out/'retrieval'/f"{c['case_id']}.json"
        candidates=load(candidate_path)
        if candidates is None:
            query=request['output']['information_request']+' '+' '.join(request['output']['required_evidence'])
            candidates=retrieve(pool[c['accession_number']],query,plan['config']['top_k']);atomic(candidate_path,candidates)
        if not completed(out,c['case_id'],'evidence'):
            call(c,'evidence',judge_prompt(out,c,request['output'],candidates),Judgment,lambda raw:validate_judgment(raw,candidates))
        report(out)
    return report(out)

def migrate(source,out):
    """Same frozen cases and request outputs; new evidence protocol, no inference."""
    import shutil
    if source.resolve()==out.resolve():raise ValueError('Migration requires a new directory')
    source_plan=load(source/'plan.json')
    if not source_plan:raise ValueError('Source plan missing')
    verify(source,source_plan)
    if (out/'plan.json').exists():
        prior=load(out/'plan.json');verify(out,prior)
        if prior.get('migration',{}).get('source')!=str(source.resolve()):raise ValueError('Different migration source')
        return prior
    with (source/'runner.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ['cases.json','examples.json','filing_pool.json.gz']:shutil.copyfile(source/name,out/name)
        (out/'instructions.md').write_text(INSTRUCTIONS.read_text())
        inherited=[]
        for c in load(out/'cases.json'):
            row=completed(source,c['case_id'],'request')
            if row:
                row=dict(row,inherited_from=str(stage_path(source,c['case_id'],'request')))
                atomic(stage_path(out,c['case_id'],'request'),row);inherited.append(c['case_id'])
                candidate=source/'retrieval'/f"{c['case_id']}.json"
                if candidate.exists():
                    target=out/'retrieval'/candidate.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,target)
        plan=dict(source_plan,version='remark-filing-summary-trial-v2',config=dict(source_plan['config'],runner_sha256=sha(Path(__file__))),files={name:sha(out/name) for name in ['cases.json','instructions.md','examples.json','filing_pool.json.gz']},migration=dict(source=str(source.resolve()),source_plan_sha256=sha(source/'plan.json'),inherited_request_ids=inherited,evidence_policy='Re-infer every eligible evidence judgment under summary schema, including formerly completed judgments; source results unchanged.'),source_hashes=dict(source_plan['source_hashes'],instructions=sha(INSTRUCTIONS)))
        atomic(out/'plan.json',plan)
        print(f'MIGRATED: {len(inherited)} completed requests retained; evidence judgments will use NEW inference.',flush=True)
        return plan

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','prompts','run','status','migrate'])
    p.add_argument('--output-dir',type=Path,default=DEFAULT);p.add_argument('--database',type=Path,default=ROOT/'data/corpus.sqlite');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/experiments/remark_filing_validation/fresh-topic30-v2')
    p.add_argument('--model',choices=['nemotron6bit','qwen','qwen14b6bit'],default='nemotron6bit');p.add_argument('--top-k',type=int,default=5);p.add_argument('--max-tokens',type=int,default=1400);p.add_argument('--limit',type=int)
    p.add_argument('--instructions',type=Path);p.add_argument('--examples',type=Path)
    p.add_argument('--exclude-cases',type=Path,action='append',default=[])
    a=p.parse_args(argv)
    if a.top_k<1 or a.max_tokens<1 or (a.limit is not None and a.limit<1):p.error('top-k, max-tokens, limit must be positive')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another process owns this trial')
        if a.action=='migrate':
            migrate(a.source_dir.resolve(),out);report(out);return
        if a.action=='status':
            if not (out/'plan.json').exists():p.error('Prepare the trial first')
            report(out);return
        plan=prepare(out,a.database,a.seed,a.model,a.top_k,a.max_tokens,a.instructions,a.examples,a.exclude_cases)
        print(json.dumps(dict(cases=plan['query_count'],issuers=plan['issuer_count'],sampling_topics=plan['sampling_topics'])),flush=True)
        if a.action in ['prepare','prompts']:
            if a.action=='prompts':atomic(out/'request_prompts.json',[dict(case_id=c['case_id'],prompt=request_prompt(out,c)) for c in load(out/'cases.json')])
            print('READY: frozen inputs; no model loaded or inference started.');return
        try:result=run(out,plan,a.limit)
        except KeyboardInterrupt:report(out);print('PAUSED: rerun the same command to resume.');return
        unfinished=any(k.startswith(('pending_','failed_')) for k in result['counts'])
        print('INCOMPLETE: rerun the same command to resume/retry.' if unfinished else 'DONE: model judgments saved for human review; no training started.')
if __name__=='__main__':main()
