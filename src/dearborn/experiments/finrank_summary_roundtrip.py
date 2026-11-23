"""Summary→question→MPNet→MS MARCO→JEV pilot, with isolated resumable stages."""
import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel,ConfigDict,Field
from .finrank_generation_run import ROOT,read,append
from .finrank_generation_cycle import freeze
from dearborn.finrank_policy import FINRANK_TICKERS,assert_not_finrank_tickers

MPNET="sentence-transformers/all-mpnet-base-v2"
RERANKER="cross-encoder/ms-marco-MiniLM-L-6-v2"


class Statement(BaseModel):
    model_config=ConfigDict(extra="forbid")
    text:str
    evidence_quotes:list[str]


class EvidenceSummary(BaseModel):
    model_config=ConfigDict(extra="forbid")
    status:Literal["ready","declined"]
    statements:list[Statement]=Field(max_length=6)
    reason:str


class GeneratedQuestion(BaseModel):
    model_config=ConfigDict(extra="forbid")
    status:Literal["ready","declined"]
    question:str
    reason:str


def validate_summary(raw):
    result=EvidenceSummary.model_validate_json(raw)
    if result.status=="ready":
        if not 1<=len(result.statements)<=6:raise ValueError("Ready summary needs 1–6 statements")
        if any(not s.text.strip() or not s.evidence_quotes or any(not q.strip() for q in s.evidence_quotes) for s in result.statements):raise ValueError("Empty summary evidence")
    elif result.statements or not result.reason.strip():raise ValueError("Invalid decline")
    return result


def validate_question(raw):
    result=GeneratedQuestion.model_validate_json(raw)
    if result.status=="ready" and not result.question.strip():raise ValueError("Empty question")
    if result.status=="declined" and (result.question.strip() or not result.reason.strip()):raise ValueError("Invalid decline")
    return result


def latest(file,key="id"):
    return {r[key]:r for r in read(file)}


def done(file,key="id"):
    return {k:r for k,r in latest(file,key).items() if r["status"]=="completed"}


def source_rank(ids,source):
    return ids.index(source)+1 if source in ids else None


def ranked_indices(scores,ids):
    values=np.asarray(scores,dtype=float)
    if values.shape!=(len(ids),) or not np.isfinite(values).all():raise ValueError("Invalid ranking scores")
    return sorted(range(len(ids)),key=lambda i:(-values[i],ids[i]))


def make_plan(source,out,limit,snapshot,package,retrieve_k,rerank_k,generator="nemotron"):
    original=json.loads((source/"run_plan.json").read_text())
    groups={}
    for pair in original["pairs"]:
        key=pair["passage_id"]
        assert_not_finrank_tickers([pair["values"]["ticker"]])
        if key not in groups:groups[key]=dict(id=hashlib.sha256(key.encode()).hexdigest(),passage_id=key,values=pair["values"],remarks=[])
        if pair["values"]["remark_text"] not in groups[key]["remarks"]:groups[key]["remarks"].append(pair["values"]["remark_text"])
    if not groups:raise ValueError("No matched inputs")
    cases=list(groups.values())[:limit] if limit else list(groups.values())
    accessions=sorted({key.split(":")[0] for key in groups})
    with sqlite3.connect((ROOT/"data/corpus.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        query="SELECT x.chunk_id,x.text,c.ticker FROM filing_chunk x JOIN filing f ON f.accession_number=x.accession_number JOIN company c ON c.cik=f.cik WHERE x.accession_number IN ("+",".join("?" for _ in accessions)+") ORDER BY x.chunk_id"
        docs=[dict(id=i,text=t,ticker=k) for i,t,k in db.execute(query,accessions) if k.upper() not in FINRANK_TICKERS]
    corpus={d["id"]:d for d in docs}
    for case in cases:
        if case["passage_id"] not in corpus or corpus[case["passage_id"]]["text"]!=case["values"]["passage_text"]:raise ValueError("Source passage absent or changed in corpus")
    digest=hashlib.sha256()
    temporary=out/"corpus.jsonl.gz.tmp"
    with gzip.open(temporary,"wt") as handle:
        for d in docs:
            line=json.dumps(d,sort_keys=True,ensure_ascii=False)+"\n";digest.update(line.encode());handle.write(line)
    temporary.replace(out/"corpus.jsonl.gz")
    return dict(version="summary-roundtrip-v1",source=str(source.resolve()),limit=limit,snapshot=str(snapshot),jev_package=str(package),
        retrieve_k=retrieve_k,rerank_k=rerank_k,mpnet=MPNET,reranker=RERANKER,max_length=512,
        corpus_count=len(docs),corpus_sha256=digest.hexdigest(),corpus_scope="all_chunks_from_all_source_filings_unfiltered_query",
        retrieval_encoding="token-windows-parent-max-v1",window_overlap_tokens=32,
        generator=generator,generation_tokens=1024,prompts=json.loads((ROOT/"prompts/finrank_summary_roundtrip_v2.json").read_text()),cases=cases)


def load_corpus(out,plan):
    digest=hashlib.sha256();docs=[]
    with gzip.open(out/"corpus.jsonl.gz","rt") as handle:
        for line in handle:digest.update(line.encode());docs.append(json.loads(line))
    if digest.hexdigest()!=plan["corpus_sha256"] or len(docs)!=plan["corpus_count"]:raise ValueError("Frozen corpus changed")
    return docs


def summary_input(case, plan):
    values=case["values"]
    data=dict(company=values["company_name"],filing_type=values["filing_type"],
              filing_period=values["filing_period"],passage=values["passage_text"])
    if plan.get("summary_input_mode", "remark_and_passage")=="remark_and_passage":
        data["remarks"]=case["remarks"]
    elif plan["summary_input_mode"]!="passage_only":
        raise ValueError("Unknown summary input mode")
    if plan.get("include_filing_context"):
        data["filing_context"]=case["filing_context"]
    return data


def skippable_generation_failures(out):
    failures={}
    for stage,filename in [("summary","summaries.jsonl"),("question","questions.jsonl")]:
        for sid,row in latest(out/filename).items():
            if row["status"]=="failed" and row.get("error", "").startswith(("ValidationError:","JSONDecodeError:","ValueError:")):
                failures[sid]=dict(stage=stage,error=row["error"])
    return failures


def generate_stage(out,plan):
    summaries=done(out/"summaries.jsonl");questions=done(out/"questions.jsonl")
    cases=[c for c in plan["cases"] if c["id"] not in summaries or (summaries[c["id"]]["output"]["status"]=="ready" and c["id"] not in questions)]
    if plan.get("skip_generation_failures"):
        skipped=skippable_generation_failures(out)
        cases=[c for c in cases if c["id"] not in skipped]
        if skipped:print(f"Skipping {len(skipped)} saved malformed generation records; retaining errors for review.",flush=True)
    if not cases:return
    from .finrank_ragas_qwen import StructuredQwen
    import mlx.core as mx
    from .jev_matching_diagnostic import snapshot
    print(f"Loading {plan['generator']} for {len(cases)} passage-summary/question tasks",flush=True)
    generate=StructuredQwen(plan["generation_tokens"],model_name=plan["generator"])
    for c in cases:
        for stage,schema,validator,filename in [("summary",EvidenceSummary,validate_summary,"summaries.jsonl"),("question",GeneratedQuestion,validate_question,"questions.jsonl")]:
            if stage=="summary" and c["id"] in summaries:continue
            if stage=="question":
                prior=summaries.get(c["id"])
                if not prior or prior["output"]["status"]!="ready" or c["id"] in questions:continue
                data=dict(company=c["values"]["company_name"],summary=[s["text"] for s in prior["output"]["statements"]])
            else:
                data=summary_input(c,plan)
            prompt=plan["prompts"][stage]+"\nCURRENT INPUT:\n"+json.dumps(data,ensure_ascii=False)
            row=dict(id=c["id"],prompt=prompt);start=time.perf_counter();mx.reset_peak_memory()
            try:
                raw=generate(prompt,schema);row["raw_response"]=raw;result=validator(raw)
                row.update(status="completed",output=result.model_dump())
            except Exception as error:row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row.update(seconds=time.perf_counter()-start,memory=snapshot(mx))
            with (out/filename).open("a") as handle:append(handle,row)
            if row["status"]=="completed":(summaries if stage=="summary" else questions)[c["id"]]=row
            print(f"{c['id'][:12]} {stage}: {row.get('output',{}).get('status',row['status'])}",flush=True)
            if row["status"]=="failed":print(row["error"],flush=True)
            gc.collect();mx.clear_cache()


def retrieval_stage(out,plan):
    from .filing_window_retrieval import run_stage
    return run_stage(out,plan)


def jev_stage(out,plan):
    summaries=done(out/"summaries.jsonl");retrieval=done(out/"retrieval.jsonl");completed=done(out/"verification.jsonl",key="key")
    tasks=[]
    for c in plan["cases"]:
        match=retrieval.get(c["id"],{})
        if not match.get("retrieval_pass") or not match.get("rerank_pass"):continue
        for i,s in enumerate(summaries[c["id"]]["output"]["statements"]):
            key=f"{c['id']}:{i}"
            if key not in completed:tasks.append((c,i,s,key))
    if not tasks:return
    import mlx.core as mx
    import jev27b_mlx
    from .jev_matching_diagnostic import snapshot
    print(f"Loading JEV for {len(tasks)} summary statements from retrieval/rerank survivors",flush=True)
    model=jev27b_mlx.load(plan["snapshot"])
    question=model.question("choice",plan["prompts"]["jev"],["pass","fail"])
    with (out/"verification.jsonl").open("a") as handle:
        for c,i,s,key in tasks:
            row=dict(id=c["id"],key=key,statement_index=i,statement=s["text"]);start=time.perf_counter();mx.reset_peak_memory()
            try:
                # Summary claims are tested individually; no other statements or generated answer.
                state=dict(company=c["values"]["company_name"],filing_period=c["values"]["filing_period"],passage=c["values"]["passage_text"],statement=s["text"])
                result=model.decide(json.dumps(state,ensure_ascii=False),question)
                if result["choice"] not in {"pass","fail"}:raise ValueError("Invalid JEV verdict")
                row.update(status="completed",result=result)
            except Exception as error:row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row.update(seconds=time.perf_counter()-start,memory=snapshot(mx));append(handle,row)
            print(f"{key} JEV: {row.get('result',{}).get('choice','failed')}",flush=True)
            gc.collect();mx.clear_cache()


def case_status(c,summaries,questions,retrieval,verification):
    sid=c["id"];summary=summaries.get(sid)
    if not summary:return "pending_summary"
    if summary["output"]["status"]=="declined":return "declined_summary"
    question=questions.get(sid)
    if not question:return "pending_question"
    if question["output"]["status"]=="declined":return "declined_question"
    rank=retrieval.get(sid)
    if not rank:return "pending_retrieval"
    if not rank["retrieval_pass"]:return "rejected_retrieval"
    if not rank["rerank_pass"]:return "rejected_rerank"
    verdicts=[verification.get(f"{sid}:{i}") for i in range(len(summary["output"]["statements"]))]
    if any(v is None for v in verdicts):return "pending_verification"
    return "accepted" if all(v["result"]["choice"]=="pass" for v in verdicts) else "rejected_verification"


def report(out,plan):
    skipped=skippable_generation_failures(out) if plan.get("skip_generation_failures") else {}
    summaries=done(out/"summaries.jsonl");questions=done(out/"questions.jsonl");ranks=done(out/"retrieval.jsonl");verification=done(out/"verification.jsonl",key="key")
    results=[]
    for c in plan["cases"]:
        results.append(dict(id=c["id"],passage_id=c["passage_id"],ticker=c["values"]["ticker"],status=case_status(c,summaries,questions,ranks,verification),
            summary=summaries.get(c["id"],{}).get("output"),question=questions.get(c["id"],{}).get("output"),retrieval=ranks.get(c["id"])))
    for row in results:
        if row["id"] in skipped:
            row["status"]="failed_generation_skipped"
            row["generation_error"]=skipped[row["id"]]
    if skipped:(out/"generation_failures.json").write_text(json.dumps(skipped,indent=2))
    counts={status:sum(r["status"]==status for r in results) for status in sorted({r["status"] for r in results})}
    metrics={}
    ready=[r for r in ranks.values()]
    for k in [1,2,3,4,5,10,20]:
        metrics[f"source_retrieval_at_{k}"]=sum(r["retrieval_source_rank"] is not None and r["retrieval_source_rank"]<=k for r in ready)
    summary=dict(planned=len(results),counts=counts,ranked_questions=len(ready),retrieval_counts=metrics,results=results,
        note="Pilot rank cutoffs are provisional. JEV verifies summary claims, not complete question validity. Accepted pairs still require human audit.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(counts),flush=True)
    return any(k.startswith("pending") for k in counts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--jev-package",type=Path,default=Path.home()/".cache/uv/git-v0/checkouts/2836f55b3e3abb68/5ef5405")
    parser.add_argument("--limit",type=int,default=3);parser.add_argument("--retrieve-k",type=int,default=50);parser.add_argument("--rerank-k",type=int,default=5)
    parser.add_argument("--generator",choices=["nemotron","nemotron6bit","qwen14b6bit"],default="nemotron")
    parser.add_argument("--generation-only",action="store_true",help="Run only summary and question generation for the precision comparison")
    parser.add_argument("--skip-generation-failures",action="store_true",help="Exclude malformed generation records and continue valid examples; errors remain logged")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--stage",choices=["run","generate","retrieve","jev"],default="run",help=argparse.SUPPRESS)
    a=parser.parse_args(argv);source=a.source_dir.resolve();out=a.output_dir.resolve();snap=a.snapshot.resolve();package=a.jev_package.resolve()
    if min(a.limit,a.retrieve_k,a.rerank_k)<1 or a.rerank_k>a.retrieve_k:parser.error("Invalid limits/ranks")
    if a.stage!="run":
        plan=json.loads((out/"plan.json").read_text());plan["skip_generation_failures"]=a.skip_generation_failures
        plan.update(snapshot=str(snap), jev_package=str(package))
        dict(generate=generate_stage,retrieve=retrieval_stage,jev=jev_stage)[a.stage](out,plan);return 0
    out.mkdir(parents=True,exist_ok=True)
    with (out/"runner.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/"plan.json"
        if path.exists():
            plan=json.loads(path.read_text())
            if (str((ROOT / plan["source"]).resolve()),plan["limit"],plan.get("snapshot",str(snap)),plan.get("jev_package",str(package)),plan["retrieve_k"],plan["rerank_k"])!=(str(source),a.limit,str(snap),str(package),a.retrieve_k,a.rerank_k):parser.error("Resume options differ; use original options or a new directory")
            if plan["generator"]!=a.generator:parser.error("Resume generator differs; use a new directory for a precision comparison")
        else:freeze(path,make_plan(source,out,a.limit,snap,package,a.retrieve_k,a.rerank_k,a.generator));plan=json.loads(path.read_text())
        plan["skip_generation_failures"]=a.skip_generation_failures
        print(f"Frozen {len(plan['cases'])} matched inputs; corpus={plan['corpus_count']} chunks; MPNet top {plan['retrieve_k']} → MS MARCO top {plan['rerank_k']} → JEV.",flush=True)
        if a.plan_only:print("PLAN ONLY: no models loaded.");return 0
        if not snap.is_dir() or not package.is_dir():parser.error("Local JEV snapshot/package missing")
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH",""))
        base=["-u","-m","dearborn.experiments.finrank_summary_roundtrip","--source-dir",str(source),"--output-dir",str(out),"--snapshot",str(snap),"--jev-package",str(package)]
        if a.skip_generation_failures:base.append("--skip-generation-failures")
        commands=[[sys.executable,*base,"--stage",stage] for stage in ["generate","retrieve"]]
        commands.append(["uv","run","--no-project","--with",str(package),"python",*base,"--stage","jev"])
        if a.generation_only:commands=commands[:1]
        try:
            for command in commands:
                subprocess.run(command,check=True,cwd=ROOT,env=env)
                if report(out,plan) and command is commands[0]:
                    if any(v["status"]=="failed" for v in latest(out/"summaries.jsonl").values()) or any(v["status"]=="failed" for v in latest(out/"questions.jsonl").values()):
                        failed={sid for filename in ["summaries.jsonl","questions.jsonl"] for sid,row in latest(out/filename).items() if row["status"]=="failed"}
                        if not a.skip_generation_failures or failed-set(skippable_generation_failures(out)):
                            print("STOPPED: generation errors; rerun same command to retry.");return 1
                        print("Continuing valid examples; malformed generations excluded and logged.",flush=True)
        except subprocess.CalledProcessError as error:report(out,plan);print("STOPPED: rerun same command to resume.");return error.returncode
        except KeyboardInterrupt:report(out,plan);print("PAUSED: rerun same command to resume.");return 130
        if a.generation_only:
            report(out,plan)
            print("DONE: summary/question comparison saved; retrieval and JEV not run.");return 0
        if report(out,plan):print("INCOMPLETE: rerun same command to retry failures.");return 1
        print("DONE: summary round-trip candidates and all stage diagnostics saved.");return 0


if __name__=="__main__":raise SystemExit(main())
