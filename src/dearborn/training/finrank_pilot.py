"""Resumable 500-pair silver-data pilot using the original Nemotron pipeline."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
from dearborn.finrank_policy import FINRANK_TICKERS,assert_not_finrank_tickers
from dearborn.experiments.finrank_generation_run import ROOT
from dearborn.experiments.finrank_summary_roundtrip import make_plan
from dearborn.experiments.filing_matching_batch import diverse_sample

DEFAULT=ROOT/"data/experiments/finrank_training/pilot500-v1"


def atomic_json(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def canonical_question(text):return " ".join(text.casefold().split())


def accepted_from_run(directory):
    if not (directory/"summary.json").exists():return []
    plan=json.loads((directory/"plan.json").read_text());cases={c["id"]:c for c in plan["cases"]}
    with gzip.open(directory/"corpus.jsonl.gz","rt") as handle:docs={r["id"]:r for r in map(json.loads,handle)}
    output=[]
    for result in json.loads((directory/"summary.json").read_text())["results"]:
        if result["status"]!="accepted":continue
        case=cases[result["id"]];ticker=case["values"]["ticker"].upper();assert_not_finrank_tickers([ticker])
        text=case["values"]["passage_text"];question=result["question"]["question"]
        negatives=[];seen=set()
        for candidate in result["retrieval"]["candidates"]:
            doc=docs[candidate["passage_id"]];other=doc["ticker"].upper()
            if other==ticker or other in FINRANK_TICKERS or doc["text"]==text or doc["text"] in seen:continue
            negatives.append(dict(text=doc["text"],ticker=other,passage_id=doc["id"]));seen.add(doc["text"])
        output.append(dict(id=hashlib.sha256((question+"\0"+text).encode()).hexdigest(),question=question,positive_text=text,ticker=ticker,passage_id=case["passage_id"],negative_candidates=negatives,source_run=str(directory),label_source="original_pipeline_silver_not_human_verified"))
    return output


def collect(out,plan):
    records=[];seen=set();duplicates=0
    directories=[ROOT / p for p in plan["seed_runs"]]+sorted((out/"batches").glob("batch-*")) if (out/"batches").exists() else [ROOT / p for p in plan["seed_runs"]]
    counts=Counter()
    for directory in directories:
        if (directory/"summary.json").exists():counts.update(json.loads((directory/"summary.json").read_text())["counts"])
        for record in accepted_from_run(directory):
            key=canonical_question(record["question"])
            if key in seen:duplicates+=1;continue
            seen.add(key);records.append(record)
    selected=records[:plan["target"]]
    tmp=out/"accepted.tmp";tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected));tmp.replace(out/"accepted.jsonl")
    status=dict(target=plan["target"],accepted_unique=len(selected),remaining=max(0,plan["target"]-len(selected)),available_unique=len(records),duplicate_questions_excluded=duplicates,pipeline_counts=dict(counts),batch_count=len(list((out/"batches").glob("batch-*/plan.json"))),
        phase="ready_for_dataset_preparation" if len(selected)>=plan["target"] else "generating",baseline=plan["baseline"],note="Accepted means pipeline-approved silver labels; measured precision is not implied.")
    reference=ROOT / plan["baseline"]/"results.json"
    if reference.exists():status["baseline_metrics"]={name:value["metrics"] for name,value in json.loads(reference.read_text())["models"].items()}
    if (out/"dataset_manifest.json").exists():status["dataset"]=json.loads((out/"dataset_manifest.json").read_text())
    if (out/"training/status.json").exists():status["training"]=json.loads((out/"training/status.json").read_text())
    if (out/"training/evaluation.json").exists():status["evaluation"]=json.loads((out/"training/evaluation.json").read_text())
    if (out/"retriever/status.json").exists():status["retriever_training"]=json.loads((out/"retriever/status.json").read_text())
    if (out/"retriever/evaluation.json").exists():
        result=json.loads((out/"retriever/evaluation.json").read_text())
        status["retriever_evaluation"]={name:dict(metrics=value["metrics"],delta_percentage_points=value["delta_percentage_points"]) for name,value in result["models"].items()}
    atomic_json(out/"status.json",status);return status


def prepare(out,target=500,count=2000,seed=42):
    if (out/"pilot_plan.json").exists():return json.loads((out/"pilot_plan.json").read_text())
    seedrun=ROOT/"data/experiments/finrank_synthesis/nemotron-original-100-v1";original=json.loads((seedrun/"plan.json").read_text())
    used={c["values"]["remark_text"] for c in original["cases"]}
    with sqlite3.connect((ROOT/"data/qwen_corpus_labels.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        db.row_factory=sqlite3.Row;db.execute("ATTACH DATABASE ? AS corpus",((ROOT/"data/corpus.sqlite").as_uri()+"?mode=ro",))
        rows=[dict(r) for r in db.execute("""SELECT r.remark_chunk_id AS remark_id,r.remark_text,r.remark_hash,r.ticker,r.call_id,f.accession_number,f.form,f.fiscal_year,f.fiscal_period,c.name FROM remark_label r JOIN corpus.earnings_call e ON e.call_id=r.call_id JOIN corpus.filing f ON f.company_quarter_id=e.company_quarter_id JOIN corpus.company c ON c.cik=f.cik WHERE r.run_id='qwen-corpus-v1' AND r.status='completed' AND r.label='yes' AND f.parse_complete=1 ORDER BY r.remark_chunk_id""")]
    frequencies=Counter(r["remark_id"] for r in rows)
    rows=[r for r in rows if frequencies[r["remark_id"]]==1 and r["ticker"].upper() not in FINRANK_TICKERS and r["remark_text"] not in used]
    selected=diverse_sample(rows,min(count,len(rows)),seed)
    for r in selected:
        if hashlib.sha256(r["remark_text"].encode()).hexdigest()!=r["remark_hash"]:raise ValueError("Remark text hash mismatch")
    if not selected:raise ValueError("No eligible labeled remarks")
    out.mkdir(parents=True,exist_ok=True)
    prompts=original["prompts"]
    plan=dict(version="finrank-pilot500-v1",target=target,candidate_count=len(selected),seed=seed,batch_size=100,candidates=selected,seed_runs=[str(seedrun)],prompts=prompts,baseline=str(ROOT/"data/experiments/finrank_quantitative_baseline"),protocol="Original Nemotron6bit summary/question, MPNet top50/MS MARCO top5, summary-claim JEV; no new context or final-question gate",label_run="qwen-corpus-v1",blocked_tickers=sorted(FINRANK_TICKERS))
    atomic_json(out/"pilot_plan.json",plan);return plan


def prepare_batch(out,plan,index,snapshot,jev_package):
    directory=out/"batches"/f"batch-{index:03d}"
    if (directory/"plan.json").exists():return directory
    from rank_bm25 import BM25Okapi
    from dearborn.experiments.filing_matching_development import tokenize
    rows=plan["candidates"][index*plan["batch_size"]:(index+1)*plan["batch_size"]]
    pairs=[];cache={}
    with sqlite3.connect((ROOT/"data/corpus.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        for row in rows:
            accession=row["accession_number"]
            if accession not in cache:
                chunks=db.execute("SELECT chunk_id,text FROM filing_chunk WHERE accession_number=? ORDER BY chunk_id",(accession,)).fetchall()
                if not chunks:raise ValueError("Frozen filing lost parsed chunks")
                cache[accession]=(chunks,BM25Okapi([tokenize(t) for _,t in chunks]))
            chunks,bm25=cache[accession];scores=bm25.get_scores(tokenize(row["remark_text"]))
            top=max(range(len(chunks)),key=lambda i:float(scores[i]));key,text=chunks[top]
            values=dict(company_name=row["name"],ticker=row["ticker"],call_period=row["call_id"],filing_type=row["form"],filing_period=f"{row['fiscal_year']} {row['fiscal_period']}",remark_text=row["remark_text"],passage_text=text)
            pairs.append(dict(passage_id=key,values=values))
    inputs=directory/"inputs";inputs.mkdir(parents=True,exist_ok=True)
    atomic_json(inputs/"run_plan.json",dict(pairs=pairs))
    frozen=make_plan(inputs,directory,100,snapshot,jev_package,50,5,"nemotron6bit");frozen["prompts"]=plan["prompts"]
    atomic_json(directory/"plan.json",frozen);return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["plan","run","status"]);parser.add_argument("--output-dir",type=Path,default=DEFAULT);parser.add_argument("--max-batches",type=int,help="Limit batches handled by this invocation")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--jev-package", type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/"pilot.lock").open("w") as lock:
        # Status is a snapshot command and does not contend with a running worker.
        if args.action=="status":
            print((out/"status.json").read_text() if (out/"status.json").exists() else "No pilot status yet.");return 0
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(out);status=collect(out,plan)
        print(json.dumps(status,indent=2),flush=True)
        if args.action=="plan":print(f"Frozen {plan['candidate_count']} remarks; no models loaded.");return 0
        if args.max_batches is not None and args.max_batches<1:parser.error("max-batches must be positive")
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH",""));handled=0
        try:
            for index in range((len(plan["candidates"])+99)//100):
                if status["remaining"]==0:break
                directory=out/"batches"/f"batch-{index:03d}"
                summary=directory/"summary.json"
                if summary.exists() and not any(k.startswith("pending") or (k.startswith("failed") and k!="failed_generation_skipped") for k in json.loads(summary.read_text())["counts"]):
                    continue
                if not args.snapshot or not args.jev_package:
                    parser.error("New inference requires --snapshot and --jev-package")
                snapshot=args.snapshot.resolve()
                jev_package=args.jev_package.resolve()
                directory=prepare_batch(out,plan,index,snapshot,jev_package)
                command=[sys.executable,"-u","-m","dearborn.experiments.finrank_summary_roundtrip","--source-dir",str(directory/"inputs"),"--output-dir",str(directory),"--snapshot",str(snapshot),"--jev-package",str(jev_package),"--generator","nemotron6bit","--limit","100","--retrieve-k","50","--rerank-k","5","--skip-generation-failures"]
                subprocess.run(command,check=True,cwd=ROOT,env=env);status=collect(out,plan)
                print(f"PILOT accepted={status['accepted_unique']}/{plan['target']} remaining={status['remaining']} batch={index+1}",flush=True)
                handled+=1
                if args.max_batches and handled>=args.max_batches:break
        except (KeyboardInterrupt,subprocess.CalledProcessError):collect(out,plan);print("PAUSED/STOPPED: rerun the same command to resume.");return 1
        if status["remaining"]==0:print("DONE: 500 silver pairs ready; no training started.");return 0
        print("INCOMPLETE: target not yet met; resume or review candidate exhaustion.");return 1


if __name__=="__main__":raise SystemExit(main())
