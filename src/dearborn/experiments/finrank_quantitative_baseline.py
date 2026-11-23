"""Original FinRank retrieval protocol, quantitative queries, unchanged full corpus."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from dearborn.finrank import RECORDS,TEXT,INDEX,per_record_metrics,aggregate,tokenize
from .finrank_generation_run import ROOT,append

DENSE="sentence-transformers/all-mpnet-base-v2"
CROSS="cross-encoder/ms-marco-MiniLM-L-6-v2"


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def prepare(out):
    records=[r for r in RECORDS if r["reasoning_type"]=="Quantitative"]
    manifest=dict(version="finrank-quantitative-baseline-v1",dataset_sha256=hashlib.sha256((ROOT/"data/FinRank/FinRank.jsonl").read_bytes()).hexdigest(),
        query_ids=[r["id"] for r in records],query_sha256=digest([r["question"] for r in records]),query_count=len(records),corpus_count=len(TEXT),corpus_sha256=digest(TEXT),
        corpus_scope="Original full FinRank first-occurrence exact-text corpus: positives and hard negatives from ALL records",
        models=dict(mpnet=DENSE,reranker=CROSS),rerank_top_k=20,query_rewrites=False,metadata_filters=False,windowing=False,
        tokenizer=r"Original baseline tokenize regex \w+ lowercase",tfidf="TfidfVectorizer(sublinear_tf=True), other defaults",bm25="BM25Okapi defaults",dense="SentenceTransformer default max_seq_length; normalized vectors and cosine",metrics="Original per_record_metrics; macro averages multiplied by 100; same np.argsort tie behavior",
        topic_counts=dict(Counter(r["topic"] for r in records)),note="Quantitative includes direct lookup, calculations and interpretation. No training or gold-answer generation.")
    out.mkdir(parents=True,exist_ok=True);path=out/"run_manifest.json"
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError("Frozen baseline configuration changed; use a new directory")
    path.write_text(json.dumps(manifest,indent=2))
    frozen=out/"quantitative_records.jsonl"
    if not frozen.exists():frozen.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    golds=[{INDEX[p["text"]] for p in r["passages"] if p["text"].strip()} for r in records]
    return records,golds,manifest


def metrics_for(rankings,records,golds):
    rows=[dict(id=r["id"],**per_record_metrics(rankings[i],golds[i])) for i,r in enumerate(records)]
    names=[k for k in rows[0] if k!="id"]
    totals={k:aggregate(rows,k) for k in names}
    strata={}
    for field in ["topic","difficulty","num_passages","industry"]:
        strata[field]={}
        for value in sorted({str(r[field]) for r in records}):
            group=[m for m,r in zip(rows,records) if str(r[field])==value]
            strata[field][value]=dict(n=len(group),**{k:aggregate(group,k) for k in names})
    return dict(metrics=totals,strata=strata,per_question=rows)


def atomic_array(path,array):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as handle:np.save(handle,array)
    tmp.replace(path)


def run(out,records,golds,lexical_only=False,device=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rank_bm25 import BM25Okapi
    queries=[r["question"] for r in records];rankings={};timings={}
    for method in ["tfidf","bm25"]:
        path=out/(method+"_rankings.npy");started=time.perf_counter()
        if path.exists():ranking=np.load(path)
        else:
            print(f"Scoring {method}: {len(queries)} questions against {len(TEXT)} passages",flush=True)
            if method=="tfidf":
                vectorizer=TfidfVectorizer(sublinear_tf=True);corpus=vectorizer.fit_transform(TEXT)
                scores=(vectorizer.transform(queries)@corpus.T).toarray()
            else:
                bm25=BM25Okapi([tokenize(t) for t in TEXT]);scores=np.stack([bm25.get_scores(tokenize(q)) for q in queries])
            ranking=np.argsort(-scores,axis=1);atomic_array(path,ranking)
        rankings[method]=ranking;timings[method]=time.perf_counter()-started
    if not lexical_only:
        from sentence_transformers import SentenceTransformer,CrossEncoder
        path=out/"dense_rankings.npy";started=time.perf_counter()
        if path.exists():rankings["dense"]=np.load(path)
        else:
            print("Loading cached MPNet; original prefix truncation (no windowing)",flush=True)
            kwargs=dict(local_files_only=True)
            if device:kwargs["device"]=device
            encoder=SentenceTransformer(DENSE,**kwargs)
            (out/"mpnet_runtime.json").write_text(json.dumps(dict(model=DENSE,max_seq_length=encoder.max_seq_length,device=str(encoder.device)),indent=2))
            cache=out/"corpus_embeddings.npy"
            if cache.exists():corpus=np.load(cache)
            else:
                corpus=encoder.encode(TEXT,batch_size=64,normalize_embeddings=True,show_progress_bar=True);atomic_array(cache,corpus)
            query=encoder.encode(queries,batch_size=64,normalize_embeddings=True,show_progress_bar=True)
            ranking=np.argsort(-(query@corpus.T).astype(np.float32),axis=1);atomic_array(path,ranking);rankings["dense"]=ranking
            del encoder,corpus,query;gc.collect()
        timings["dense"]=time.perf_counter()-started
        path=out/"cross_encoder.jsonl";completed={}
        if path.exists():completed={r["id"]:r for r in map(json.loads,path.read_text().splitlines()) if r["status"]=="completed"}
        pending=[i for i,r in enumerate(records) if r["id"] not in completed]
        if pending:
            kwargs=dict(local_files_only=True)
            if device:kwargs["device"]=device
            scorer=CrossEncoder(CROSS,**kwargs)
            (out/"reranker_runtime.json").write_text(json.dumps(dict(model=CROSS,max_length=scorer.max_length),indent=2))
            with path.open("a") as handle:
                for i in pending:
                    started=time.perf_counter();head=rankings["dense"][i][:20]
                    scores=scorer.predict([(queries[i],TEXT[j]) for j in head],show_progress_bar=False)
                    row=dict(id=records[i]["id"],status="completed",head=head[np.argsort(-scores,kind="stable")].tolist(),seconds=time.perf_counter()-started)
                    append(handle,row);completed[row["id"]]=row
                    if (i+1)%20==0 or i==pending[-1]:print(f"MS MARCO completed={len(completed)}/{len(records)}",flush=True)
        rankings["ce_dense"]=np.stack([np.concatenate([completed[r["id"]]["head"],rankings["dense"][i][20:]]) for i,r in enumerate(records)])
        timings["ce_dense_inference_seconds"]=sum(r["seconds"] for r in completed.values())
    result=dict(query_count=len(records),corpus_count=len(TEXT),models={method:metrics_for(ranking,records,golds) for method,ranking in rankings.items()},current_invocation_seconds=timings)
    (out/("lexical_results.json" if lexical_only else "results.json")).write_text(json.dumps(result,indent=2))
    for method,value in result["models"].items():print(method,json.dumps(value["metrics"]),flush=True)
    print("DONE: results saved to",out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,default=ROOT/"data/experiments/finrank_quantitative_baseline")
    parser.add_argument("--plan-only",action="store_true");parser.add_argument("--lexical-only",action="store_true");parser.add_argument("--device",choices=["cpu","mps","cuda"])
    args=parser.parse_args();out=args.output_dir.resolve()
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/"runner.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);records,golds,manifest=prepare(out)
        print(f"Frozen quantitative queries={len(records)}; unchanged full corpus={len(TEXT)}; rerank top20",flush=True)
        if args.plan_only:print("PLAN ONLY: no scoring/model inference.");return
        run(out,records,golds,args.lexical_only,args.device)


if __name__=="__main__":main()
