"""Token-bounded search windows; rank original parent passages by maximum score."""
import hashlib
import json
import time
import numpy as np

ENCODING="token-windows-parent-max-v1"
OVERLAP=32


def token_windows(text,tokenizer,budget,overlap=OVERLAP):
    if budget<1 or overlap<0 or overlap>=budget:raise ValueError("Invalid token budget/overlap")
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    offsets=encoded["offset_mapping"]
    if not offsets:return []
    windows=[];start=0
    while start<len(offsets):
        end=min(start+budget,len(offsets))
        # Re-tokenization at a substring boundary can alter token counts.
        while end>start:
            lo=offsets[start][0];hi=offsets[end-1][1];fragment=text[lo:hi]
            if len(tokenizer.encode(fragment,add_special_tokens=False,truncation=False))<=budget:break
            end-=1
        if end==start:raise ValueError("One source token exceeds the window budget")
        windows.append(dict(text=fragment,char_start=lo,char_end=hi,token_start=start,token_end=end))
        if end==len(offsets):break
        start=max(start+1,end-overlap)
    return windows


def parent_scores(windows,scores):
    values=np.asarray(scores,dtype=float)
    if values.shape!=(len(windows),) or not np.isfinite(values).all():raise ValueError("Invalid window scores")
    best={}
    for i,(window,score) in enumerate(zip(windows,values)):
        parent=window["parent_id"]
        if parent not in best or score>best[parent][0]:best[parent]=(float(score),i)
    order=sorted(best,key=lambda p:(-best[p][0],p))
    return [(p,best[p][0],best[p][1]) for p in order]


def pair_windows(question,text,tokenizer,max_length):
    question_tokens=len(tokenizer.encode(question,add_special_tokens=False,truncation=False))
    budget=max_length-question_tokens-tokenizer.num_special_tokens_to_add(pair=True)
    if budget<1:raise ValueError("Question leaves no reranker passage token budget")
    windows=token_windows(text,tokenizer,budget,overlap=min(OVERLAP,budget-1))
    for window in windows:
        pair=tokenizer(question,window["text"],truncation=False,add_special_tokens=True)
        if len(pair["input_ids"])>max_length:raise ValueError("Reranker pair exceeds maximum length")
    return windows


def run_stage(out,plan):
    from sentence_transformers import SentenceTransformer,CrossEncoder
    from .finrank_summary_roundtrip import done,load_corpus,source_rank,append
    questions=done(out/"questions.jsonl");finished=done(out/"retrieval.jsonl")
    cases=[c for c in plan["cases"] if questions.get(c["id"],{}).get("output",{}).get("status")=="ready" and c["id"] not in finished]
    if not cases:return
    if plan.get("retrieval_encoding")!=ENCODING:raise ValueError("Unwindowed frozen plan: use the new windowed evaluation directory")
    docs=load_corpus(out,plan);doc_by_id={d["id"]:d for d in docs}
    print(f"Loading MPNet; full coverage of {len(docs)} parent passages, max-window parent scoring.",flush=True)
    encoder=SentenceTransformer(plan["mpnet"],local_files_only=True,device="cpu")
    tokenizer=encoder.tokenizer;max_tokens=encoder.max_seq_length
    budget=max_tokens-tokenizer.num_special_tokens_to_add(pair=False)
    windows=[]
    for doc in docs:
        chunks=token_windows(doc["text"],tokenizer,budget,plan["window_overlap_tokens"])
        for i,chunk in enumerate(chunks):windows.append(dict(chunk,parent_id=doc["id"],window_index=i))
        if not chunks:raise ValueError(f"Empty tokenized passage {doc['id']}")
    window_hash=hashlib.sha256(json.dumps(windows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    metadata=dict(encoding=ENCODING,window_hash=window_hash,model=plan["mpnet"],max_tokens=max_tokens,
                  parent_count=len(docs),window_count=len(windows),overlap=plan["window_overlap_tokens"])
    manifest=out/"mpnet_windows.json"
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:raise ValueError("Window/tokenizer configuration changed")
    manifest.write_text(json.dumps(metadata,indent=2))
    # Keep offsets/parent links for inspection, without replacing corpus records.
    layout=out/"window_layout.jsonl"
    if not layout.exists():layout.write_text("".join(json.dumps({k:v for k,v in w.items() if k!='text'})+"\n" for w in windows))
    print(f"MPNet windows={len(windows)}; model maximum={max_tokens}; text budget={budget}.",flush=True)
    dimension=encoder.get_sentence_embedding_dimension();cache=out/"mpnet_windows.npy";progress=out/"mpnet_windows_progress.json"
    if cache.exists() and progress.exists():
        state=json.loads(progress.read_text())
        if state["window_hash"]!=window_hash:raise ValueError("Window cache mismatch")
        vectors=np.load(cache,mmap_mode="r+");start_index=state["completed"]
        if vectors.shape!=(len(windows),dimension) or not 0<=start_index<=len(windows):raise ValueError("Invalid window cache")
    else:
        vectors=np.lib.format.open_memmap(cache,mode="w+",dtype=np.float32,shape=(len(windows),dimension));start_index=0
    from .mpnet_embedding_cache import WindowEmbeddingCache, encoder_identity, cached_encode
    from .finrank_summary_roundtrip import ROOT
    configuration=dict(encoder_identity(encoder,plan["mpnet"]), encoding=ENCODING,
                       max_tokens=max_tokens, overlap=plan["window_overlap_tokens"])
    shared=WindowEmbeddingCache(ROOT/"data/cache/mpnet/window_embeddings.sqlite",configuration,dimension)
    totals=dict(total=0,hits=0,encoded=0,seconds=0.0)
    try:
        for start in range(start_index,len(windows),64):
            end=min(start+64,len(windows))
            batch,stats=cached_encode([w["text"] for w in windows[start:end]],
                lambda texts: encoder.encode(texts,normalize_embeddings=True,show_progress_bar=False,batch_size=32),shared)
            vectors[start:end]=batch;vectors.flush()
            for key in totals:totals[key]+=stats[key]
            temp=out/"mpnet_windows_progress.tmp";temp.write_text(json.dumps(dict(completed=end,window_hash=window_hash)));temp.replace(progress)
        print(f"MPNet cache={shared.path} hits={totals['hits']} newly_encoded={totals['encoded']} existing_run_windows={start_index}",flush=True)
        (out/"mpnet_cache_usage.json").write_text(json.dumps(dict(totals,cache=str(shared.path),existing_run_windows=start_index),indent=2))
    finally:
        shared.close()
    scorer=CrossEncoder(plan["reranker"],max_length=plan["max_length"],local_files_only=True,device="cpu")
    with (out/"retrieval.jsonl").open("a") as handle:
        for c in cases:
            row=dict(id=c["id"],encoding=ENCODING);start=time.perf_counter()
            try:
                question=questions[c["id"]]["output"]["question"]
                if len(tokenizer.encode(question,add_special_tokens=True,truncation=False))>max_tokens:raise ValueError("Question exceeds MPNet limit")
                query=encoder.encode(question,normalize_embeddings=True,show_progress_bar=False)
                ranked=parent_scores(windows,np.asarray(vectors)@np.asarray(query,dtype=np.float32))
                parent_ids=[p for p,_,_ in ranked];dense_rank=source_rank(parent_ids,c["passage_id"])
                head=ranked[:plan["retrieve_k"]]
                rerank_windows=[]
                for parent,_,_ in head:
                    for i,window in enumerate(pair_windows(question,doc_by_id[parent]["text"],scorer.tokenizer,plan["max_length"])):
                        rerank_windows.append(dict(window,parent_id=parent,window_index=i))
                raw=[float(x) for x in scorer.predict([(question,w["text"]) for w in rerank_windows],show_progress_bar=False)]
                reranked=parent_scores(rerank_windows,raw);reranked_ids=[p for p,_,_ in reranked]
                rerank_by_parent={p:(score,i) for p,score,i in reranked}
                rank=source_rank(reranked_ids,c["passage_id"])
                candidates=[]
                for i,(parent,score,wi) in enumerate(head):
                    rs,ri=rerank_by_parent[parent]
                    candidates.append(dict(passage_id=parent,dense_rank=i+1,dense_score=score,
                        dense_best_window={k:v for k,v in windows[wi].items() if k!='parent_id'},
                        rerank_score=rs,rerank_best_window={k:v for k,v in rerank_windows[ri].items() if k!='parent_id'}))
                row.update(status="completed",question=question,retrieval_source_rank=dense_rank,
                    retrieval_pass=dense_rank is not None and dense_rank<=plan["retrieve_k"],rerank_source_rank=rank,
                    rerank_pass=rank is not None and rank<=plan["rerank_k"],candidates=candidates,
                    reranked_ids=reranked_ids,rerank_window_count=len(rerank_windows))
            except Exception as error:row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row["seconds"]=time.perf_counter()-start;append(handle,row)
            print(f"{c['id'][:12]} parent rank={row.get('retrieval_source_rank')} rerank={row.get('rerank_source_rank')} {row['status']}",flush=True)
            if row["status"]=="failed":print(row["error"],flush=True)
