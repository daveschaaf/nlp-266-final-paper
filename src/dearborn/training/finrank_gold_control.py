"""Gold-data positive control on FinRank's official issuer-disjoint split.

Deliberately permits FinRank TRAIN issuers only for this isolated supervised control.
"""
import argparse
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from . import finrank_ablation as engine
from .finrank_hard_negatives import normalized
from .finrank_metadata_training import sha
from .finrank_pilot import atomic_json
from dearborn.finrank import RECORDS,TEXT,INDEX
from dearborn.experiments.finrank_source_recovery import source_labels,assess
from dearborn.experiments.finrank_question_structure import ALIASES
from dearborn.experiments.finrank_eda import table
from dearborn.experiments.finrank_generation_run import ROOT
from dearborn.experiments.finrank_quantitative_baseline import digest,atomic_array

DEFAULT=ROOT/'data/experiments/finrank_training/gold-control474-v1'
SPLITS=ROOT/'data/FinRank/baselines/splits.json'


def identity(row):
    return (row['ticker'].upper(),normalized(row['question']))


def build_data(records,splits,texts,index,labels,pair_count=474,seed=42):
    by_id={r['id']:r for r in records};names={k:v[0] for k,v in ALIASES.items()}
    issuer_sets={s:{by_id[i]['ticker'].upper() for i in splits[s]} for s in ['train','dev','test']}
    if any(issuer_sets[a]&issuer_sets[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('Issuer split overlaps')
    record_sets={s:set(splits[s]) for s in ['train','dev','test']}
    if any(record_sets[a]&record_sets[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):raise ValueError('Record split overlaps')
    def doc(j):
        sources=[dict(ticker=t,filing_year=str(y),form=f,name=names.get(t,t)) for t,y,f in sorted(labels[j])]
        return dict(passage_id=str(j),corpus_id=j,text=texts[j],**sources[0],sources=sources)
    def indices(items):
        return {index[p['text']] for p in items if p['text'].strip()}
    normalized_issuers=defaultdict(set)
    for j,observed in labels.items():normalized_issuers[normalized(texts[j])].update(t[0] for t in observed)
    safe={s:{j for j,observed in labels.items() if observed and normalized_issuers[normalized(texts[j])]<=issuers} for s,issuers in issuer_sets.items()}
    all_positives=defaultdict(set)
    for r in records:all_positives[identity(r)].update(normalized(p['text']) for p in r['passages'] if p['text'].strip())
    groups={};audit=defaultdict(list)
    for split in ['train','dev']:
        eligible=[]
        for id in sorted(splits[split]):
            r=by_id[id]
            if r['reasoning_type']!='Quantitative':continue
            gold=sorted(indices(r['passages'])&safe[split])
            if not gold:audit[split+':no_split_safe_positive'].append(id);continue
            candidates=sorted(j for j in indices(r['hard_negatives'])&safe[split] if normalized(texts[j]) not in all_positives[identity(r)])
            # Curated gold negatives are data, not inferred relevance from non-gold status.
            rng=random.Random(int(hashlib.sha256(f'{seed}:{id}'.encode()).hexdigest(),16));rng.shuffle(candidates)
            negatives=[];seen=set()
            for j in candidates:
                if normalized(texts[j]) in seen:continue
                negatives.append(doc(j));seen.add(normalized(texts[j]))
                if len(negatives)==4:break
            if not negatives:
                audit[split+':no_split_safe_curated_negative'].append(id)
                if split=='train':continue
            eligible.append(dict(record=r,golds=gold,negatives=negatives))
        groups[split]=eligible
    train_groups=groups['train'];first=[];extra=[]
    rng=random.Random(seed)
    for g in train_groups:
        positives=list(g['golds']);rng.shuffle(positives)
        first.append((g,positives[0]));extra.extend((g,j) for j in positives[1:])
    rng.shuffle(first);rng.shuffle(extra)
    chosen=(first+extra)[:pair_count]
    if len(chosen)!=pair_count:raise ValueError(f'Only {len(first)+len(extra)} eligible gold pairs, requested {pair_count}')
    frozen=[]
    def example(g,j,split):
        r=g['record'];key=f'{r["id"]}:{j}'
        return dict(id=hashlib.sha256(key.encode()).hexdigest(),record_id=r['id'],question=r['question'],ticker=r['ticker'].upper(),split=split,positive=doc(j),negatives=g['negatives'],query_group=digest(identity(r)),gold_corpus_ids=g['golds'],provenance='FinRank human gold, official ticker split; no generated text')
    frozen.extend(example(g,j,'train') for g,j in chosen)
    frozen.extend(example(g,g['golds'][0],'validation') for g in groups['dev'])
    # Include every eligible dev-source passage, not just sampled negatives.
    validation=[doc(j) for j in sorted(safe['dev'])];mapping={p['corpus_id']:i for i,p in enumerate(validation)}
    held=[r for r in frozen if r['split']=='validation'];golds=[[mapping[j] for j in r['gold_corpus_ids']] for r in held]
    counts=dict(training_pairs=len(chosen),training_questions=len({g['record']['id'] for g,j in chosen}),validation_questions=len(held),validation_loss_questions=sum(bool(r['negatives']) for r in held),validation_passages=len(validation),test_questions=sum(by_id[i]['reasoning_type']=='Quantitative' for i in splits['test']),negatives_by_split={s:sum(len(r['negatives']) for r in frozen if r['split']==s) for s in ['train','validation']})
    return frozen,validation,golds,dict(counts=counts,issuer_sets={s:sorted(v) for s,v in issuer_sets.items()},excluded=dict(audit))


def prepare(out,pair_count=474,seed=42):
    out.mkdir(parents=True,exist_ok=True)
    request=dict(version='finrank-gold-control-v1',source_sha256=sha(ROOT/'data/FinRank/FinRank.jsonl'),splits_sha256=sha(SPLITS),pair_count=pair_count,seed=seed,rates=[2e-6,2e-7],epochs=3,evaluate_every_steps=25,temperature=.05,retriever_batch_size=2,reranker_batch_size=8,reranker_max_length=512,metadata=True,query_scope='Quantitative only; original human questions',losses='Existing paired InfoNCE, no in-batch negatives; reranker pointwise BCE',negative_policy='Up to four provided curated hard negatives, same official issuer split; exclude all known same-question positives and normalized-text cross-split source overlaps. No assumed fallback negatives.',selection='Best dev MRR including step0, earliest tie; rate per component chosen on dev only, ties prefer lower rate.',caveat='Gold positive and negative supervision replace silver labels; cannot isolate positive quality from negative quality. Training contains multiple gold pairs per query. Previously inspected benchmark; not a fresh untouched holdout.')
    path=out/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['request']!=request:raise ValueError('Frozen control configuration differs')
        for name,value in plan['files'].items():
            if sha(out/name)!=value:raise ValueError('Frozen data changed: '+name)
        return plan
    splits=json.loads(SPLITS.read_text())['ticker'];labels=source_labels(RECORDS,INDEX)
    frozen,pool,golds,audit=build_data(RECORDS,splits,TEXT,INDEX,labels,pair_count,seed)
    engine.write_rows(out/'records.jsonl.gz',frozen);engine.write_rows(out/'validation_pool.jsonl.gz',pool)
    held=[r for r in frozen if r['split']=='validation']
    atomic_json(out/'validation.json',dict(query_ids=[r['id'] for r in held],golds=golds,query_count=len(held),candidate_count=len(pool),candidate_ids=[p['passage_id'] for p in pool]))
    test_ids=[r['id'] for r in RECORDS if r['id'] in set(splits['test']) and r['reasoning_type']=='Quantitative']
    atomic_json(out/'test_queries.json',dict(ids=test_ids,official_split='ticker.test',reasoning_type='Quantitative'))
    files={name:sha(out/name) for name in ['records.jsonl.gz','validation_pool.jsonl.gz','validation.json','test_queries.json']}
    plan=dict(request=request,counts=audit['counts'],files=files,split_audit=audit)
    atomic_json(path,plan);return plan


def select(out):
    plan=json.loads((out/'plan.json').read_text());selected={}
    for kind in ['retriever','reranker']:
        candidates=[]
        for rate in plan['request']['rates']:
            directory=engine.run_dir(out,True,kind,rate);status=json.loads((directory/'status.json').read_text())
            if status['phase']!='trained':raise ValueError('Complete all training runs first')
            candidates.append(dict(rate=rate,mrr=status['best_validation_mrr'],step=status['best_step'],checkpoint=str((directory/'best').resolve())))
        selected[kind]=dict(max(candidates,key=lambda v:(v['mrr'],-v['rate'])),candidates=candidates)
    atomic_json(out/'selection.json',selected);return selected


def evaluate(out,device=None):
    from sentence_transformers import SentenceTransformer
    from dearborn.experiments.finrank_financials_benchmark import load_saved_rankings
    plan=json.loads((out/'plan.json').read_text());chosen=select(out)
    quant,rankings,manifest,*_=load_saved_rankings()
    ids=json.loads((out/'test_queries.json').read_text())['ids'];positions=[i for i,r in enumerate(quant) if r['id'] in set(ids)]
    records=[quant[i] for i in positions]
    if [r['id'] for r in records]!=ids:raise ValueError('Test query IDs/order changed')
    directory=out/'evaluation';directory.mkdir(exist_ok=True)
    protocol=dict(plan_sha256=digest(plan),selection=chosen,checkpoint_hashes={k:engine.checkpoint_hash(Path(v['checkpoint'])) for k,v in chosen.items()},baseline_manifest=manifest,query_ids=ids,corpus_sha256=digest(TEXT),query_count=len(records),corpus_count=len(TEXT),rerank_top_k=20)
    path=directory/'plan.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Evaluation changed; use a new evaluation directory')
    atomic_json(path,protocol)
    labels=source_labels(RECORDS,INDEX)
    from dearborn.retrieval.metadata import enrich_passage
    names={k:v[0] for k,v in ALIASES.items()};texts=[enrich_passage(t,labels[i],names) for i,t in enumerate(TEXT)]
    cache=directory/'dense_rankings.npy'
    if cache.exists():dense=np.load(cache)
    else:
        encoder=SentenceTransformer(chosen['retriever']['checkpoint'],local_files_only=True,device=engine.device_name(device))
        vectors_path=directory/'corpus_embeddings.npy'
        vectors=np.load(vectors_path) if vectors_path.exists() else encoder.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        if not vectors_path.exists():atomic_array(vectors_path,vectors)
        queries=encoder.encode([r['question'] for r in records],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        dense=np.argsort(-(queries@vectors.T).astype(np.float32),axis=1);atomic_array(cache,dense)
        del encoder,vectors,queries;engine.cleanup()
    ranked=engine.rerank(directory,'gold_control',chosen['reranker']['checkpoint'],dense,records,texts,device)
    atomic_array(directory/'rankings.npy',ranked)
    golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    models={name:assess(rankings[key][positions],records,golds,labels) for name,key in [('Original baseline','Original MPNet + MS MARCO'),('Metadata only','Metadata only')]}
    models['Gold-trained + metadata']=assess(ranked,records,golds,labels)
    result=dict(protocol=protocol,models=models,retriever_only=assess(dense,records,golds,labels),note='Positive control: official issuer-held-out quantitative test queries; identical 5,230 corpus and top20 reranking for original, enriched and trained models. No test metric used for checkpoint/rate selection. Not a ceiling or proof that every weak-label source would succeed/fail.')
    atomic_json(directory/'results.json',result)
    report=table(['Case','R@1','R@5','R@10','R@20','MRR','nDCG@10'],[[name,*[f'{m["gold_passage_metrics"][k]:.2f}' for k in ['recall@1','recall@5','recall@10','recall@20','mrr','ndcg@10']]] for name,m in models.items()])
    (directory/'REPORT.md').write_text('# FinRank gold-data positive control\n\n'+report+'\n'+result['note']+'\nSee ../plan.json for sample size, unique queries, issuer boundaries, negative filtering, and validation coverage.\n')
    print(report);print('DONE: gold-data positive control saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','run'])
    parser.add_argument('--output-dir',type=Path,default=DEFAULT);parser.add_argument('--pairs',type=int,default=474);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(out,args.pairs,args.seed);print(json.dumps(plan['counts'],indent=2),flush=True)
        if args.action=='prepare':print('DONE: gold control prepared; no models loaded.');return
        if args.action in ['train','run']:
            engine.freeze_reranker_candidates(out,args.device)
            for kind in ['retriever','reranker']:
                for rate in plan['request']['rates']:
                    print(f'STAGE: gold {kind} lr={rate}',flush=True);engine.train_one(out,True,kind,rate,args.device)
        if args.action in ['evaluate','run']:evaluate(out,args.device)


if __name__=='__main__':main()
