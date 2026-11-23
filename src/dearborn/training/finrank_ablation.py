"""Matched metadata x synthetic-training ablation for MPNet and MS MARCO.

Run prepare without models; run trains/evaluates sequentially and resumes saved checks.
"""
import argparse
from collections import Counter
import fcntl
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
from .finrank_hard_negatives import read_pool,sample_records,validation_pool,training_records,render
from .finrank_metadata_training import sha
from .finrank_retriever import batch_loss,checkpoint_hash,rerank
from .finrank_reranker import pair_rows
from .finrank_pilot import atomic_json
from dearborn.experiments.finrank_generation_run import ROOT
from dearborn.experiments.finrank_quantitative_baseline import DENSE,CROSS,atomic_array,digest

SOURCE=ROOT/'data/experiments/finrank_training/pilot500-v1'
DEFAULT=SOURCE.parent/'hard-negatives-ablation-v1'
RATES=(2e-6,2e-7)


def write_rows(path,rows):
    tmp=path.with_suffix('.tmp')
    with gzip.open(tmp,'wt') as handle:
        for r in rows:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    tmp.replace(path)


def read_rows(path):
    with gzip.open(path,'rt') as handle:return [json.loads(line) for line in handle]


def prepare(source,out,database,seed=42):
    out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((source/'dataset_manifest.json').read_text())
    source_sha=sha(source/'training_records.jsonl')
    if source_sha!=source_manifest['training_records_sha256']:raise ValueError('Source dataset changed')
    request=dict(version='hard-negative-ablation-v1',source=str(source.resolve()),source_sha256=source_sha,database=str(database.resolve()),seed=seed,rates=list(RATES),epochs=3,evaluate_every_steps=25,temperature=.05,retriever_batch_size=2,reranker_batch_size=8,reranker_max_length=512,selection='Per-checkpoint validation MRR including step 0; earliest tie. Learning rate selected per component by mean best validation MRR across raw/enriched, ties prefer lower rate.',validation='All parsed passages of existing held-out issuers, exact-text deduplicated; reranker fixed raw original MPNet top20, no injected gold.',negative_rules='One per category; seeded random, same split; normalized-text dedup; same section across filings additionally requires same form; unknown industries skipped; prefer same fiscal year/period for different issuer.',labels='Silver positives and assumed negatives; comparative and cross-section evidence may create false negatives.',baseline=str(Path(json.loads((source/'pilot_plan.json').read_text())['baseline']).resolve()),enriched_reference=str((ROOT/'data/experiments/finrank_metadata_enrichment/v1').resolve()))
    plan_path=out/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if plan['request']!=request:raise ValueError('Frozen experiment differs; use a new output directory')
        for filename,value in plan['files'].items():
            if sha(out/filename)!=value:raise ValueError(f'Frozen artifact changed: {filename}')
        return plan
    original=[json.loads(line) for line in (source/'training_records.jsonl').read_text().splitlines()]
    pool=read_pool(database,{r['ticker'] for r in original})
    frozen,stats=sample_records(original,pool,seed)
    held,validation,golds=validation_pool(frozen,pool)
    write_rows(out/'records.jsonl.gz',frozen);write_rows(out/'validation_pool.jsonl.gz',validation)
    atomic_json(out/'validation.json',dict(query_ids=[r['id'] for r in held],golds=golds,query_count=len(held),candidate_count=len(validation),candidate_ids=[p['passage_id'] for p in validation]))
    counts=dict(pairs=len(frozen),split_counts=dict(Counter(r['split'] for r in frozen)),validation_passages=len(validation),total_negatives=sum(len(r['negatives']) for r in frozen),negative_counts=stats)
    files={name:sha(out/name) for name in ['records.jsonl.gz','validation_pool.jsonl.gz','validation.json']}
    plan=dict(request=request,counts=counts,files=files)
    atomic_json(plan_path,plan)
    return plan


def device_name(device):
    import torch
    return device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def cleanup():
    gc.collect()
    import torch
    if torch.backends.mps.is_available():torch.mps.empty_cache()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def freeze_reranker_candidates(out,device=None):
    path=out/'reranker_candidates.npy'
    manifest=out/'reranker_candidates.json'
    if path.exists() and manifest.exists():
        if sha(path)!=json.loads(manifest.read_text())['sha256']:raise ValueError('Reranker candidates changed')
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    plan=json.loads((out/'plan.json').read_text());records=read_rows(out/'records.jsonl.gz');held=[r for r in records if r['split']=='validation'];pool=read_rows(out/'validation_pool.jsonl.gz')
    encoder=SentenceTransformer(DENSE,local_files_only=True,device=device_name(device))
    vectors=encoder.encode([p['text'] for p in pool],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    queries=encoder.encode([r['question'] for r in held],batch_size=32,normalize_embeddings=True,show_progress_bar=False)
    heads=np.argsort(-(queries@vectors.T).astype(np.float32),axis=1)[:,:20]
    atomic_array(path,heads);atomic_json(manifest,dict(sha256=sha(path),model=DENSE,plan_sha256=digest(plan),raw_inputs=True,top_k=20,injected_gold=False))
    del encoder,vectors,queries;cleanup()
    return heads


def rank_metrics(scores,golds,candidates=None):
    scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise FloatingPointError('Non-finite validation scores')
    order=np.argsort(-scores,axis=1,kind='stable')
    positions=[]
    for i,(row,gold) in enumerate(zip(order,golds)):
        ids=row if candidates is None else candidates[i][row]
        match=np.flatnonzero(np.isin(ids,np.atleast_1d(gold)));positions.append(float(match[0]+1) if len(match) else float('inf'))
    p=np.asarray(positions)
    return {**{f'recall@{k}':float(np.mean(p<=k)) for k in (1,5,10,20)},'mrr':float(np.mean(1/p))}


def cross_scores(encoder,pairs,device,batch_size=32):
    import torch
    values=[]
    with torch.no_grad():
        for offset in range(0,len(pairs),batch_size):
            batch=pairs[offset:offset+batch_size]
            tokens=encoder.tokenizer([r[0] for r in batch],[r[1] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
            values.extend(encoder.model(**tokens).logits.reshape(-1).detach().cpu().tolist())
    return np.asarray(values,dtype=np.float32)


def margins(scores,negative_indices,golds):
    values=[float(np.max(scores[i,np.atleast_1d(gold)])-max(scores[i,j] for j in negatives)) for i,(gold,negatives) in enumerate(zip(golds,negative_indices)) if negatives]
    return dict(mean_positive_minus_hardest_negative=float(np.mean(values)),fraction_positive_above_all_negatives=float(np.mean(np.asarray(values)>0)))


def validate(encoder,kind,records,pool,golds,heads,device,temperature):
    import torch
    held=[r for r in records if r['split']=='validation'];loss_held=[r for r in held if r['negatives']];model=encoder if kind=='retriever' else encoder.model
    was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            if kind=='retriever':
                vectors=encoder.encode([p['text'] for p in pool],batch_size=32,normalize_embeddings=True,show_progress_bar=False)
                queries=encoder.encode([r['question'] for r in held],batch_size=32,normalize_embeddings=True,show_progress_bar=False)
                scores=queries@vectors.T
                total=sum(batch_loss(encoder,loss_held[i:i+2],device,temperature).item()*len(loss_held[i:i+2]) for i in range(0,len(loss_held),2))
                metric=rank_metrics(scores,golds);loss=total/len(loss_held) if loss_held else None
                lookup={p['text']:j for j,p in enumerate(pool)}
                diagnostic=margins(scores,[[lookup[n['text']] for n in r['negatives']] for r in held],golds)
            else:
                fixed=[(r['question'],pool[j]['text']) for r,head in zip(held,heads) for j in head]
                scores=cross_scores(encoder,fixed,device).reshape(len(held),heads.shape[1])
                metric=rank_metrics(scores,golds,heads)
                pairs=pair_rows(loss_held,'validation');logits=torch.tensor(cross_scores(encoder,pairs,device));labels=torch.tensor([r[2] for r in pairs])
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels).item() if pairs else None
                offset=0;diff=[]
                for r in loss_held:
                    size=1+len(r['negatives']);diff.append(float(logits[offset]-logits[offset+1:offset+size].max()));offset+=size
                diagnostic=dict(mean_positive_minus_hardest_negative=float(np.mean(diff)),fraction_positive_above_all_negatives=float(np.mean(np.asarray(diff)>0)))
        if loss is not None and not np.isfinite(loss):raise FloatingPointError('Non-finite validation loss')
        return dict(loss=loss,query_count=len(held),loss_query_count=len(loss_held),candidate_count=len(pool),**metric,**diagnostic)
    finally:model.train(was_training)


def run_dir(out,metadata,kind,rate):
    return out/('enriched' if metadata else 'raw')/kind/f'lr-{rate:.0e}'


def save_best(encoder,path):
    temporary=path.with_name('best.tmp');backup=path.with_name('best.old')
    if temporary.exists():shutil.rmtree(temporary)
    encoder.save(str(temporary))
    if backup.exists():shutil.rmtree(backup)
    if path.exists():path.rename(backup)
    temporary.rename(path)
    if backup.exists():shutil.rmtree(backup)


def train_one(out,metadata,kind,rate,device=None):
    import torch
    from sentence_transformers import SentenceTransformer,CrossEncoder
    plan=json.loads((out/'plan.json').read_text());settings=plan['request']
    directory=run_dir(out,metadata,kind,rate);directory.mkdir(parents=True,exist_ok=True)
    config=dict(plan_sha256=digest(plan),metadata=metadata,component=kind,learning_rate=rate)
    config_path=directory/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Training configuration changed')
    atomic_json(config_path,config)
    last=directory/'last.pt';status_path=directory/'status.json'
    if status_path.exists() and json.loads(status_path.read_text())['phase']=='trained':
        print(f'SKIP completed {directory.relative_to(out)}',flush=True);return
    frozen=read_rows(out/'records.jsonl.gz');records=training_records(frozen,metadata)
    raw_pool=read_rows(out/'validation_pool.jsonl.gz');pool=[dict(p,text=render(p,metadata)) for p in raw_pool]
    # Enriched copies of a shared raw text can differ by source; all gold and negative
    # lookups must use the representative source chosen when freezing the pool.
    representative={p['text']:q['text'] for p,q in zip(raw_pool,pool)}
    for r,raw in zip(records,frozen):
        if r['split']=='validation':
            r['positive_text']=representative[raw['positive']['text']]
            for n,original in zip(r['negatives'],raw['negatives']):n['text']=representative[original['text']]
    validation=json.loads((out/'validation.json').read_text());golds=validation['golds'];heads=np.load(out/'reranker_candidates.npy')
    device=device_name(device);seed=settings['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    encoder=SentenceTransformer(DENSE,local_files_only=True,device=device) if kind=='retriever' else CrossEncoder(CROSS,local_files_only=True,device=device)
    model=encoder if kind=='retriever' else encoder.model
    optimizer=torch.optim.AdamW(model.parameters(),lr=rate)
    items=[r for r in records if r['split']=='train'] if kind=='retriever' else pair_rows(records,'train')
    batch_size=settings[kind+'_batch_size'];steps_per_epoch=(len(items)+batch_size-1)//batch_size;total_steps=steps_per_epoch*settings['epochs']
    start=0;history=[];best=-1.;best_step=0;total_loss=0.;count=0;started=time.perf_counter();cpu=time.process_time()
    def checkpoint(step):
        nonlocal best,best_step
        metrics=validate(encoder,kind,records,pool,golds,heads,device,settings['temperature'])
        history.append(dict(step=step,epoch=step/steps_per_epoch,train_loss=total_loss/count if count else None,validation=metrics))
        if metrics['mrr']>best:
            best=metrics['mrr'];best_step=step;save_best(encoder,directory/'best')
        payload=dict(config=config,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best,best_step=best_step,total_loss=total_loss,count=count,torch_rng=torch.get_rng_state())
        if device=='mps':payload['mps_rng']=torch.mps.get_rng_state()
        if device.startswith('cuda'):payload['cuda_rng']=torch.cuda.get_rng_state_all()
        temporary=directory/'last.tmp';torch.save(payload,temporary);temporary.replace(last)
        status=dict(phase='trained' if step==total_steps else 'training',config=config,step=step,total_steps=total_steps,best_step=best_step,best_validation_mrr=best,history=history,device=device,seconds_this_invocation=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6))
        if device=='mps':status.update(mps_allocated_gb=torch.mps.current_allocated_memory()/1e9,mps_driver_gb=torch.mps.driver_allocated_memory()/1e9)
        atomic_json(status_path,status);print(f'{directory.relative_to(out)} '+json.dumps(history[-1]),flush=True)
    if last.exists():
        p=torch.load(last,map_location='cpu',weights_only=False)
        if p['config']!=config:raise ValueError('Checkpoint differs')
        model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
        start=p['step'];history=p['history'];best=p['best'];best_step=p['best_step'];total_loss=p['total_loss'];count=p['count'];torch.set_rng_state(p['torch_rng'].cpu())
        if device=='mps' and 'mps_rng' in p:torch.mps.set_rng_state(p['mps_rng'].cpu())
        if device.startswith('cuda') and 'cuda_rng' in p:torch.cuda.set_rng_state_all(p['cuda_rng'])
        del p;print(f'RESUME {directory.relative_to(out)} step={start}',flush=True)
    else:checkpoint(0)
    model.train()
    for epoch in range(settings['epochs']):
        group=list(items);random.Random(seed+epoch+1).shuffle(group)
        for offset in range(0,len(group),batch_size):
            step=epoch*steps_per_epoch+offset//batch_size+1
            if step<=start:continue
            batch=group[offset:offset+batch_size];optimizer.zero_grad(set_to_none=True)
            if kind=='retriever':loss=batch_loss(encoder,batch,device,settings['temperature'])
            else:
                tokens=encoder.tokenizer([r[0] for r in batch],[r[1] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(**tokens).logits.reshape(-1),torch.tensor([r[2] for r in batch],dtype=torch.float32,device=device))
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss step={step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total_loss+=loss.item()*len(batch);count+=len(batch)
            if step%settings['evaluate_every_steps']==0 or step==total_steps:checkpoint(step)
    del model,encoder,optimizer;cleanup()


def select(out):
    plan=json.loads((out/'plan.json').read_text());selected={}
    for kind in ['retriever','reranker']:
        candidates=[]
        for rate in plan['request']['rates']:
            statuses=[json.loads((run_dir(out,m,kind,rate)/'status.json').read_text()) for m in (False,True)]
            if any(s['phase']!='trained' for s in statuses):raise ValueError('Complete every training run before selecting')
            candidates.append(dict(rate=rate,mean_validation_mrr=float(np.mean([s['best_validation_mrr'] for s in statuses]))))
        chosen=max(candidates,key=lambda r:(r['mean_validation_mrr'],-r['rate']))
        selected[kind]=dict(chosen,candidates=candidates,checkpoints={name:str((run_dir(out,m,kind,chosen['rate'])/'best').resolve()) for name,m in [('raw',False),('enriched',True)]})
    atomic_json(out/'selection.json',selected)
    return selected


def evaluate(out,device=None,chosen=None,training_cases=None):
    from sentence_transformers import SentenceTransformer
    from dearborn.finrank import INDEX
    from dearborn.experiments.finrank_metadata_enrichment import prepare as prepare_enriched
    from dearborn.experiments.finrank_source_recovery import assess
    from dearborn.experiments.finrank_eda import table
    plan=json.loads((out/'plan.json').read_text());chosen=select(out) if chosen is None else chosen
    base=Path(plan['request']['baseline']);reference=Path(plan['request']['enriched_reference'])
    records,enriched,labels,benchmark=prepare_enriched(base,reference)
    from dearborn.finrank import TEXT
    references=json.loads((reference/'results.json').read_text())
    models={'baseline':references['references']['original_mpnet_msmarco'],'metadata_only':references['models']['metadata_mpnet_msmarco']}
    golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    directory=out/'evaluation';directory.mkdir(exist_ok=True)
    hashes={kind:{mode:checkpoint_hash(Path(path)) for mode,path in selection['checkpoints'].items()} for kind,selection in chosen.items()}
    protocol=dict(plan_sha256=digest(plan),selection=chosen,checkpoint_hashes=hashes,benchmark=benchmark,reference_sha256=sha(reference/'results.json'),query_count=len(records),corpus_count=len(TEXT),rerank_top_k=20)
    path=directory/'plan.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Evaluation inputs changed; archive the evaluation directory first')
    atomic_json(path,protocol)
    for name,mode,texts in (training_cases if training_cases is not None else [('training_only','raw',TEXT),('metadata_training','enriched',enriched)]):
        case=directory/name;case.mkdir(exist_ok=True);cache=case/'dense_rankings.npy'
        if cache.exists():dense=np.load(cache)
        else:
            encoder=SentenceTransformer(chosen['retriever']['checkpoints'][mode],local_files_only=True,device=device_name(device))
            vectors_path=case/'corpus_embeddings.npy'
            vectors=np.load(vectors_path) if vectors_path.exists() else encoder.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
            if not vectors_path.exists():atomic_array(vectors_path,vectors)
            queries=encoder.encode([r['question'] for r in records],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
            dense=np.argsort(-(queries@vectors.T).astype(np.float32),axis=1);atomic_array(cache,dense)
            del encoder,vectors,queries;cleanup()
        ranked=rerank(case,'trained_both',chosen['reranker']['checkpoints'][mode],dense,records,texts,device)
        atomic_array(case/'rankings.npy',ranked)
        models[name]=assess(ranked,records,golds,labels)
        atomic_json(case/'retriever_metrics.json',assess(dense,records,golds,labels));cleanup()
    result=dict(protocol=protocol,models=models,note='Same FinRank raw queries, passage IDs and top20 reranking. Checkpoints selected using silver validation only. The experiment plan records training inputs and rate selection. Assumed negatives are not exhaustive relevance judgments. Prefixes consume token budget; source recovery is document year/form, not fiscal quarter.')
    atomic_json(directory/'results.json',result)
    rows=[]
    for name,m in models.items():
        sm=m['source_metrics'];gm=m['gold_passage_metrics']
        rows.append([name,*[f'{v:.2f}' for v in [sm['issuer_hit@1'],sm['source_hit@1'],gm['recall@1'],gm['recall@5'],gm['recall@10'],gm['recall@20'],gm['mrr'],gm['ndcg@10']]]])
    report=table(['Case','Issuer Hit@1','Source Hit@1','R@1','R@5','R@10','R@20','MRR','nDCG@10'],rows)
    (directory/'REPORT.md').write_text('# Metadata and synthetic training: matched ablation\n\n'+report+'\n'+result['note']+'\n\nTraining and sampling provenance: ../plan.json. Checkpoint selection: ../selection.json. This is an experiment record, not final-paper prose.\n')
    print(report);print('DONE: all four ablation cases saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','run'])
    parser.add_argument('--source-dir',type=Path,default=SOURCE);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    parser.add_argument('--database',type=Path,default=ROOT/'data/corpus.sqlite');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args.source_dir.resolve(),out,args.database,args.seed)
        print(json.dumps(plan['counts'],indent=2),flush=True)
        if args.action=='prepare':print('DONE: frozen data and validation pool; no model loaded.');return
        if args.action in {'train','run'}:
            print('STAGE: freeze raw original MPNet validation top20 for reranker',flush=True)
            freeze_reranker_candidates(out,args.device)
            for metadata in [False,True]:
                for kind in ['retriever','reranker']:
                    for rate in plan['request']['rates']:
                        print(f'STAGE: {"enriched" if metadata else "raw"} {kind} lr={rate}',flush=True)
                        train_one(out,metadata,kind,rate,args.device)
            selected=select(out);print('Validation-selected rates: '+json.dumps({k:v['rate'] for k,v in selected.items()}),flush=True)
        if args.action in {'evaluate','run'}:
            print('STAGE: four-case FinRank comparison',flush=True);evaluate(out,args.device)


if __name__=='__main__':main()
