"""Fine-tune MPNet on frozen silver pairs, then evaluate retriever/reranker combinations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
import numpy as np
from .finrank_pilot import DEFAULT,atomic_json,collect
from .finrank_reranker import prepare_dataset
from dearborn.experiments.finrank_generation_run import ROOT,append
from dearborn.experiments.finrank_quantitative_baseline import DENSE,CROSS,metrics_for,atomic_array


def refresh_status(out):
    if not (out/'metadata_training_plan.json').exists():
        collect(out,json.loads((out/'pilot_plan.json').read_text()))


def paired_loss(query,positive,negative,negative_mask,temperature=.05):
    import torch
    import torch.nn.functional as F
    if temperature<=0:raise ValueError("temperature must be positive")
    if not negative_mask.any(dim=1).all():raise ValueError("Each query requires a negative")
    query=F.normalize(query,dim=-1);positive=F.normalize(positive,dim=-1);negative=F.normalize(negative,dim=-1)
    positive_scores=(query*positive).sum(-1,keepdim=True)
    negative_scores=(query.unsqueeze(1)*negative).sum(-1).masked_fill(~negative_mask,float('-inf'))
    return F.cross_entropy(torch.cat([positive_scores,negative_scores],dim=1)/temperature,torch.zeros(len(query),dtype=torch.long,device=query.device))


def batch_loss(model,records,device,temperature):
    import torch
    size=max(len(r['negatives']) for r in records)
    queries=[r['question'] for r in records];positives=[r['positive_text'] for r in records]
    # Pad the list of texts; the padding embeddings never enter the denominator.
    negatives=[r['negatives'][j]['text'] if j<len(r['negatives']) else '' for r in records for j in range(size)]
    mask=torch.tensor([[j<len(r['negatives']) for j in range(size)] for r in records],dtype=torch.bool,device=device)
    def embed(texts):
        features={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in model.preprocess(texts).items()}
        return model(features)['sentence_embedding']
    return paired_loss(embed(queries),embed(positives),embed(negatives).reshape(len(records),size,-1),mask,temperature)


def train(out,device=None):
    import torch
    from sentence_transformers import SentenceTransformer
    records=prepare_dataset(out);dataset=json.loads((out/'dataset_manifest.json').read_text());directory=out/'retriever';directory.mkdir(exist_ok=True)
    config=dict(model=DENSE,seed=42,epochs=3,batch_size=2,learning_rate=2e-5,temperature=.05,dataset_sha256=dataset['training_records_sha256'],loss='Paired InfoNCE, explicit same-split negatives only; no in-batch negatives')
    path=directory/'train_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Frozen retriever configuration differs')
    atomic_json(path,config)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model=SentenceTransformer(DENSE,local_files_only=True,device=device)
    atomic_json(directory/'runtime.json',dict(device=device,max_seq_length=model.max_seq_length,normalization='L2 cosine; original MPNet prefix truncation, no windows'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']);history=[];best=float('inf');start=1
    if (directory/'last.pt').exists():
        payload=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
        if payload['config']!=config:raise ValueError('Checkpoint configuration differs')
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
        history=payload['history'];best=payload['best'];start=payload['epoch']+1
        print(f'Resumed MPNet at epoch {start}',flush=True)
    started=time.perf_counter();cpu_started=time.process_time()
    for epoch in range(start,config['epochs']+1):
        stats={};epoch_started=time.perf_counter()
        for split in ['train','validation']:
            group=[r for r in records if r['split']==split];model.train(split=='train')
            if split=='train':random.Random(config['seed']+epoch).shuffle(group);torch.manual_seed(config['seed']+epoch)
            total=0.;count=0
            for offset in range(0,len(group),config['batch_size']):
                batch=group[offset:offset+config['batch_size']]
                with torch.set_grad_enabled(split=='train'):
                    loss=batch_loss(model,batch,device,config['temperature'])
                    if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite {split} loss')
                    if split=='train':
                        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=loss.item()*len(batch);count+=len(batch)
                if offset//config['batch_size']%25==0:print(f'MPNet epoch={epoch} {split} queries={min(offset+len(batch),len(group))}/{len(group)} loss={loss.item():.4f}',flush=True)
            stats[split+'_loss']=total/count
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-epoch_started,**stats))
        if stats['validation_loss']<best:best=stats['validation_loss'];model.save(str(directory/'best'))
        payload=dict(config=config,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=best)
        temporary=directory/'last.tmp';torch.save(payload,temporary);temporary.replace(directory/'last.pt')
        status=dict(phase='trained' if epoch==config['epochs'] else 'training',epoch=epoch,epochs=config['epochs'],best_validation_loss=best,history=history,device=device,current_invocation_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e9/1024))
        if device=='mps':status.update(mps_allocated_gb=torch.mps.current_allocated_memory()/1e9,mps_driver_gb=torch.mps.driver_allocated_memory()/1e9)
        atomic_json(directory/'status.json',status);refresh_status(out);print(json.dumps(history[-1]),flush=True)
    print('DONE: best MPNet selected on synthetic validation; no gold evaluation started.',flush=True)


def checkpoint_hash(path):
    digest=hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        with file.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def rerank(directory,name,model_path,dense,records,texts,device):
    from sentence_transformers import CrossEncoder
    path=directory/(name+'_scores.jsonl');completed={}
    if path.exists():completed={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
    pending=[i for i,r in enumerate(records) if r['id'] not in completed]
    if pending:
        kwargs=dict(local_files_only=True)
        if device:kwargs['device']=device
        model=CrossEncoder(str(model_path),**kwargs)
        with path.open('a') as handle:
            for i in pending:
                head=dense[i][:20];scores=model.predict([(records[i]['question'],texts[j]) for j in head],show_progress_bar=False)
                row=dict(id=records[i]['id'],head=head[np.argsort(-scores,kind='stable')].tolist());append(handle,row);completed[row['id']]=row
                if (i+1)%25==0 or i==pending[-1]:print(f'{name} {len(completed)}/{len(records)}',flush=True)
        del model;gc.collect()
    return np.stack([np.concatenate([completed[r['id']]['head'],dense[i][20:]]) for i,r in enumerate(records)])


def evaluate(out,device=None):
    from sentence_transformers import SentenceTransformer
    from dearborn.experiments.finrank_quantitative_baseline import prepare as baseline_prepare
    from dearborn.finrank import TEXT
    pilot=json.loads((out/'pilot_plan.json').read_text());base=Path(pilot['baseline']);directory=out/'retriever';checkpoint=directory/'best'
    if not checkpoint.exists() or not (out/'training/best').exists():raise ValueError('Both best checkpoints must exist')
    records,golds,manifest=baseline_prepare(base)
    protocol=dict(retriever_checkpoint_sha256=checkpoint_hash(checkpoint),reranker_checkpoint_sha256=checkpoint_hash(out/'training/best'),baseline_manifest=manifest,top_k=20)
    path=directory/'evaluation_plan.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Evaluation inputs changed; archive existing evaluation before rerunning')
    atomic_json(path,protocol)
    cache=directory/'finrank_dense_rankings.npy'
    if cache.exists():dense=np.load(cache)
    else:
        kwargs=dict(local_files_only=True)
        if device:kwargs['device']=device
        encoder=SentenceTransformer(str(checkpoint),**kwargs)
        corpus_cache=directory/'finrank_corpus_embeddings.npy';corpus=encoder.encode(TEXT,batch_size=32,normalize_embeddings=True,show_progress_bar=True) if not corpus_cache.exists() else np.load(corpus_cache)
        if not corpus_cache.exists():atomic_array(corpus_cache,corpus)
        queries=encoder.encode([r['question'] for r in records],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        dense=np.argsort(-(queries@corpus.T).astype(np.float32),axis=1);atomic_array(cache,dense)
        del encoder,corpus,queries;gc.collect()
    models={};files={}
    configs=[('trained_mpnet',None),('trained_mpnet_original_reranker',CROSS),('trained_mpnet_trained_reranker',str(out/'training/best'))]
    reference=json.loads((base/'results.json').read_text());original_dense=reference['models']['dense']['metrics'];original_combined=reference['models']['ce_dense']['metrics']
    for name,path in configs:
        ranking=dense if path is None else rerank(directory,name,path,dense,records,TEXT,device)
        target=directory/(name+'_rankings.npy');atomic_array(target,ranking)
        result=metrics_for(ranking,records,golds);comparison=original_dense if path is None else original_combined
        result['reference_metrics']=comparison;result['delta_percentage_points']={k:result['metrics'][k]-comparison[k] for k in comparison}
        models[name]=result;files[name]=str(target.resolve());print(name,json.dumps(result['metrics']),flush=True)
    output=dict(query_count=len(records),corpus_count=len(TEXT),models=models,ranking_files=files,protocol=protocol,note='Same evaluation queries and full corpus; new top20 from trained retriever. Best checkpoints selected on synthetic validation only.')
    atomic_json(directory/'evaluation.json',output);collect(out,pilot)
    print('DONE: trained retriever alone, original reranker combination, and both trained saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['train','evaluate']);parser.add_argument('--output-dir',type=Path,default=DEFAULT);parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args();out=args.output_dir.resolve()
    import fcntl
    with (out/'retriever.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return train(out,args.device) if args.action=='train' else evaluate(out,args.device)


if __name__=='__main__':main()
