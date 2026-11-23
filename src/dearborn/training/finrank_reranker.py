"""Train MS MARCO on the silver pilot; evaluate on frozen FinRank dense candidates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from dearborn.finrank_policy import assert_not_finrank_tickers
from dearborn.experiments.finrank_quantitative_baseline import CROSS,metrics_for,atomic_array
from dearborn.experiments.finrank_generation_run import ROOT
from .finrank_pilot import DEFAULT,atomic_json,collect


def issuer_split(ticker,seed=42):
    return "validation" if int(hashlib.sha256(f"{seed}:{ticker.upper()}".encode()).hexdigest(),16)%10==0 else "train"


def build_dataset(records,seed=42):
    assert_not_finrank_tickers([r["ticker"] for r in records]);output=[];rng=random.Random(seed)
    for record in records:
        split=issuer_split(record["ticker"],seed);pool=list(record["negative_candidates"])
        fallback=[dict(text=r["positive_text"],ticker=r["ticker"],passage_id=r["passage_id"]) for r in records if r["ticker"]!=record["ticker"] and issuer_split(r["ticker"],seed)==split]
        rng.shuffle(fallback);pool+=fallback;negatives=[];seen={record["positive_text"]}
        for candidate in pool:
            if candidate["ticker"]==record["ticker"] or issuer_split(candidate["ticker"],seed)!=split or candidate["text"] in seen:continue
            assert_not_finrank_tickers([candidate["ticker"]]);negatives.append(candidate);seen.add(candidate["text"])
            if len(negatives)==3:break
        if not negatives:raise ValueError(f"No same-split other-issuer negative for {record['id']}; need more issuer coverage")
        output.append(dict(record,split=split,negatives=negatives))
    if {r["split"] for r in output}!={"train","validation"}:raise ValueError("Need both train and issuer-disjoint validation examples")
    return output


def refresh_status(out):
    if (out/"metadata_training_plan.json").exists():
        return  # This frozen experiment owns its status; never recollect raw passages.
    return collect(out,json.loads((out/"pilot_plan.json").read_text()))


def prepare_dataset(out):
    if (out/"metadata_training_plan.json").exists():
        from .finrank_metadata_training import load_dataset
        return load_dataset(out)
    plan=json.loads((out/"pilot_plan.json").read_text());status=collect(out,plan)
    if status["accepted_unique"]<plan["target"]:raise ValueError(f"Need {plan['target']} accepted pairs; have {status['accepted_unique']}")
    source=out/"accepted.jsonl";digest=hashlib.sha256(source.read_bytes()).hexdigest();path=out/"training_records.jsonl"
    if path.exists():
        manifest=json.loads((out/"dataset_manifest.json").read_text())
        if manifest["accepted_sha256"]!=digest:raise ValueError("Accepted data changed after dataset freeze")
        return [json.loads(l) for l in path.read_text().splitlines()]
    records=build_dataset([json.loads(l) for l in source.read_text().splitlines()],plan["seed"])
    tmp=path.with_suffix('.tmp');tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records));tmp.replace(path)
    manifest=dict(accepted_sha256=digest,training_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=plan["seed"],accepted_count=len(records),split_counts={s:sum(r["split"]==s for r in records) for s in ["train","validation"]},split_issuers={s:sorted({r["ticker"] for r in records if r["split"]==s}) for s in ["train","validation"]},negatives="Up to 3 MPNet other-issuer candidates in same split; fallback other same-split positives; exact duplicate texts excluded",labels="Silver; not human gold",model=CROSS,epochs=3,learning_rate=2e-5,max_length=512)
    atomic_json(out/"dataset_manifest.json",manifest);collect(out,plan);return records


def pair_rows(records,split):
    return [(r["question"],text,label) for r in records if r["split"]==split for text,label in [(r["positive_text"],1.)]+[(n["text"],0.) for n in r["negatives"]]]


def train(out,device):
    import torch
    from sentence_transformers import CrossEncoder
    records=prepare_dataset(out);manifest=json.loads((out/"dataset_manifest.json").read_text());directory=out/"training";directory.mkdir(exist_ok=True)
    seed=manifest["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    encoder=CrossEncoder(CROSS,local_files_only=True,device=device);model=encoder.model;tokenizer=encoder.tokenizer
    optimizer=torch.optim.AdamW(model.parameters(),lr=manifest["learning_rate"]);loss_fn=torch.nn.BCEWithLogitsLoss()
    history=[];best=float('inf');start=1;last=directory/"last.pt";batch_size=8
    if last.exists():
        payload=torch.load(last,map_location="cpu",weights_only=False)
        if payload["dataset_sha256"]!=manifest["training_records_sha256"]:raise ValueError("Checkpoint dataset differs")
        model.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(device)
        start=payload["epoch"]+1;history=payload["history"];best=payload["best_validation_loss"]
        print(f"Resumed epoch={start}; best validation loss={best:.6f}",flush=True)
    started=time.perf_counter()
    for epoch in range(start,manifest["epochs"]+1):
        stats={}
        for split in ["train","validation"]:
            model.train(split=="train");pairs=pair_rows(records,split)
            if split=="train":random.Random(seed+epoch).shuffle(pairs);torch.manual_seed(seed+epoch)
            total=0.;count=0
            for offset in range(0,len(pairs),batch_size):
                batch=pairs[offset:offset+batch_size];tokens=tokenizer([r[0] for r in batch],[r[1] for r in batch],padding=True,truncation=True,max_length=512,return_tensors="pt").to(device)
                labels=torch.tensor([r[2] for r in batch],dtype=torch.float32,device=device)
                with torch.set_grad_enabled(split=="train"):
                    logits=model(**tokens).logits.squeeze(-1);loss=loss_fn(logits,labels)
                    if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite {split} loss")
                    if split=="train":
                        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
                total+=loss.item()*len(batch);count+=len(batch)
                if offset//batch_size%25==0:print(f"epoch={epoch} {split} pairs={min(offset+len(batch),len(pairs))}/{len(pairs)} loss={loss.item():.4f}",flush=True)
            stats[split+"_loss"]=total/count
        history.append(dict(epoch=epoch,**stats));improved=stats["validation_loss"]<best
        if improved:
            best=stats["validation_loss"];encoder.save(str(directory/"best"))
        payload=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best_validation_loss=best,dataset_sha256=manifest["training_records_sha256"])
        tmp=directory/"last.tmp";torch.save(payload,tmp);tmp.replace(last)
        atomic_json(directory/"status.json",dict(epoch=epoch,epochs=manifest["epochs"],best_validation_loss=best,history=history,device=device,current_invocation_seconds=time.perf_counter()-started,phase="trained" if epoch==manifest["epochs"] else "training"))
        refresh_status(out);print(json.dumps(history[-1]),flush=True)
    print("DONE: best reranker selected on synthetic validation only; no FinRank evaluation yet.",flush=True)


def evaluate(out,device):
    from sentence_transformers import CrossEncoder
    from dearborn.experiments.finrank_quantitative_baseline import prepare as prepare_baseline
    pilot=json.loads((out/"pilot_plan.json").read_text());base=Path(pilot["baseline"]);directory=out/"training"
    if not (directory/"best").exists():raise ValueError("Train the reranker first")
    records,golds,baseline_manifest=prepare_baseline(base);dense=np.load(base/"dense_rankings.npy");baseline=json.loads((base/"results.json").read_text())
    from dearborn.finrank import TEXT
    if dense.shape!=(len(records),len(TEXT)):raise ValueError("Frozen dense ranking shape mismatch")
    kwargs=dict(local_files_only=True)
    if device:kwargs["device"]=device
    model=CrossEncoder(str(directory/"best"),**kwargs);rankings=[];started=time.perf_counter()
    for i,record in enumerate(records):
        head=dense[i][:20];scores=model.predict([(record["question"],TEXT[j]) for j in head],show_progress_bar=False)
        rankings.append(np.concatenate([head[np.argsort(-scores,kind="stable")],dense[i][20:]]))
        if (i+1)%25==0 or i+1==len(records):print(f"FinRank evaluation={i+1}/{len(records)}",flush=True)
    result=metrics_for(np.stack(rankings),records,golds);reference=baseline["models"]["ce_dense"]["metrics"]
    result.update(baseline=reference,delta_percentage_points={k:result["metrics"][k]-reference[k] for k in reference},seconds=time.perf_counter()-started,query_count=len(records),corpus_count=len(TEXT),protocol="Same frozen MPNet top20 and full FinRank corpus; only reranker weights changed",checkpoint="best",dataset_sha256=json.loads((out/"dataset_manifest.json").read_text())["training_records_sha256"])
    atomic_json(directory/"evaluation.json",result);atomic_array(directory/"finrank_rankings.npy",np.stack(rankings));collect(out,pilot)
    print(json.dumps({k:result[k] for k in ["metrics","baseline","delta_percentage_points"]},indent=2));print("DONE: trained checkpoint versus frozen baseline saved.")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["prepare","train","evaluate"]);parser.add_argument("--output-dir",type=Path,default=DEFAULT);parser.add_argument("--device",choices=["cpu","mps","cuda"])
    args=parser.parse_args();out=args.output_dir.resolve()
    import fcntl
    with (out/"training.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=="prepare":prepare_dataset(out);print((out/"dataset_manifest.json").read_text())
        elif args.action=="train":train(out,args.device)
        else:evaluate(out,args.device)


if __name__=="__main__":main()
