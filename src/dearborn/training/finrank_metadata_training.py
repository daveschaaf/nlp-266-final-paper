"""Train both original models on frozen metadata-enriched silver pairs; compare on FinRank."""
import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from dearborn.retrieval.metadata import enrich_passage
from dearborn.experiments.finrank_generation_run import ROOT
from dearborn.experiments.finrank_quantitative_baseline import atomic_array
from dearborn.experiments.finrank_metadata_enrichment import prepare as prepare_benchmark
from dearborn.experiments.finrank_source_recovery import assess
from dearborn.experiments.finrank_eda import table
from .finrank_pilot import atomic_json

SOURCE=ROOT/'data/experiments/finrank_training/pilot500-v1'
DEFAULT=ROOT/'data/experiments/finrank_training/pilot500-metadata-v1'
ENRICHED=ROOT/'data/experiments/finrank_metadata_enrichment/v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich_records(records, connection):
    """Resolve every actual passage accession, including negatives, without query hints."""
    connection.row_factory=sqlite3.Row
    provenance={}
    def enriched(text,ticker,passage_id):
        accession=passage_id.split(':',1)[0]
        row=connection.execute('SELECT f.filing_year,f.form,c.ticker,c.name FROM filing f JOIN company c ON c.cik=f.cik WHERE f.accession_number=?',(accession,)).fetchone()
        if row is None:raise ValueError(f'No filing metadata for {passage_id}')
        if row['ticker'].upper()!=ticker.upper():raise ValueError(f'Issuer mismatch for {passage_id}: {ticker} != {row["ticker"]}')
        year=str(row['filing_year']);form=row['form']
        if not year.isdigit() or form not in {'10-K','10-Q'}:raise ValueError(f'Invalid source metadata for {passage_id}')
        provenance[accession]=dict(ticker=ticker,company_name=row['name'],document_year=year,form=form)
        return enrich_passage(text,{(ticker,year,form)},{ticker:row['name']})
    output=[]
    for record in records:
        item=dict(record)
        item['positive_text']=enriched(record['positive_text'],record['ticker'],record['passage_id'])
        item['negatives']=[dict(n,text=enriched(n['text'],n['ticker'],n['passage_id'])) for n in record['negatives']]
        output.append(item)
    return output,provenance


def load_dataset(out):
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    if sha(out/'training_records.jsonl')!=manifest['training_records_sha256']:raise ValueError('Frozen enriched dataset changed')
    return [json.loads(line) for line in (out/'training_records.jsonl').read_text().splitlines()]


def prepare(source,out,database,enriched_base):
    out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((source/'dataset_manifest.json').read_text())
    if sha(source/'training_records.jsonl')!=source_manifest['training_records_sha256']:raise ValueError('Original dataset hash mismatch')
    baseline=Path(json.loads((source/'pilot_plan.json').read_text())['baseline'])
    plan=dict(version='metadata-training-v1',source=str(source.resolve()),source_sha256=source_manifest['training_records_sha256'],database=str(database.resolve()),baseline=str(baseline.resolve()),enriched_reference=str(enriched_base.resolve()),query_mode='Unchanged raw questions',period='Filing year, not fiscal measurement period; no inferred quarter',training='Original weights; unchanged split, negatives, seed, losses and hyperparameters')
    path=out/'metadata_training_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Frozen training configuration differs; use a new directory')
        load_dataset(out)
        return plan
    records=[json.loads(line) for line in (source/'training_records.jsonl').read_text().splitlines()]
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        enriched,provenance=enrich_records(records,connection)
    target=out/'training_records.jsonl'
    temporary=target.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in enriched));temporary.replace(target)
    manifest=dict(source_manifest,training_records_sha256=sha(target),source_training_records_sha256=source_manifest['training_records_sha256'],encoding='Source prefix on positive and selected negative passages; raw questions unchanged')
    atomic_json(out/'dataset_manifest.json',manifest);atomic_json(out/'source_metadata.json',provenance);atomic_json(path,plan)
    return plan


def evaluate(out,device=None,reranker_checkpoint=None):
    from sentence_transformers import SentenceTransformer
    from dearborn.finrank import INDEX
    from .finrank_retriever import checkpoint_hash,rerank
    plan=json.loads((out/'metadata_training_plan.json').read_text());load_dataset(out)
    base=Path(plan['baseline']);reference_dir=Path(plan['enriched_reference'])
    if not (reference_dir/'results.json').exists():raise ValueError('Complete the original metadata-enrichment comparison first')
    # Read the already frozen benchmark prefix format and validate its source plan.
    records,texts,labels,benchmark=prepare_benchmark(base,reference_dir)
    checkpoints=[out/'retriever/best',Path(reranker_checkpoint) if reranker_checkpoint else out/'training/best']
    if not all(p.exists() for p in checkpoints):raise ValueError('Train both models first')
    directory=out/'evaluation';directory.mkdir(exist_ok=True)
    protocol=dict(training_plan=plan,training_records_sha256=sha(out/'training_records.jsonl'),benchmark=benchmark,checkpoint_hashes=[checkpoint_hash(p) for p in checkpoints],rerank_top_k=20)
    path=directory/'plan.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Evaluation configuration changed; use a fresh experiment directory')
    atomic_json(path,protocol)
    dense_path=directory/'dense_rankings.npy'
    if dense_path.exists():dense=np.load(dense_path)
    else:
        kwargs=dict(local_files_only=True)
        if device:kwargs['device']=device
        model=SentenceTransformer(str(checkpoints[0]),**kwargs)
        cache=directory/'corpus_embeddings.npy'
        corpus=np.load(cache) if cache.exists() else model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        if not cache.exists():atomic_array(cache,corpus)
        queries=model.encode([r['question'] for r in records],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        dense=np.argsort(-(queries@corpus.T).astype(np.float32),axis=1);atomic_array(dense_path,dense)
        del model,corpus,queries;gc.collect()
    combined=rerank(directory,'metadata_trained_both',checkpoints[1],dense,records,texts,device)
    atomic_array(directory/'reranked_rankings.npy',combined)
    golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    original=json.loads((reference_dir/'results.json').read_text())
    models={'original_raw':original['references']['original_mpnet_msmarco'],'original_enriched':original['models']['metadata_mpnet_msmarco'],'trained_enriched':assess(combined,records,golds,labels)}
    result=dict(protocol=protocol,models=models,trained_retriever=assess(dense,records,golds,labels),note='Same 376 questions and 5,230 passage IDs; no query metadata or filters. Development benchmark, not an untouched test set. Silver validation selects checkpoints.')
    result['training_delta_percentage_points']={k:v-models['original_enriched']['gold_passage_metrics'][k] for k,v in models['trained_enriched']['gold_passage_metrics'].items()}
    atomic_json(directory/'results.json',result)
    rows=[]
    for name,m in models.items():
        sm=m['source_metrics'];gm=m['gold_passage_metrics']
        rows.append([name,*[f'{v:.2f}' for v in [sm['issuer_hit@1'],sm['source_hit@1'],gm['recall@5'],gm['recall@10'],gm['recall@20'],gm['ndcg@10']]]])
    report=table(['Configuration','Issuer Hit@1','Source Hit@1','Gold R@5','Gold R@10','Gold R@20','nDCG@10'],rows)
    (directory/'REPORT.md').write_text('# Metadata-enriched training comparison\n\n'+report+'\n'+result['note']+'\nSource means issuer plus document year/form, not verified fiscal quarter. Prefixes consume token budget.\n')
    print(report);print('DONE: metadata-enriched training comparison saved.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','run'])
    parser.add_argument('--source-dir',type=Path,default=SOURCE);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    parser.add_argument('--database',type=Path,default=ROOT/'data/corpus.sqlite');parser.add_argument('--enriched-reference',type=Path,default=ENRICHED)
    parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'experiment.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args.source_dir.resolve(),out,args.database,args.enriched_reference)
        if args.action=='prepare':print(f'Frozen {len(load_dataset(out))} enriched pairs; no inference.');return
        if args.action in {'train','run'}:
            from .finrank_reranker import train as train_reranker
            from .finrank_retriever import train as train_retriever
            import torch
            print('STAGE: train metadata-enriched MS MARCO from original weights',flush=True)
            train_reranker(out,args.device);gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
            print('STAGE: train metadata-enriched MPNet from original weights',flush=True)
            train_retriever(out,args.device);gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        if args.action in {'evaluate','run'}:
            print('STAGE: frozen FinRank comparison',flush=True)
            evaluate(out,args.device)


if __name__=='__main__':main()
