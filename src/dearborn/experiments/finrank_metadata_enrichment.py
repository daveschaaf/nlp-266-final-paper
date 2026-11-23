"""Original MPNet/MS MARCO with passage metadata; fixed raw queries and candidate IDs."""
import argparse
import fcntl
import gc
import gzip
import json
from pathlib import Path
import time
import numpy as np
from dearborn.finrank import RECORDS,TEXT,INDEX
from dearborn.retrieval.metadata import enrich_passage
from .finrank_question_structure import ALIASES
from .finrank_generation_run import ROOT,append
from .finrank_quantitative_baseline import digest,atomic_array,DENSE,CROSS
from .finrank_source_recovery import source_labels,assess
from .finrank_eda import table

DEFAULT=ROOT/"data/experiments/finrank_metadata_enrichment/v1"


def prepare(base,out):
    baseline=json.loads((base/'run_manifest.json').read_text());records=[r for r in RECORDS if r['reasoning_type']=='Quantitative']
    if baseline['query_ids']!=[r['id'] for r in records] or baseline['corpus_sha256']!=digest(TEXT) or baseline['query_sha256']!=digest([r['question'] for r in records]):raise ValueError('Frozen baseline differs')
    labels=source_labels(RECORDS,INDEX);names={ticker:aliases[0] for ticker,aliases in ALIASES.items()}
    texts=[enrich_passage(text,labels[i],names) for i,text in enumerate(TEXT)]
    plan=dict(version='metadata-enrichment-v1',baseline=baseline,baseline_dir=str(base.resolve()),query_mode='Original questions unchanged; no oracle query metadata or filtering',corpus_count=len(TEXT),enriched_corpus_sha256=digest(texts),issuer_display_names=names,
        encoding='Source metadata prefix + original passage, all observed sources retained',ambiguous_source_texts=sum(len(v)>1 for v in labels.values()),dense_model=DENSE,reranker=CROSS,rerank_top_k=20,
        period_semantics='Released document year, NOT fiscal measurement period or quarter',truncation='Original model maximum lengths unchanged; prefixes consume some passage token budget',fine_tuning=False)
    out.mkdir(parents=True,exist_ok=True);path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen enrichment configuration differs; use a new directory')
    path.write_text(json.dumps(plan,indent=2))
    if not (out/'enriched_corpus.jsonl.gz').exists():
        with gzip.open(out/'enriched_corpus.jsonl.gz','wt') as handle:
            for i,text in enumerate(texts):handle.write(json.dumps(dict(corpus_id=i,text=text,sources=[list(t) for t in sorted(labels[i])]),ensure_ascii=False)+'\n')
    return records,texts,labels,plan


def run(base,out,device=None):
    records,texts,labels,plan=prepare(base,out)
    from sentence_transformers import SentenceTransformer,CrossEncoder
    kwargs=dict(local_files_only=True)
    if device:kwargs['device']=device
    dense_path=out/'dense_rankings.npy';started=time.perf_counter()
    if dense_path.exists():dense=np.load(dense_path)
    else:
        encoder=SentenceTransformer(DENSE,**kwargs)
        cache=out/'corpus_embeddings.npy'
        corpus=np.load(cache) if cache.exists() else encoder.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        if not cache.exists():atomic_array(cache,corpus)
        queries=encoder.encode([r['question'] for r in records],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
        dense=np.argsort(-(queries@corpus.T).astype(np.float32),axis=1);atomic_array(dense_path,dense)
        (out/'runtime.json').write_text(json.dumps(dict(max_seq_length=encoder.max_seq_length,dense_seconds=time.perf_counter()-started,device=str(encoder.device)),indent=2))
        del encoder,corpus,queries;gc.collect()
    completed={};path=out/'rerank.jsonl'
    if path.exists():completed={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
    pending=[i for i,r in enumerate(records) if r['id'] not in completed]
    if pending:
        model=CrossEncoder(CROSS,**kwargs)
        with path.open('a') as handle:
            for i in pending:
                head=dense[i][:20];begin=time.perf_counter()
                scores=model.predict([(records[i]['question'],texts[j]) for j in head],show_progress_bar=False)
                row=dict(id=records[i]['id'],head=head[np.argsort(-scores,kind='stable')].tolist(),seconds=time.perf_counter()-begin)
                append(handle,row);completed[row['id']]=row
                if (i+1)%25==0 or i==pending[-1]:print(f'Metadata-enriched rerank={len(completed)}/{len(records)}',flush=True)
    combined=np.stack([np.concatenate([completed[r['id']]['head'],dense[i][20:]]) for i,r in enumerate(records)])
    atomic_array(out/'reranked_rankings.npy',combined)
    golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    models={name:assess(ranking,records,golds,labels) for name,ranking in [('metadata_mpnet',dense),('metadata_mpnet_msmarco',combined)]}
    baseline=json.loads((base/'results.json').read_text())
    base_dense=np.load(base/'dense_rankings.npy');base_ce={r['id']:r for r in map(json.loads,(base/'cross_encoder.jsonl').read_text().splitlines())}
    base_combined=np.stack([np.concatenate([base_ce[r['id']]['head'],base_dense[i][20:]]) for i,r in enumerate(records)])
    references={'original_mpnet':assess(base_dense,records,golds,labels),'original_mpnet_msmarco':assess(base_combined,records,golds,labels)}
    result=dict(plan=plan,models=models,references=references,note='Metadata-assisted, off-the-shelf models. No fine-tuning and no query metadata added. Source labels inherit release metadata; no quarter-recovery claim.')
    for name,reference in [('metadata_mpnet','original_mpnet'),('metadata_mpnet_msmarco','original_mpnet_msmarco')]:
        models[name]['gold_delta_percentage_points']={k:v-references[reference]['gold_passage_metrics'][k] for k,v in models[name]['gold_passage_metrics'].items()}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    header=['Configuration','Issuer Hit@1','Source Hit@1','Gold Recall@10','Gold Recall@20','nDCG@10'];rows=[]
    for name,value in list(references.items())+list(models.items()):
        sm=value['source_metrics'];gm=value['gold_passage_metrics'];rows.append([name,*[f'{v:.2f}' for v in [sm['issuer_hit@1'],sm['source_hit@1'],gm['recall@10'],gm['recall@20'],gm['ndcg@10']]]])
    report='# Passage metadata enrichment pilot\n\n'+table(header,rows)+'\nSame 376 raw questions, same 5,230 passage IDs, original model weights, top20 reranking. Only passage serialization changes. Source metadata is added to both MPNet and MS MARCO inputs. All observed labels are listed for ambiguous deduplicated texts. Document year is not measurement year or quarter. Prefixes consume token budget and can change truncation; interpret results as the operational effect of metadata-prefixed inputs, not a perfectly isolated semantic effect. Source labels are supplied by the benchmark, not independently recovered. No query-time gold metadata filters.\n'
    (out/'REPORT.md').write_text(report);print(table(header,rows));print('DONE: metadata-enrichment comparison saved; no weights trained.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-dir',type=Path,default=ROOT/'data/experiments/finrank_quantitative_baseline');parser.add_argument('--output-dir',type=Path,default=DEFAULT);parser.add_argument('--plan-only',action='store_true');parser.add_argument('--device',choices=['cpu','mps','cuda'])
    args=parser.parse_args();base=args.baseline_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.plan_only:
            records,texts,_,plan=prepare(base,out);print(f'Frozen {len(records)} unchanged questions and {len(texts)} metadata-enriched passages. No inference.');return
        run(base,out,args.device)


if __name__=='__main__':main()
