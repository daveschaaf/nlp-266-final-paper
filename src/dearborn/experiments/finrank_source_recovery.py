"""Issuer/source-label diagnostics on frozen retrieval rankings; no model inference."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from dearborn.finrank import RECORDS,TEXT,INDEX,per_record_metrics,aggregate
from .finrank_quantitative_baseline import digest
from .finrank_eda import table
from .finrank_generation_run import ROOT


def source_labels(records,index):
    labels=defaultdict(set)
    for row in records:
        for passage in row["passages"]:
            if passage["text"].strip():labels[index[passage["text"]]].add((row["ticker"].upper(),str(row["year"]),row["doc_type"].upper()))
        for passage in row["hard_negatives"]:
            if passage["text"].strip():labels[index[passage["text"]]].add((str(passage["ticker"]).upper(),str(passage["year"]),str(passage["doc_type"]).upper()))
    return labels


def match_state(values,target):
    if not values:return "unknown"
    if target not in values:return "wrong"
    return "right" if len(values)==1 else "ambiguous"


def classify(labels,target):
    issuer=match_state({v[0] for v in labels},target[0])
    year_form=match_state({v[1:] for v in labels},target[1:])
    source=match_state(labels,target)
    return dict(issuer=issuer,year_form=year_form,source=source)


def assess(rankings,records,golds,labels):
    details=[];matrix=Counter();exact=[]
    for ranking,row,gold in zip(rankings,records,golds):
        target=(row["ticker"].upper(),str(row["year"]),row["doc_type"].upper())
        states=[classify(labels.get(int(i),set()),target) for i in ranking[:20]]
        first=states[0];cell=first['issuer']+" issuer / "+first['year_form']+" year+form";matrix[cell]+=1
        metrics={}
        for k in [1,5,10,20]:
            for field in ["issuer","year_form","source"]:
                values=[state[field] for state in states[:k]]
                metrics[f"{field}_hit@{k}"]=float("right" in values)
                metrics[f"{field}_possible_hit@{k}"]=float(any(v in {"right","ambiguous"} for v in values))
                metrics[f"{field}_precision@{k}"]=sum(v=="right" for v in values)/k
                metrics[f"{field}_ambiguous_fraction@{k}"]=sum(v in {"ambiguous","unknown"} for v in values)/k
        exact.append(per_record_metrics(ranking,gold))
        details.append(dict(id=row["id"],ticker=row["ticker"],target_source=list(target),top1_cell=cell,top1_is_gold=int(ranking[0]) in gold,
            metrics=metrics,top20=[dict(corpus_id=int(i),sources=[list(t) for t in sorted(labels.get(int(i),set()))],**state,is_gold=int(i) in gold) for i,state in zip(ranking[:20],states)]))
    return dict(source_metrics={key:100*float(np.mean([row['metrics'][key] for row in details])) for key in details[0]['metrics']},gold_passage_metrics={k:aggregate(exact,k) for k in exact[0]},top1_confusion=dict(matrix),per_question=details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir",type=Path,default=ROOT/"data/experiments/finrank_quantitative_baseline")
    parser.add_argument("--trained-dir",type=Path,default=ROOT/"data/experiments/finrank_training/pilot500-v1/training")
    parser.add_argument("--retriever-dir",type=Path,default=ROOT/"data/experiments/finrank_training/pilot500-v1/retriever")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/experiments/finrank_source_recovery")
    args=parser.parse_args();base=args.baseline_dir;out=args.output_dir;manifest=json.loads((base/'run_manifest.json').read_text())
    records=[r for r in RECORDS if r['reasoning_type']=='Quantitative']
    if manifest['query_ids']!=[r['id'] for r in records] or manifest['corpus_sha256']!=digest(TEXT) or manifest['query_sha256']!=digest([r['question'] for r in records]):raise ValueError("Frozen baseline input mismatch")
    labels=source_labels(RECORDS,INDEX);golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    rankings={name:np.load(base/(name+'_rankings.npy')) for name in ['tfidf','bm25','dense']}
    completed={r['id']:r for r in map(json.loads,(base/'cross_encoder.jsonl').read_text().splitlines()) if r['status']=='completed'}
    rankings['ce_dense']=np.stack([np.concatenate([completed[r['id']]['head'],rankings['dense'][i][20:]]) for i,r in enumerate(records)])
    if (args.trained_dir/'finrank_rankings.npy').exists():rankings['trained_ce_dense']=np.load(args.trained_dir/'finrank_rankings.npy')
    retriever_result=args.retriever_dir/'evaluation.json'
    if retriever_result.exists():
        expanded=json.loads(retriever_result.read_text())
        if expanded['protocol']['baseline_manifest']!=manifest:raise ValueError('Retriever evaluation baseline differs')
        for name,path in expanded['ranking_files'].items():rankings[name]=np.load(path)
    results={}
    for name,ranking in rankings.items():
        if ranking.shape!=(len(records),len(TEXT)):raise ValueError("Ranking shape mismatch")
        if not np.all(np.sort(ranking,axis=1)==np.arange(len(TEXT))):raise ValueError("Ranking is not a corpus permutation")
        results[name]=assess(ranking,records,golds,labels)
    # Ensure reconstructed rankings exactly reproduce saved benchmark metrics.
    saved=json.loads((base/'results.json').read_text())
    for name in ['tfidf','bm25','dense','ce_dense']:
        for key,value in saved['models'][name]['metrics'].items():
            if not np.isclose(results[name]['gold_passage_metrics'][key],value):raise ValueError("Gold metric reconstruction mismatch")
    transitions=Counter()
    if 'trained_ce_dense' in results:
        for old,new in zip(results['ce_dense']['per_question'],results['trained_ce_dense']['per_question']):transitions[(old['top1_cell'],new['top1_cell'])]+=1
    output=dict(query_count=len(records),corpus_count=len(TEXT),source_labels='Observed release metadata: issuer, document year, document form. Not fiscal measurement periods, accession IDs or quarters.',
        ambiguous_source_texts=sum(len(v)>1 for v in labels.values()),ambiguous_issuer_texts=sum(len({t[0] for t in v})>1 for v in labels.values()),quarter_recovery='Not available: release has no verified source-quarter field.',
        definitions=dict(issuer_hit='Any definitely correct issuer in top k; no date/form restriction',year_form_hit='Any definitely matching document year/form, ignoring issuer',source_hit='Any definitely matching issuer + document year + document form',possible_hit='Same hit definition allowing source-label ambiguity when target is among observed labels; diagnostic upper bound',precision='Fraction of top k candidates definitely matching the target label',gold_recall='Original macro fraction of gold passages recovered; unlike hit@k, not any-hit',confusion='Top1 issuer status crossed with year+form status, independent of issuer; right/right is also checked against gold passage'),models=results,top1_transitions=[dict(before=k[0],after=k[1],count=v) for k,v in transitions.most_common()])
    out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps(output,indent=2))
    headers=['Method','Issuer Hit@1','Issuer Hit@10','Source Hit@1','Source Hit@10','Gold Recall@10','Gold nDCG@10']
    comparison=[]
    for name,value in results.items():
        m=value['source_metrics'];g=value['gold_passage_metrics'];comparison.append([name,*[f"{x:.2f}" for x in [m['issuer_hit@1'],m['issuer_hit@10'],m['source_hit@1'],m['source_hit@10'],g['recall@10'],g['ndcg@10']]]])
    report=['# FinRank issuer and source-label recovery\n',f"{len(records)} quantitative questions; unchanged {len(TEXT)}-passage corpus. No inference or training performed. All values below are percentages or metrics scaled to 0–100.\n",
        '## Comparison\n',table(headers,comparison),
        'Issuer/source Hit@k means at least one matching result. Gold Recall@k is the mean fraction of gold passages recovered; the denominators differ. Source means issuer + released document year + form, NOT a unique filing accession or verified fiscal quarter. This diagnostic does not establish correct measurement-period matching.\n',
        f"## Metadata ambiguity\n{output['ambiguous_source_texts']} texts have multiple source triples; {output['ambiguous_issuer_texts']} have multiple issuers. A matching label among conflicting labels is marked ambiguous, not automatically right or wrong. Definite hit rates shown above are conservative; possible-hit rates and ambiguous result fractions are in results.json. The target labels come from each benchmark record's metadata, not semantic extraction of its question.\n",
        '## Top-1 confusion\nColumns assess source year/form independently of issuer, allowing right-company/wrong-year and wrong-company/right-year cells.\n']
    for name,value in results.items():
        report.extend([f"### {name}\n",table(['Top-1 category','Questions'],sorted(value['top1_confusion'].items()))])
    if transitions:report.extend(['## Before/after top-1 transitions\n',table(['Original reranker','Trained reranker','Questions'],[(k[0],k[1],v) for k,v in transitions.most_common()])])
    report.extend(['## Limits\n- Positive-passage source labels are inherited from the record; negative labels come from the negative metadata. These release annotations are not independently verified provenance.\n- Identical passage texts can legitimately occur across years; source identity cannot always be inferred from text alone.\n- Correct issuer/year/form does not guarantee the right segment, measure, event status or claim. Gold-passage recovery remains the final relevance diagnostic.\n- Quarter recovery requires an additional trustworthy source mapping; it is deliberately not guessed from the question or filing text.\n'])
    (out/'REPORT.md').write_text('\n'.join(report))
    print(table(headers,comparison));print('Saved',out.resolve())


if __name__=='__main__':main()
