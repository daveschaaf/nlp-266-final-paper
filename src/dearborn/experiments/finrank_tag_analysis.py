"""Compare all existing FinRank tags on the 376 questions with saved rankings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from dearborn.finrank import TEXT,INDEX,per_record_metrics,aggregate
from .finrank_financials_benchmark import load_saved_rankings
from .finrank_eda import table
from .finrank_generation_run import ROOT
from dearborn.training.finrank_metadata_training import sha
from dearborn.training.finrank_pilot import atomic_json

DEFAULT=ROOT/'data/experiments/finrank_tag_analysis'
FIELDS=('topic','industry','difficulty','passage_type','num_passages','doc_type','year','ticker')
BASELINE='Original MPNet + MS MARCO'
COMPARISONS=('Metadata only','Question training only','Metadata + question training','Metadata + remark training')
METRICS=('recall@1','recall@5','recall@10','recall@20','mrr','ndcg@10')


def summarize(rows,records,fields=FIELDS):
    if len(rows)!=len(records):raise ValueError('Question rows do not align')
    groups={}
    for field in fields:
        groups[field]={}
        for value in sorted({str(r[field]) for r in records}):
            selected=[row for row,r in zip(rows,records) if str(r[field])==value]
            groups[field][value]=dict(n=len(selected),**{key:aggregate(selected,key) for key in METRICS})
    return groups


def run(out):
    records,rankings,manifest,sources,trace,protocols=load_saved_rankings()
    inputs={str(p.resolve()):sha(p) for p in [*sources.values(),trace,*protocols]}
    plan=dict(query_count=len(records),corpus_count=len(TEXT),baseline_manifest=manifest,fields=list(FIELDS),input_sha256=inputs,scope='All existing tag values among the 376 quantitative queries; qualitative queries do not have saved rankings in this experiment.',note='Exploratory post-hoc slices. Overlapping groups, small samples, and multiple comparisons; no significance or new holdout claims.')
    out.mkdir(parents=True,exist_ok=True);path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Analysis inputs changed; use a new output directory')
    atomic_json(path,plan)
    golds=[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]
    results={}
    for name,ranking in rankings.items():
        if ranking.shape!=(len(records),len(TEXT)) or not np.all(np.sort(ranking,axis=1)==np.arange(len(TEXT))):raise ValueError(f'Invalid ranking: {name}')
        rows=[dict(id=r['id'],**per_record_metrics(ranking[i],golds[i])) for i,r in enumerate(records)]
        results[name]=dict(overall={key:aggregate(rows,key) for key in METRICS},groups=summarize(rows,records),per_question=rows)
    differences=[]
    for field in FIELDS:
        for value,baseline in results[BASELINE]['groups'][field].items():
            for name in COMPARISONS:
                m=results[name]['groups'][field][value];enriched=results['Metadata only']['groups'][field][value]
                differences.append(dict(field=field,value=value,configuration=name,**m,**{f'delta_baseline_{k}':m[k]-baseline[k] for k in METRICS},**{f'delta_metadata_{k}':m[k]-enriched[k] for k in METRICS}))
    atomic_json(out/'results.json',dict(plan=plan,models=results,comparisons=differences))
    with (out/'comparisons.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(differences[0]));writer.writeheader();writer.writerows(differences)
    report=['# FinRank performance by existing tags\n',plan['scope']+'\n',plan['note']+'\n','All numbers use the original macro gold-passage protocol, scaled to 0–100. The full 5,230-passage corpus and each saved candidate order remain unchanged.\n']
    for field in FIELDS:
        report.append(f'## {field}\n')
        for metric in ['recall@5','recall@20','ndcg@10']:
            rows=[]
            for value,b in results[BASELINE]['groups'][field].items():
                rows.append([value,b['n'],f'{b[metric]:.2f}',*[f'{results[name]["groups"][field][value][metric]:.2f}' for name in COMPARISONS]])
            report += [f'### {metric}\n',table([field,'N','Original','Metadata','Question train','Metadata + question','Metadata + remark'],rows)]
    for name in COMPARISONS[1:]:
        wins=[r for r in differences if r['configuration']==name and r['n']>=20 and r['delta_baseline_recall@5']>0 and r['delta_baseline_ndcg@10']>0]
        wins.sort(key=lambda r:r['delta_baseline_ndcg@10'],reverse=True)
        report += [f'## {name}: slices improving both R@5 and nDCG@10 over original, N >= 20\n',table(['Tag','Value','N','Delta R@5','Delta nDCG@10','Delta nDCG vs metadata'],[[r['field'],r['value'],r['n'],f'{r["delta_baseline_recall@5"]:+.2f}',f'{r["delta_baseline_ndcg@10"]:+.2f}',f'{r["delta_metadata_ndcg@10"]:+.2f}'] for r in wins]) if wins else 'None.\n']
    (out/'REPORT.md').write_text('\n'.join(report))
    print('TOPIC nDCG@10')
    print(table(['Topic','N','Original','Metadata','Question train','Metadata + question','Metadata + remark'],[[value,b['n'],f'{b["ndcg@10"]:.2f}',*[f'{results[name]["groups"]["topic"][value]["ndcg@10"]:.2f}' for name in COMPARISONS]] for value,b in results[BASELINE]['groups']['topic'].items()]))
    print(f'DONE: all quantitative tag slices saved to {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();run(args.output_dir.resolve())


if __name__=='__main__':main()
