"""HyST metadata filters on the 376 FinRank quantitative questions.

Three dense MPNet conditions over the full 5,230-passage pool, in-process Qdrant:
  unfiltered  no filter (the dense baseline)
  oracle      the record's own (ticker, year, doc_type), an upper bound
  hyst        filters extracted from the question by the LLM

Passages excluded by a filter are not ranked, so a filtered-out gold is a miss.
Pool metadata is first-occurrence (FinRank Section 3.6), so even the oracle can
exclude a gold passage. Extracted filters are cached per question so reruns resume.
"""
import argparse
import json
from pathlib import Path
from dearborn.finrank import RECORDS,TEXT,METADATA,INDEX
from dearborn.query_filters import QueryFilters
from dearborn.vector_store import VectorStore
from .finrank_generation_run import ROOT
from .finrank_quantitative_baseline import metrics_for

DEFAULT=ROOT/'data/experiments/finrank_hyst_filter/v1'
MODEL='nemotron6bit'


def records_and_golds():
    records=[r for r in RECORDS if r['reasoning_type']=='Quantitative']
    return records,[{INDEX[p['text']] for p in r['passages'] if p['text'].strip()} for r in records]


def oracle(r):
    return QueryFilters(ticker={'$eq':r['ticker']},year={'$eq':int(r['year'])},doc_type={'$eq':r['doc_type']})


def extract(records,path,model):
    """LLM extractions per question, appended to a JSONL cache; filters are rebuilt from the
    cached extraction so a change to period_filters needs no new model calls."""
    from dearborn.hyst_query import QueryExtraction,period_filters
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);done[row['id']]=period_filters(QueryExtraction(**row['extraction'])).model_dump()
    todo=[r for r in records if r['id'] not in done]
    if todo:
        from dearborn.hyst_query import HySTQuery
        hyst=HySTQuery(model)
        with path.open('a') as handle:
            for i,r in enumerate(todo,1):
                result=hyst.query_filters(r['question']);filters=result.filters.model_dump()
                handle.write(json.dumps(dict(id=r['id'],extraction=result.extraction.model_dump(),filters=filters))+'\n');handle.flush()
                done[r['id']]=filters
                print(f'extract {i}/{len(todo)}',flush=True)
    return {k:QueryFilters.model_validate(v) for k,v in done.items()}


def field_agreement(records,extracted):
    """How often each extracted field equals the record's own metadata."""
    out={}
    for field,truth in [('ticker',lambda r:r['ticker']),('year',lambda r:int(r['year'])),('doc_type',lambda r:r['doc_type'])]:
        present=[(r,getattr(extracted[r['id']],field)) for r in records if getattr(extracted[r['id']],field)]
        exact=sum(f.get('$eq')==truth(r) or truth(r) in (f.get('$in') or []) for r,f in present)
        out[field]=dict(present=len(present),matches_record=exact,of=len(records))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT);parser.add_argument('--model',default=MODEL)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    records,golds=records_and_golds()
    extracted=extract(records,out/'filters.jsonl',args.model)
    store=VectorStore(':memory:','finrank_mpnet');store.upsert(TEXT,METADATA)
    conditions=dict(unfiltered=lambda r:QueryFilters(),oracle=oracle,hyst=lambda r:extracted[r['id']])
    results={};empty={}
    for name,make in conditions.items():
        rankings=[];empty[name]=0
        for r in records:
            ids=[p.id for p in store.retrieve(r['question'],make(r),top_k=len(TEXT))]
            empty[name]+=not ids;rankings.append(ids)
        results[name]=metrics_for(rankings,records,golds)
        print(name,json.dumps(results[name]['metrics']),flush=True)
    summary=dict(model=args.model,questions=len(records),empty_rankings=empty,field_agreement=field_agreement(records,extracted),
        metrics={k:v['metrics'] for k,v in results.items()})
    (out/'results.json').write_text(json.dumps(dict(summary=summary,detail=results),indent=1))
    print(json.dumps(summary,indent=1))


if __name__=='__main__':main()
