"""Seeded, section-aware assumed negatives. No model judgments or retrieval selection."""
from collections import Counter,defaultdict
import hashlib
import json
import random
import sqlite3
from dearborn.finrank_policy import assert_not_finrank_tickers,is_finrank_ticker
from dearborn.retrieval.metadata import enrich_passage

CATEGORIES=('different_section','different_period','different_issuer','random')


def normalized(text):
    return ' '.join(text.casefold().split())


def render(passage,metadata=False):
    if not metadata:return passage['text']
    sources=passage.get('sources',[passage])
    return enrich_passage(passage['text'],{(p['ticker'],str(p['filing_year']),p['form']) for p in sources},{p['ticker']:p['name'] for p in sources})


def read_pool(database,issuers):
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        query='''SELECT fc.chunk_id AS passage_id,fc.text,fc.part,fc.item,fc.item_title,
          f.accession_number,f.form,f.filing_year,f.fiscal_year,f.fiscal_period,
          f.fiscal_period_end,c.ticker,c.name,c.industry
          FROM filing_chunk fc JOIN filing f ON f.accession_number=fc.accession_number
          JOIN company c ON c.cik=f.cik ORDER BY fc.chunk_id'''
        return [dict(row) for row in c.execute(query) if row['ticker'] in issuers and not is_finrank_ticker(row['ticker']) and row['text'].strip()]


def period(p):
    return (p['fiscal_year'],p['fiscal_period'])


def same_section(a,b):
    return (a['form'],a['part'],a['item'])==(b['form'],b['part'],b['item'])


def sample_records(records,pool,seed=42):
    assert_not_finrank_tickers([r['ticker'] for r in records])
    splits=defaultdict(set)
    for r in records:splits[r['split']].add(r['ticker'])
    if splits['train'] & splits['validation']:raise ValueError('Issuer split overlaps')
    by_id={p['passage_id']:p for p in pool}
    known_positives=defaultdict(set)
    for r in records:known_positives[(r['ticker'],normalized(r['question']))].add(normalized(r['positive_text']))
    normalized_text={p['passage_id']:normalized(p['text']) for p in pool}
    split_pool={split:[p for p in pool if p['ticker'] in issuers] for split,issuers in splits.items()}
    by_accession=defaultdict(list);by_ticker_section=defaultdict(list);by_industry_section=defaultdict(list)
    for p in pool:
        by_accession[p['accession_number']].append(p)
        by_ticker_section[(p['ticker'],p['form'],p['part'],p['item'])].append(p)
        by_industry_section[(p['industry'],p['form'],p['part'],p['item'])].append(p)
    frozen=[];counts=Counter();missing=Counter();period_matches=Counter()
    # Sampling order and RNG are stable per question, independent of the source row order.
    for r in records:
        if r['passage_id'] not in by_id:raise ValueError(f"Positive missing from corpus: {r['passage_id']}")
        p=by_id[r['passage_id']]
        if p['ticker']!=r['ticker'] or p['text']!=r['positive_text']:raise ValueError('Positive source text or issuer changed')
        if not p['part'] or not p['item']:raise ValueError('Positive section metadata missing')
        available=split_pool[r['split']]
        seen=set(known_positives[(r['ticker'],normalized(r['question']))]);negatives=[]
        rng=random.Random(int(hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest(),16))
        for category in CATEGORIES:
            eligible=[]
            candidates=by_accession[p['accession_number']] if category=='different_section' else by_ticker_section[(p['ticker'],p['form'],p['part'],p['item'])] if category=='different_period' else by_industry_section[(p['industry'],p['form'],p['part'],p['item'])] if category=='different_issuer' else available
            for candidate in candidates:
                if candidate['ticker'] not in splits[r['split']] or normalized_text[candidate['passage_id']] in seen:continue
                if category=='different_section':
                    ok=candidate['accession_number']==p['accession_number'] and (candidate['part'],candidate['item'])!=(p['part'],p['item'])
                elif category=='different_period':
                    ok=candidate['ticker']==p['ticker'] and same_section(candidate,p) and all(period(candidate)) and all(period(p)) and period(candidate)!=period(p)
                elif category=='different_issuer':
                    ok=candidate['ticker']!=p['ticker'] and bool(p['industry']) and candidate['industry']==p['industry'] and same_section(candidate,p)
                else:ok=candidate['ticker']!=p['ticker']
                if ok:eligible.append(candidate)
            if category=='different_issuer':
                matched=[x for x in eligible if all(period(p)) and period(x)==period(p)]
                if matched:eligible=matched;period_matches[r['split']+':matched']+=1
                elif eligible:period_matches[r['split']+':fallback']+=1
            if not eligible:
                missing[r['split']+':'+category]+=1;continue
            chosen=rng.choice(sorted(eligible,key=lambda x:x['passage_id']))
            negatives.append(dict(chosen,category=category));seen.add(normalized(chosen['text']));counts[r['split']+':'+category]+=1
        if not negatives:raise ValueError(f'No eligible negatives for {r["id"]}')
        frozen.append(dict(id=r['id'],question=r['question'],ticker=r['ticker'],split=r['split'],positive=p,negatives=negatives,
            **{k:r[k] for k in ('query_group','provenance') if k in r}))
    stats=dict(selected=dict(counts),unavailable=dict(missing),different_issuer_period_preference=dict(period_matches),unknown_industry_positives=sum(not r['positive']['industry'] for r in frozen))
    return frozen,stats


def validation_pool(records,pool):
    issuers={r['ticker'] for r in records if r['split']=='validation'}
    # Deduplicate exact raw text, retaining stable representative IDs. Gold mapping is by text.
    unique={}
    for p in sorted(pool,key=lambda p:p['passage_id']):
        if p['ticker'] in issuers:
            if p['text'] not in unique:unique[p['text']]=dict(p,sources=[])
            source={k:p[k] for k in ('ticker','name','filing_year','form')}
            if source not in unique[p['text']]['sources']:unique[p['text']]['sources'].append(source)
    passages=list(unique.values());index={p['text']:i for i,p in enumerate(passages)}
    held=[r for r in records if r['split']=='validation']
    golds=[index[r['positive']['text']] for r in held]
    return held,passages,golds


def training_records(records,metadata):
    return [dict(id=r['id'],question=r['question'],split=r['split'],positive_text=render(r['positive'],metadata),negatives=[dict(text=render(n,metadata),category=n.get('category','provided')) for n in r['negatives']]) for r in records]
