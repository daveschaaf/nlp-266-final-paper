"""Lexical company/time diagnostics plus explicitly annotated request structures."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from .finrank_generation_run import ROOT
from .finrank_eda import table

ALIASES={"ABBV":["AbbVie"],"AMGN":["Amgen"],"COP":["ConocoPhillips"],"CTRA":["Coterra"],"CVX":["Chevron"],"DVN":["Devon"],"EOG":["EOG Resources"],"EPD":["Enterprise Products","Enterprise"],"ET":["Energy Transfer"],"F":["Ford"],"GM":["General Motors","Generals Motors"],"JNJ":["Johnson & Johnson","Johnson and Johnson","Johnson Johnson","J&J"],"KMI":["Kinder Morgan"],"LLY":["Eli Lilly","Lilly"],"LNG":["Cheniere"],"MPC":["Marathon Petroleum"],"MRK":["Merck"],"MRNA":["Moderna"],"PFE":["Pfizer"],"TSLA":["Tesla"],"WMB":["Williams"],"XOM":["ExxonMobil","Exxon Mobil","Exxon"]}
YEAR=r"\b(?:19|20)\d{2}\b|\bFY\s*\d{2,4}\b"
RELATIVE=r"\b(?:recent(?:ly)?|latest|current|previous|prior|ongoing|future|next year|last year|past (?:\w+ )?years|last (?:\w+ )?years|most recent|year.over.year|yoy|y/y)\b"


def company_reference(question,ticker):
    names=[name for name in ALIASES[ticker] if re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",question,re.I)]
    tick=bool(re.search(r"(?<![A-Za-z])"+re.escape(ticker)+r"(?:[’']s|s)?(?![A-Za-z])",question))
    if names:return "company_name_detected"
    if tick:return "ticker_detected"
    return "no_target_name_or_ticker_detected"


def time_reference(question):
    years=re.findall(YEAR,question,re.I);relative=re.findall(RELATIVE,question,re.I)
    return dict(time_form="absolute_and_relative" if years and relative else "absolute_year_detected" if years else "relative_only" if relative else "no_year_or_relative_marker_detected",year_mentions=years,relative_markers=relative,
        quarter_marker=bool(re.search(r"\bQ[1-4]\b|\bquarter\b|three.month",question,re.I)))


def analyze(rows):
    output=[]
    for r in rows:
        q=r['question'];ref=company_reference(q,r['ticker']);other=[k for k in ALIASES if k!=r['ticker'] and company_reference(q,k)!="no_target_name_or_ticker_detected"]
        output.append(dict(id=r['id'],question_id=r['question_id'],ticker=r['ticker'],topic=r['topic'],difficulty=r['difficulty'],question=q,
            company_reference=ref,other_issuer_markers=other,**time_reference(q)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--dataset",type=Path,default=ROOT/"data/FinRank/FinRank.jsonl");parser.add_argument("--output-dir",type=Path,default=ROOT/"data/experiments/finrank_eda")
    a=parser.parse_args();rows=[json.loads(l) for l in a.dataset.read_text().splitlines() if l.strip()];features=analyze(rows);out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    annotations=json.loads((out/"question_deconstruction_annotations.json").read_text());by={r['id']:r for r in rows};n=len(rows)
    metrics=dict(n=n,company_reference=dict(Counter(r['company_reference'] for r in features)),time_reference=dict(Counter(r['time_form'] for r in features)),quarter_markers=sum(r['quarter_marker'] for r in features),other_issuer_markers=[r for r in features if r['other_issuer_markers']],manual_sample_request_counts=dict(Counter(r['request_count'] for r in annotations['rows'])))
    (out/"question_structure_metrics.json").write_text(json.dumps(metrics,indent=2))
    with (out/"question_structure.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(features[0]));writer.writeheader();writer.writerows(features)
    report=["# FinRank question deconstruction\n","This analyzes question wording, not whether gold evidence is sufficient. Full-dataset company/time categories are lexical diagnostics; request counts below are assistant-proposed readings of 24 selected examples.\n","## Company references\n",table(["Detected form","Questions","Share"],[(k,v,f"{v/n:.1%}") for k,v in metrics['company_reference'].items()]),"Aliases and tickers are detected with an explicit dictionary. No detected name does not mean no company context: questions can identify a subsidiary, transaction, product, management or simply rely on the source filing. Other-issuer markers are review flags, not proven annotation errors.\n","## Time references\n",table(["Detected form","Questions","Share"],[(k,v,f"{v/n:.1%}") for k,v in metrics['time_reference'].items()]),"These are overlapping concepts collapsed into four lexical categories. A year may identify a document, a measurement period, an event or a comparison. No year does not imply a defective question. Hypothetical and qualitative questions often have no fixed measurement period. Quarter marker count: "+str(metrics['quarter_markers'])+".\n","## Amount of requested information\n","A request unit is a separately requested deliverable. It is NOT an atomic fact, answer item, source passage or operand. 'List the segments' is one request with several items. 'Compute a ratio' is one request requiring several facts. 'Describe change and explain its drivers' has two deliverables. Interpretation remains subjective.\n",table(["Requests per sampled question","Sample questions"],sorted(metrics['manual_sample_request_counts'].items())),"The 24-example set covers all eight topics and three difficulties, across "+str(len({by[r['id']]['ticker'] for r in annotations['rows']}))+" issuers. It is deliberately balanced and cannot establish population proportions.\n","## Deconstructed examples\n"]
    for ann in annotations['rows']:
        row=by[ann['id']]
        report.extend([f"### {row['ticker']} · {row['topic']} · {row['difficulty']}\n","> "+row['question']+"\n",f"- Company: {ann['company_reference']}\n- Time: {ann['time_reference']}\n- Requested outputs ({ann['request_count']}): "+"; ".join(ann['requested_outputs'])+f"\n- Interpretation: {ann['notes']}\n- Gold evidence passages: {row['num_passages']}. Record: `{row['id']}`.\n"])
    report.extend(["## Implications for our generation and acceptance checks\n","1. Company identity can come from a name, ticker, subsidiary or transaction; decide explicitly whether issuer metadata accompanies the question at retrieval time.\n2. Resolve the TYPE of time reference: document year, measurement period, event date, comparison window or relative reference. Do not turn all of them into the filing quarter.\n3. Preserve all requested deliverables and their scope. Do not enforce one fact or one sentence as a benchmark requirement.\n4. A broad question is acceptable when its complete information need is supported; unsupported implied events, wrong periods and changed business scope remain errors.\n5. Some gold questions require calculations and strategic interpretation. Directly stated numbers are a subset, not the entire task.\n"])
    (out/"QUESTION_DECONSTRUCTION.md").write_text("\n".join(report))
    print(json.dumps({k:v for k,v in metrics.items() if k!='other_issuer_markers'},indent=2));print('Other-issuer marker flags:',len(metrics['other_issuer_markers']))


if __name__=="__main__":main()
