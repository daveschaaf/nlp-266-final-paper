"""Read-only EDA of the local normalized FinRank gold records; no models loaded."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from .finrank_generation_run import ROOT


def words(text):return len(text.split())
def normalized(text):return " ".join(text.lower().split())
def percentile(values,p):
    ordered=sorted(values)
    if not ordered:return None
    index=(len(ordered)-1)*p/100;lo=math.floor(index);hi=math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
def describe(values):
    return dict(n=len(values),min=min(values) if values else None,p10=percentile(values,10),median=percentile(values,50),
                mean=statistics.mean(values) if values else None,p90=percentile(values,90),p95=percentile(values,95),max=max(values) if values else None)
def distribution(rows,key):return dict(Counter(str(r.get(key)) for r in rows).most_common())
def table(headers,rows):
    def cell(x):return str(x).replace("|",r"\|").replace("\n"," ")
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"".join("| "+" | ".join(cell(x) for x in row)+" |\n" for row in rows)


def profile(rows):
    features=[];positives=[];negatives=[];issues=defaultdict(list);questions=defaultdict(list);passage_uses=defaultdict(set);id_text=defaultdict(set)
    for row in rows:
        q=row["question"];ps=row["passages"];ns=row["hard_negatives"];positive_texts={normalized(p["text"]) for p in ps}
        questions[normalized(q)].append(row["id"])
        for p in ps:
            positives.append(p);passage_uses[normalized(p["text"])].add(row["id"]);id_text[p["passage_id"]].add(p["text"])
            if p.get("text_sha1")!=hashlib.sha1(p["text"].encode()).hexdigest():issues["positive_raw_text_sha1_mismatch"].append(dict(id=row["id"],passage_id=p["passage_id"]))
            if p.get("text_sha1")!=hashlib.sha1(p["text"].strip().encode()).hexdigest():issues["positive_trimmed_text_sha1_mismatch"].append(dict(id=row["id"],passage_id=p["passage_id"]))
        negatives.extend(ns)
        if len(ps)!=row["num_passages"]:issues["declared_passage_count_mismatch"].append(row["id"])
        if not q.strip() or not row["answer"].strip():issues["empty_question_or_answer"].append(row["id"])
        if len(positive_texts)!=len(ps):issues["duplicate_positive_text_within_question"].append(row["id"])
        if any(normalized(n["text"]) in positive_texts for n in ns):issues["positive_negative_text_collision"].append(row["id"])
        features.append(dict(id=row["id"],question_id=row["question_id"],ticker=row["ticker"],industry=row["industry"],topic=row["topic"],difficulty=row["difficulty"],reasoning_type=row["reasoning_type"],doc_type=row["doc_type"],year=row["year"],question=q,
            question_words=words(q),answer_words=words(row["answer"]),positive_count=len(ps),negative_count=len(ns),positive_words=sum(words(p["text"]) for p in ps),
            explicit_year=bool(re.search(r"\b(?:19|20)\d{2}\b|FY\s*\d{2,4}",q,re.I)),
            quarter_reference=bool(re.search(r"\bQ[1-4]\b|\bquarter\b|three.month",q,re.I)),
            conjunction=bool(re.search(r"\band\b",q,re.I)),
            causal_language=bool(re.search(r"\bwhy\b|\bdrivers?\b|\breasons?\b|\bexplain\w*\b|what.*(?:led to|caused|contributed)",q,re.I)),
            question_marks=q.count("?"),query_rewrites=len(row.get("query_rewrite",[]))))
    metrics=dict(records=len(rows),issuers=len({r["ticker"] for r in rows}),distributions={k:distribution(rows,k) for k in ["ticker","industry","topic","difficulty","reasoning_type","num_passages","passage_type","doc_type","year"]},
        lengths=dict(question_words=describe([f["question_words"] for f in features]),answer_words=describe([f["answer_words"] for f in features]),
            positive_passage_words=describe([words(p["text"]) for p in positives]),total_evidence_words=describe([f["positive_words"] for f in features]),negative_passage_words=describe([words(p["text"]) for p in negatives])),
        evidence=dict(positive_assignments=len(positives),unique_positive_texts=len(passage_uses),unique_positive_ids=len(id_text),positive_texts_used_by_multiple_questions=sum(len(v)>1 for v in passage_uses.values()),max_questions_per_positive_text=max(map(len,passage_uses.values())),negative_assignments=len(negatives),negative_count_distribution=dict(Counter(str(len(r["hard_negatives"])) for r in rows)),same_issuer_negative_assignments=sum(str(n.get("ticker")).upper()==str(r["ticker"]).upper() for r in rows for n in r["hard_negatives"])),
        lexical_indicators={k:sum(f[k] for f in features) for k in ["explicit_year","quarter_reference","conjunction","causal_language"]},
        quality=dict(duplicate_id_groups=[k for k,v in Counter(r["id"] for r in rows).items() if v>1],duplicate_question_id_groups=[k for k,v in Counter(r["question_id"] for r in rows).items() if v>1],normalized_duplicate_question_groups=[v for v in questions.values() if len(v)>1],passage_id_multiple_texts=[k for k,v in id_text.items() if len(v)>1],issues=dict(issues)),
        groups={key:{value:dict(n=sum(str(r[key])==value for r in rows),single_passage=sum(str(r[key])==value and r["num_passages"]==1 for r in rows),qualitative=sum(str(r[key])==value and r["reasoning_type"]=="Qualitative" for r in rows)) for value in distribution(rows,key)} for key in ["topic","industry","difficulty"]})
    return metrics,features


def write_report(out,metrics,features,source,digest):
    n=metrics["records"];dist=metrics["distributions"];pct=lambda count:f"{100*count/n:.1f}%"
    sections=["# FinRank local dataset EDA\n",f"Source: `{source}`. SHA-256: `{digest}`. Read-only analysis of the local normalized records; these counts are not asserted to describe another release.\n",
        "## Main findings\n",f"- {n:,} questions across {metrics['issuers']} issuers and {len(dist['industry'])} industries.\n- {n-dist['num_passages'].get('1',0):,} ({pct(n-dist['num_passages'].get('1',0))}) questions have multiple gold passages. Passage count is not a count of facts or reasoning steps.\n- {dist['reasoning_type'].get('Qualitative',0):,} ({pct(dist['reasoning_type'].get('Qualitative',0))}) are labeled qualitative.\n- Topics include risk, legal, governance and accounting as well as financials; a financial-number-only generator would cover only part of the benchmark.\n",
        "## Coverage and supplied labels\n"]
    for key in ["industry","ticker","topic","difficulty","reasoning_type","num_passages","doc_type","year"]:
        sections.extend([f"### {key}\n",table(["Value","Questions","Share"],[(k,v,pct(v)) for k,v in dist[key].items()])])
    sections.extend(["## Text lengths\nWhitespace-delimited words; these are not model token counts. Passage statistics are assignment-weighted, so a reused passage appears once per question using it.\n",table(["Text","N","Min","Median","Mean","P90","P95","Max"],[(k,*[round(v[x],1) if isinstance(v[x],float) else v[x] for x in ["n","min","median","mean","p90","p95","max"]]) for k,v in metrics['lengths'].items()])])
    sections.extend(["## Evidence and reuse\n",table(["Measure","Value"],list(metrics['evidence'].items())),"Repeated positive text is not automatically bad: several distinct questions can legitimately retrieve the same passage. Split or evaluate with issuer/document/passage reuse in mind; do not assume records are independent.\n"])
    sections.extend(["## Question-form diagnostics\nThese are lexical indicators, not human labels of atomicity, claim count, answerability or reasoning complexity. A conjunction can join entities instead of tasks. An explicit year need not be the measurement period.\n",table(["Indicator","Questions","Share"],[(k,v,pct(v)) for k,v in metrics['lexical_indicators'].items()])])
    sections.extend(["## Topic and difficulty structure\n",table(["Topic","N","Single-passage","Qualitative"],[(k,v['n'],v['single_passage'],v['qualitative']) for k,v in metrics['groups']['topic'].items()])])
    sections.extend(["## Data quality checks\n",table(["Check","Groups or records"],[(k,len(v)) for k,v in metrics['quality'].items() if k!='issues']+[(k,len(v)) for k,v in metrics['quality']['issues'].items()]),"Raw versus trimmed SHA-1 checks are reported separately: leading/trailing whitespace explains most raw mismatches. Remaining mismatches are unresolved provenance differences, not proven evidence errors. Exact text-hash checks test the stored strings, not whether evidence is correct. Different texts sharing a passage ID may indicate identifiers are document-scoped; inspect before treating that as corruption. Full issue IDs are in `metrics.json`.\n"])
    examples=[features[i] for i in [0,1,5,17,19] if i<len(features)]
    sections.extend(["## Illustrative gold questions\nExamples are illustrative, not a random sample or a quantitative claim about question forms.\n",table(["Question ID","Question","Gold passages"],[(f['question_id'],f['question'],f['positive_count']) for f in examples]),
        "## Implications for synthesis\n- Target one coherent information need, not necessarily one fact. Lists, comparisons, explanations and multipart questions occur in the gold examples.\n- A single-passage synthetic pilot is a deliberate subset: report it as such.\n- Match scope, measurement period and actual versus anticipated events; broad questions are acceptable when all requested information is supported.\n- Evaluate topic coverage, including business structure, risk, legal and governance questions, rather than optimizing only numerical lookups.\n- Keep FinRank gold issuers out of synthetic training; do not use this EDA to tune against held-out gold outcomes.\n- The `year` field is document metadata, not a reliable extraction of the question's measurement period.\n"])
    (out/"REPORT.md").write_text("\n".join(sections))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/FinRank/FinRank.jsonl")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/experiments/finrank_eda")
    args=parser.parse_args();raw=args.dataset.read_bytes();rows=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    metrics,features=profile(rows);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    metrics['source']=str(args.dataset.resolve());metrics['source_sha256']=hashlib.sha256(raw).hexdigest()
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    with (out/"question_features.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(features[0]));writer.writeheader();writer.writerows(features)
    write_report(out,metrics,features,args.dataset.resolve(),metrics['source_sha256'])
    print(json.dumps(dict(records=metrics['records'],issuers=metrics['issuers'],lengths=metrics['lengths'],quality={k:len(v) for k,v in metrics['quality'].items() if k!='issues'},issues={k:len(v) for k,v in metrics['quality']['issues'].items()},output=str(out.resolve())),indent=2))


if __name__=="__main__":main()
