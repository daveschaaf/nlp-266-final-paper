"""Resumable passage-centered generation; exports individual pairs for JEV review."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import fcntl

from .finrank_generation_run import ROOT, read, append


def decode_response(raw):
    """Accept fences and one unambiguous missing comma; never alter field values."""
    text=raw.strip()
    if text.startswith("```"):
        match=re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:text=match[1]
    repairs=[]
    try:obj=json.loads(text)
    except json.JSONDecodeError as error:
        # Only an answer STRING immediately followed by the known next key.
        # Never repair truncated text, numbers, quotes, missing fields or values.
        if (error.msg!="Expecting ',' delimiter"
            or not re.match(r'"evidence_quotes"\s*:',text[error.pos:])
            or not re.search(r'"answer"\s*:\s*"(?:[^"\\]|\\.)*"\s*$',text[:error.pos])):
            raise
        obj=json.loads(text[:error.pos]+","+text[error.pos:])
        repairs=[dict(kind="insert_missing_comma_after_answer",character_offset=error.pos)]
    return obj,repairs


def parse_questions(raw, passage, maximum=4):
    obj,_=decode_response(raw)
    if not isinstance(obj,dict) or set(obj)!={"questions"} or not isinstance(obj["questions"],list):
        raise ValueError("Expected questions list")
    seen=set()
    for q in obj["questions"]:
        fields={"question","answer","evidence_quotes","topic","question_form","remark_relation"}
        if not isinstance(q,dict) or set(q)!=fields:raise ValueError("Invalid candidate fields")
        if any(not isinstance(q[k],str) or not q[k].strip() for k in fields-{"evidence_quotes"}):
            raise ValueError("Empty candidate field")
        quotes=q["evidence_quotes"]
        if not isinstance(quotes,list) or not quotes or any(not isinstance(s,str) or not s.strip() for s in quotes):
            raise ValueError("Evidence quotes must be nonempty strings")
        key=" ".join(q["question"].lower().split())
        if key in seen:raise ValueError("Duplicate question")
        seen.add(key)
    return obj["questions"]



def generation_diagnostics(questions, passage, maximum=4):
    """Quote fidelity and requested count are diagnostics, not quality verdicts."""
    return dict(requested_maximum=maximum,generated_count=len(questions),
                exceeds_requested_maximum=len(questions)>maximum,
                remark_relation_checks=[dict(question_index=i,raw=q["remark_relation"],recognized=q["remark_relation"] in {"remark_topic","other_passage_topic"}) for i,q in enumerate(questions)],
                quote_checks=[dict(question_index=i,all_quotes_verbatim=all(s in passage for s in q["evidence_quotes"]),
                                   unmatched_quotes=[s for s in q["evidence_quotes"] if s not in passage])
                              for i,q in enumerate(questions)])


def latest_completed(rows):
    """Latest attempt determines state; an old success cannot hide a newer failure."""
    latest={row["id"]:row for row in rows}
    return {key:row for key,row in latest.items() if row["status"]=="completed"}


def recover_saved_responses(out, plan):
    """Reparse failed raw responses locally, retaining the original failure rows."""
    rows=read(out/"passage_generation.jsonl")
    done=set(latest_completed(rows))
    latest={r["id"]:r for r in rows}
    recovered=0
    with (out/"passage_generation.jsonl").open("a") as handle:
        for group in plan["passages"]:
            old=latest.get(group["id"],{})
            if group["id"] in done or not old.get("raw_response"):continue
            passage=group["parent"]["values"]["passage_text"]
            try:questions=parse_questions(old["raw_response"],passage)
            except (ValueError,TypeError):continue
            append(handle,dict(id=group["id"],status="completed",raw_response=old["raw_response"],questions=questions,
                               diagnostics=generation_diagnostics(questions,passage),syntax_repairs=decode_response(old["raw_response"])[1],recovered_from_saved_response=True,
                               original_generation_seconds=old.get("seconds"),seconds=0.0))
            recovered+=1
    return recovered


def prepare(source):
    original=json.loads((source/"run_plan.json").read_text())
    example_file=ROOT/"prompts/finrank_passage_examples_v2.json"
    examples=json.loads(example_file.read_text())["examples"]
    demos="\n\n".join(json.dumps(dict(company=e["ticker"],shortened_passage=e["passage_excerpt"],
        original_gold_question=e["question"],teaching_note=e["teaching_note"]),ensure_ascii=False) for e in examples)
    prompt_file=ROOT/"prompts/finrank_passage_generation_v3.md"
    system=prompt_file.read_text().replace("{{gold_examples}}",demos)
    groups={}
    for pair in original["pairs"]:
        v=pair["values"]
        key=hashlib.sha256(json.dumps([v["company_name"],v["filing_type"],v["filing_period"],v["passage_text"]]).encode()).hexdigest()
        if key not in groups:groups[key]=dict(id=key,parent=pair,remarks=[],source_ids=[])
        group=groups[key]
        if v["remark_text"] not in group["remarks"]:group["remarks"].append(v["remark_text"])
        group["source_ids"].append(pair["id"])
    for group in groups.values():
        v=group["parent"]["values"]
        user={k:v[k] for k in ["company_name","filing_type","filing_period","passage_text"]}
        user["remarks"]=group["remarks"]
        group["messages"]=[dict(role="system",content=system),dict(role="user",content=json.dumps(user,ensure_ascii=False)+"\n/no_think")]
    return dict(version="passage-generation-v3",prompt_version="standalone-specificity-v3",model="nemotron",source=str(source.resolve()),max_tokens=2048,
                examples=[e["id"] for e in examples],example_provenance=examples,passages=list(groups.values()))


def export(out, plan):
    completed=latest_completed(read(out/"passage_generation.jsonl"))
    pairs=[];rows=[]
    for group in plan["passages"]:
        for i,q in enumerate(completed.get(group["id"],{}).get("questions",[])):
            key=hashlib.sha256((group["id"]+q["question"]).encode()).hexdigest()
            pair=dict(group["parent"],id=key,source_ids=group["source_ids"],passage_group_id=group["id"],generation_details=q,generation_diagnostics=generation_diagnostics([q],group["parent"]["values"]["passage_text"]))
            pairs.append(pair)
            rows.append(dict(id=key,status="completed",output=dict(status="generated",question=q["question"],answer=q["answer"])))
    judge=(ROOT/"prompts/finrank_question_passage_judge_v1.md").read_text()
    (out/"run_plan.json").write_text(json.dumps(dict(version="passage-generation-judge-v1",pairs=pairs,judge_prompt=judge,judge_target="question_passage",snapshot=plan["snapshot"]),indent=2,ensure_ascii=False))
    (out/"generation.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    summary=dict(planned_passages=len(plan["passages"]),completed_passages=len(completed),candidate_questions=len(rows))
    (out/"passage_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path,required=True,help="Local JEV snapshot recorded for later judging; not loaded here")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--reparse-saved",action="store_true",help="Explicitly reparse saved failed responses without new inference when possible; default retries failures through the model.")
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/"passage_runner.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/"passage_plan.json"
        if path.exists():
            plan=json.loads(path.read_text())
            if plan["source"]!=str(args.source_dir.resolve()) or plan["snapshot"]!=str(args.snapshot.resolve()):
                parser.error("Resume inputs differ; use the original options or a new directory")
        else:
            if (out/"run_plan.json").exists():parser.error("Use a new output directory")
            plan=prepare(args.source_dir);plan["snapshot"]=str(args.snapshot.resolve())
            path.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        print(f"Frozen {len(plan['passages'])} unique passages; prompt={plan.get('prompt_version', plan['version'])}; requesting up to four questions each.",flush=True)
        if args.plan_only:return
        if args.reparse_saved:
            recovered=recover_saved_responses(out,plan)
            print(f"Reparse mode: recovered {recovered} saved responses without inference.",flush=True)
        done=set(latest_completed(read(out/"passage_generation.jsonl")))
        pending=[g for g in plan["passages"] if g["id"] not in done]
        print(f"Resume: skipping {len(done)} completed passages; {len(pending)} passages require NEW model inference.",flush=True)
        if not pending:print("No inference needed. Use a new output directory for fresh generation of completed passages.",flush=True)
        if pending:
            from dearborn.labeling.models import LabelingModelLoader
            from dearborn.labeling.evaluation_run import generate_response
            model=LabelingModelLoader(local_files_only=True).load(plan["model"])
            with (out/"passage_generation.jsonl").open("a") as handle:
                for group in pending:
                    row=dict(id=group["id"]);start=time.perf_counter()
                    try:
                        row["raw_response"]=generate_response(model,group["messages"],max_new_tokens=plan["max_tokens"])
                        passage=group["parent"]["values"]["passage_text"]
                        questions=parse_questions(row["raw_response"],passage)
                        row.update(status="completed",questions=questions,diagnostics=generation_diagnostics(questions,passage),syntax_repairs=decode_response(row["raw_response"])[1])
                    except Exception as error:row.update(status="failed",error=f"{type(error).__name__}: {error}")
                    row["seconds"]=time.perf_counter()-start;append(handle,row)
                    print(f"{group['id'][:12]}: {row['status']} questions={len(row.get('questions',[]))} {row['seconds']:.1f}s",flush=True)
                    if row["status"]=="failed":print(f"  {row['error']}",flush=True)
        export(out,plan)
        completed=set(latest_completed(read(out/"passage_generation.jsonl")))
        if len(completed)<len(plan["passages"]):raise SystemExit("INCOMPLETE: rerun the same command to retry failures")
        print("DONE: candidates exported; no judge or training run started.")


if __name__=="__main__":main()
