"""Sequential, resumable JEV choice judgments for generated question–passage pairs."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time

from dearborn.experiments.finrank_generation_run import ROOT, append, read

GATES=("usable_question","evidence_scope")
LABELS=["pass","fail","uncertain"]


def final_status(choices):
    for gate in GATES:
        if gate not in choices:return "pending"
        if choices[gate]=="fail":return "rejected"
        if choices[gate]=="uncertain":return "review"
    return "accepted"


def evaluate_candidate(candidate, completed, decide, save):
    """Save every successful gate immediately; never run gate 2 after non-pass."""
    choices={}
    for gate in GATES:
        key=(candidate["id"],gate)
        choice=completed.get(key)
        if choice is None:
            choice=decide(candidate,gate)
            if choice not in LABELS:raise ValueError("Unexpected judge choice")
            save(candidate,gate,choice)
            completed[key]=choice
        choices[gate]=choice
        if choice!="pass":break
    return final_status(choices)


def prepare(source,snapshot):
    original=json.loads((source/"run_plan.json").read_text())
    generations={r["id"]:r for r in read(source/"generation.jsonl") if r["status"]=="completed"}
    candidates=[]
    for pair in original["pairs"]:
        generation=generations.get(pair["id"])
        if not generation or generation["output"]["status"]!="generated":continue
        v=pair["values"]
        candidates.append(dict(id=pair["id"],passage_id=pair["passage_id"],
            question=generation["output"]["question"],passage=v["passage_text"],
            company=v["company_name"],filing_type=v["filing_type"],filing_period=v["filing_period"]))
    if len({c["id"] for c in candidates})!=len(candidates):raise ValueError("Duplicate candidate IDs")
    return dict(version="two-gate-judge-v2",source=str(source.resolve()),snapshot=str(snapshot.resolve()),
        rubric=json.loads((ROOT/"prompts/finrank_two_gate_judge_v2.json").read_text()),candidates=candidates)


def report(out,plan):
    completed={(r["id"],r["gate"]):r["choice"] for r in read(out/"decisions.jsonl") if r["status"]=="completed"}
    results=[dict(id=c["id"],question=c["question"],status=final_status({g:completed[c["id"],g] for g in GATES if (c["id"],g) in completed}),
                  gates={g:completed[c["id"],g] for g in GATES if (c["id"],g) in completed}) for c in plan["candidates"]]
    summary=dict(planned=len(results),counts={s:sum(r["status"]==s for r in results) for s in ["accepted","rejected","review","pending"]},
                 gate_calls=len(completed),results=results)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary["counts"]),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--plan-only",action="store_true")
    a=p.parse_args(argv);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/"judge.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=out/"judge_plan.json"
        if frozen.exists():
            plan=json.loads(frozen.read_text())
            if plan["source"]!=str(a.source_dir.resolve()) or plan["snapshot"]!=str(a.snapshot.resolve()):p.error("Resume inputs differ")
        else:
            plan=prepare(a.source_dir,a.snapshot)
            if not plan["candidates"]:p.error("No generated candidates to judge")
            frozen.write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        if a.plan_only:
            print(f"Planned {len(plan['candidates'])} candidates; no model loaded.");return
        completed={(r["id"],r["gate"]):r["choice"] for r in read(out/"decisions.jsonl") if r["status"]=="completed"}
        pending=[c for c in plan["candidates"] if final_status({g:completed[c["id"],g] for g in GATES if (c["id"],g) in completed})=="pending"]
        if not pending:report(out,plan);print("DONE");return
        import mlx.core as mx
        import jev27b_mlx
        from dearborn.experiments.jev_matching_diagnostic import snapshot
        print(f"Loading JEV for {len(pending)} candidates. Gate 2 runs only after gate 1 passes.",flush=True)
        model=jev27b_mlx.load(plan["snapshot"])
        questions={g:model.question("choice",plan["rubric"]["shared"]+"\n"+plan["rubric"][g],LABELS) for g in GATES}
        with (out/"decisions.jsonl").open("a") as handle:
            metrics={}
            def decide(candidate,gate):
                start=time.perf_counter();mx.reset_peak_memory()
                result=model.decide(json.dumps({k:v for k,v in candidate.items() if k not in {"id","passage_id"}},ensure_ascii=False),questions[gate])
                metrics.clear();metrics.update(result=result,seconds=time.perf_counter()-start,memory=snapshot(mx))
                return result["choice"]
            def save(candidate,gate,choice):
                append(handle,dict(id=candidate["id"],gate=gate,status="completed",choice=choice,**metrics))
                print(f"{candidate['id'][:12]} {gate}: {choice} ({metrics['seconds']:.1f}s)",flush=True)
            try:
                for candidate in pending:
                    try:evaluate_candidate(candidate,completed,decide,save)
                    except Exception as error:
                        append(handle,dict(id=candidate["id"],status="failed",error=f"{type(error).__name__}: {error}"))
                        print(f"{candidate['id'][:12]} failed: {error}",flush=True)
                    finally:gc.collect();mx.clear_cache()
            except KeyboardInterrupt:
                report(out,plan);print("PAUSED: rerun the same command to resume.");return 130
        summary=report(out,plan)
        if summary["counts"]["pending"]:print("INCOMPLETE: rerun to retry failures.");return 1
        print("DONE: two-gate judgments saved.");return 0


if __name__=="__main__":raise SystemExit(main())
