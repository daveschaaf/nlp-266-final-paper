"""Rejudge old questions, regenerate fixed passages, judge new questions, in isolated processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .finrank_generation_run import ROOT
from .finrank_passage_generation import prepare as prepare_generation
from .finrank_two_gate_judge import prepare as prepare_judge


def commands(source, out, snapshot, package):
    judge=["uv","run","--no-project","--with",str(package),"python","-u","-m","dearborn.experiments.finrank_two_gate_judge"]
    return [
        ("rejudge_existing",judge+["--source-dir",str(source),"--output-dir",str(out/"existing_judge"),"--snapshot",str(snapshot)]),
        ("generate_new",[sys.executable,"-u","-m","dearborn.experiments.finrank_passage_generation",
                         "--source-dir",str(source),"--output-dir",str(out/"generation"),"--snapshot",str(snapshot)]),
        ("judge_new",judge+["--source-dir",str(out/"generation"),"--output-dir",str(out/"new_judge"),"--snapshot",str(snapshot)])]


def freeze(path, payload):
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ValueError(f"Existing configuration differs: {path}; use a new cycle directory")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True,help="Existing generation with exported run_plan.json and generation.jsonl")
    p.add_argument("--output-dir",type=Path,required=True,help="New cycle directory; repeat to resume")
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--jev-package",type=Path,default=Path.home()/".cache/uv/git-v0/checkouts/2836f55b3e3abb68/5ef5405")
    p.add_argument("--plan-only",action="store_true")
    a=p.parse_args(argv)
    source=a.source_dir.resolve();out=a.output_dir.resolve();snap=a.snapshot.resolve();package=a.jev_package.resolve()
    if source==out or source in out.parents:p.error("Use a separate cycle directory, not the source directory or its child")
    if not snap.is_dir() or not package.is_dir():p.error("Local JEV snapshot/package missing")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"cycle.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(source=str(source),snapshot=str(snap),jev_package=str(package))
        try:freeze(out/"cycle_config.json",config)
        except ValueError as e:p.error(str(e))
        # Freeze prompts and source inputs BEFORE inference. Resume never changes them.
        generation_path=out/"generation/passage_plan.json"
        old_judge_path=out/"existing_judge/judge_plan.json"
        if not generation_path.exists():
            plan=prepare_generation(source);plan["snapshot"]=str(snap);freeze(generation_path,plan)
        if not old_judge_path.exists():freeze(old_judge_path,prepare_judge(source,snap))
        generation=json.loads(generation_path.read_text());old=json.loads(old_judge_path.read_text())
        print(f"Frozen cycle: {len(old['candidates'])} existing questions; {len(generation['passages'])} unique passages. No original outputs overwritten.",flush=True)
        jobs=commands(source,out,snap,package)
        if a.plan_only:
            for name,cmd in jobs:print(name+": "+shlex.join(cmd))
            print("PLAN ONLY: no inference started.");return 0
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH",""))
        for name,cmd in jobs:
            if name=="judge_new" and not (out/"new_judge/judge_plan.json").exists():
                # Use the same frozen rubric as the existing-question replay.
                new=prepare_judge(out/"generation",snap);new["rubric"]=old["rubric"]
                freeze(out/"new_judge/judge_plan.json",new)
            print(f"STAGE: {name}",flush=True)
            try:subprocess.run(cmd,check=True,cwd=ROOT,env=env)
            except subprocess.CalledProcessError as error:
                print(f"STOPPED: {name} exited {error.returncode}. Rerun this command to resume.",flush=True)
                return error.returncode
            except KeyboardInterrupt:
                print("PAUSED: rerun this command to resume.",flush=True);return 130
        summary={}
        for label,path in [("existing_rejudged",out/"existing_judge/summary.json"),("new_judged",out/"new_judge/summary.json")]:
            data=json.loads(path.read_text());summary[label]=dict(planned=data["planned"],counts=data["counts"],gate_calls=data["gate_calls"])
        summary["generation"]=json.loads((out/"generation/passage_summary.json").read_text())
        summary["note"]="Development comparison, not accuracy: acceptance counts require human review."
        (out/"cycle_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2));print("DONE: generation cycle complete.");return 0


if __name__=="__main__":raise SystemExit(main())
