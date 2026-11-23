"""Ragas 0.4.3 comparison on saved pairs using local Qwen MLX + Outlines JSON decoding."""
import argparse
import asyncio
import fcntl
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("RAGAS_DO_NOT_TRACK","true")
from .finrank_generation_run import ROOT,read,append
from .finrank_generation_cycle import freeze
from .finrank_two_gate_judge import prepare as prepare_candidates

METRICS=("context_relevance","faithfulness")


def prepare(source,limit=None):
    candidates=prepare_candidates(source,Path("."))["candidates"]
    answers={r["id"]:r["output"]["answer"] for r in read(source/"generation.jsonl") if r["status"]=="completed" and r["output"]["status"]=="generated"}
    rows=[dict(c,diagnostic_answer=answers[c["id"]]) for c in candidates]
    return rows if limit is None else rows[:limit]


def valid_structured_result(response_model,raw):
    result=response_model.model_validate_json(raw)
    if hasattr(result,"rating") and result.rating not in (0,1,2):raise ValueError("Ragas relevance rating outside 0/1/2")
    if response_model.__name__=="NLIStatementOutput":
        if not result.statements or any(s.verdict not in (0,1) for s in result.statements):raise ValueError("Invalid NLI verdicts")
    return result


def make_adapter(generate,save_call):
    """Actual Ragas base interface; generation is injectable for no-model tests."""
    from ragas.llms.base import InstructorBaseRagasLLM
    class LocalRagasAdapter(InstructorBaseRagasLLM):
        def __init__(self):self.failed_calls=0
        def generate(self,prompt,response_model):
            start=time.perf_counter();row=dict(prompt=prompt,response_schema=response_model.model_json_schema(),schema_name=response_model.__name__)
            try:
                raw=generate(prompt,response_model);row["raw_response"]=raw
                result=valid_structured_result(response_model,raw)
                row.update(status="completed",output=result.model_dump())
                return result
            except Exception as error:
                self.failed_calls+=1;row.update(status="failed",error=f"{type(error).__name__}: {error}");raise
            finally:
                row["seconds"]=time.perf_counter()-start;save_call(row)
        async def agenerate(self,prompt,response_model):
            # Deliberately sequential: no concurrent model requests on unified memory.
            return self.generate(prompt,response_model)
    return LocalRagasAdapter()


class StructuredQwen:
    def __init__(self,max_tokens,model_name="qwen"):
        import outlines
        from dearborn.labeling.models import LabelingModelLoader
        self.loaded=LabelingModelLoader(local_files_only=True).load(model_name)
        self.wrapped=outlines.from_mlxlm(self.loaded.model,self.loaded.input_processor)
        # We apply Qwen's template ourselves so thinking is explicitly disabled.
        # Outlines must not apply a second chat template around that prompt.
        self.wrapped.type_adapter.has_chat_template=False
        self.max_tokens=max_tokens;self.generators={}
    def __call__(self,prompt,response_model):
        import outlines
        from mlx_lm.sample_utils import make_sampler
        if response_model not in self.generators:
            self.generators[response_model]=outlines.Generator(self.wrapped,response_model)
        text=self.loaded.input_processor.apply_chat_template(
            [dict(role="user",content=prompt+("\n/no_think" if self.loaded.name.startswith("nemotron") else ""))],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return self.generators[response_model](text,max_tokens=self.max_tokens,sampler=make_sampler(temp=0),verbose=False)


def completed(rows):
    latest={(r["id"],r["metric"]):r for r in rows}
    return {k:r for k,r in latest.items() if r["status"]=="completed"}


def report(out,plan):
    done=completed(read(out/"metrics.jsonl"));rows=[]
    for c in plan["candidates"]:
        scores={m:done[c["id"],m]["score"] for m in METRICS if (c["id"],m) in done}
        rows.append(dict(id=c["id"],question=c["question"],scores=scores,completed=len(scores)==2))
    summary=dict(planned=len(rows),completed=sum(r["completed"] for r in rows),
                 pending=sum(not r["completed"] for r in rows),results=rows,
                 note="Context relevance is the question–passage metric. Faithfulness grades the existing diagnostic answer, not pair validity. No acceptance threshold calibrated.")
    for metric in METRICS:
        values=[r["scores"][metric] for r in rows if metric in r["scores"]]
        summary[metric]=dict(scored=len(values),mean=sum(values)/len(values) if values else None,
                            distribution={str(v):values.count(v) for v in sorted(set(values))})
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in {"results","note"}}),flush=True)
    return summary


async def run_metrics(out,plan):
    from ragas.metrics.collections import ContextRelevance,Faithfulness
    import mlx.core as mx
    from .jev_matching_diagnostic import snapshot
    done=completed(read(out/"metrics.jsonl"))
    tasks=[(c,m) for c in plan["candidates"] for m in METRICS if (c["id"],m) not in done]
    if not tasks:return
    print(f"Loading cached Qwen3-8B through MLX; {len(tasks)} pending Ragas metrics; schema-constrained decoding.",flush=True)
    generate=StructuredQwen(plan["max_tokens"])
    runtime=dict(versions={p:version(p) for p in ["ragas","langchain-community","mlx-lm","outlines"]},model_id=generate.loaded.model_id)
    if (out/"runtime.json").exists() and json.loads((out/"runtime.json").read_text())!=runtime:raise ValueError("Runtime changed; use a new output directory")
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2))
    with (out/"calls.jsonl").open("a") as calls,(out/"metrics.jsonl").open("a") as handle:
        active={}
        def save_call(row):
            row.update(active);row["memory"]=snapshot(mx);append(calls,row)
            print(f"  {row['schema_name']}: {row['status']} {row['seconds']:.1f}s",flush=True)
        llm=make_adapter(generate,save_call)
        metrics=dict(context_relevance=ContextRelevance(llm=llm,max_retries=1),faithfulness=Faithfulness(llm=llm))
        for index,(c,metric) in enumerate(tasks,1):
            active.clear();active.update(id=c["id"],metric=metric)
            print(f"Metric {index}/{len(tasks)} {c['id'][:12]} {metric}",flush=True)
            start=time.perf_counter();before_failures=llm.failed_calls;mx.reset_peak_memory()
            row=dict(id=c["id"],metric=metric)
            try:
                args=dict(user_input=c["question"],retrieved_contexts=[c["passage"]])
                if metric=="faithfulness":args["response"]=c["diagnostic_answer"]
                result=await metrics[metric].ascore(**args)
                score=float(result.value)
                if llm.failed_calls!=before_failures or not math.isfinite(score) or not 0<=score<=1:
                    raise ValueError("Ragas metric incomplete/invalid; inspect saved call errors")
                row.update(status="completed",score=score)
            except Exception as error:row.update(status="failed",error=f"{type(error).__name__}: {error}")
            row.update(seconds=time.perf_counter()-start,memory=snapshot(mx));append(handle,row)
            print(f"  RESULT: {row.get('score',row.get('error'))}",flush=True)
            gc.collect();mx.clear_cache()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--max-new-tokens",type=int,default=1536)
    parser.add_argument("--plan-only",action="store_true")
    a=parser.parse_args(argv);source=a.source_dir.resolve();out=a.output_dir.resolve()
    if a.max_new_tokens<1 or (a.limit is not None and a.limit<1):parser.error("Limits must be positive")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"runner.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=out/"plan.json"
        if path.exists():
            plan=json.loads(path.read_text())
            if (plan["source"],plan["limit"],plan["max_tokens"])!=(str(source),a.limit,a.max_new_tokens):parser.error("Resume inputs differ")
        else:
            candidates=prepare(source,a.limit)
            if not candidates:parser.error("No generated questions")
            plan=dict(version="qwen-ragas-v1",source=str(source),limit=a.limit,max_tokens=a.max_new_tokens,
                      ragas_version="0.4.3",candidates=candidates)
            freeze(path,plan)
        print(f"Frozen {len(plan['candidates'])} pairs; stock Ragas context relevance + diagnostic answer faithfulness.",flush=True)
        if a.plan_only:print("PLAN ONLY: no model loaded.");return 0
        if version("ragas")!=plan["ragas_version"]:parser.error("Use ragas==0.4.3 for this frozen run")
        try:asyncio.run(run_metrics(out,plan))
        except KeyboardInterrupt:report(out,plan);print("PAUSED: rerun same command to resume.");return 130
        summary=report(out,plan)
        if summary["pending"]:print("INCOMPLETE: rerun to retry failed metrics.");return 1
        print("DONE: Qwen–Ragas scores and intermediate claims/verdicts saved.");return 0


if __name__=="__main__":raise SystemExit(main())
