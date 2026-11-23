"""Small fixed-input JEV verdict replay with memory diagnostics; no generation."""
import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import os
from importlib.metadata import version


def snapshot(mx):
    def command(args):
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    return dict(active_gb=mx.get_active_memory()/1e9,
                cached_gb=mx.get_cache_memory()/1e9,
                peak_mlx_gb=mx.get_peak_memory()/1e9,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e9 if sys.platform=='darwin' else 1e6),
                rss_gb=int(command(['ps','-o','rss=','-p',str(os.getpid())]))*1024/1e9,
                swap=command(['sysctl','vm.swapusage']) if sys.platform=='darwin' else None)


def select(source, limit):
    rows=[json.loads(x) for x in (source/'annotations.jsonl').read_text().splitlines()]
    # Reviewed BWA and problematic LH, then other issuers if more requested.
    priority=lambda r: (0 if r['case_id'].startswith('fresh:908255:') else
                        1 if r['case_id'].startswith('fresh:920148:') else 2)
    return sorted(rows,key=priority)[:limit]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--repeat',type=int,default=5,help='Repeat first identical passage to probe accumulation.')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    if args.limit<1 or args.repeat<0:p.error('Invalid limit/repeat')
    pairs=select(args.source_dir,args.limit)
    if not pairs:p.error('No input passages')
    refs={x['case_id']:x for x in map(json.loads,(args.source_dir/'reference_snapshot.jsonl').read_text().splitlines())}
    tasks=[]
    for phase,items in [('repeat',[pairs[0]]*args.repeat),('evaluation',pairs)]:
        for index,row in enumerate(items):
            data=json.loads(row['messages'][1]['content'])
            data['remark_text']=refs[row['case_id']]['remark_text']
            tasks.append(dict(key=f'{phase}:{index}',phase=phase,case_id=row['case_id'],passage_id=row['passage_id'],passage_rank=row['passage_rank'],data=data))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frozen=args.output_dir/'inputs.json'
    config=dict(tasks=tasks,snapshot=str(args.snapshot.resolve()),backend='jev27b_mlx choice readout',labels=['supported','contradicted','insufficient_evidence'])
    if frozen.exists() and json.loads(frozen.read_text())!=config:p.error('Existing frozen inputs/config differ; use a new directory.')
    frozen.write_text(json.dumps(config,indent=2))
    if args.plan_only:
        print(f'Planned {len(pairs)} passages plus {args.repeat} identical repetitions; no model loaded.');return
    import mlx.core as mx
    import jev27b_mlx
    dest=args.output_dir/'decisions.jsonl'
    done={x['key'] for x in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    pending=[t for t in tasks if any(t['key']+':'+c['claim_id'] not in done for c in t['data']['claims'])]
    if not pending:print('DONE: no pending inputs');return
    instruction='''For EACH claim independently choose whether this single filing passage
supports it, contradicts it, or provides insufficient evidence. Preserve original
remark meaning. Compare values, metric, scope, quarter versus YTD, and forecast
versus actual. Unrelated segment/period facts are insufficient, not contradictions.
Allow rounding. Do not resolve unidentified transactions from retrieval. Do not
use the remark as filing evidence. No surrounding discussion or other passages.'''
    print(f'Loading JEV for {len(pending)} passage inputs. Pause other local model workers first.',flush=True)
    started=time.perf_counter();before=snapshot(mx);model=jev27b_mlx.load(str(args.snapshot))
    (args.output_dir/'load_metrics.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,before=before,after=snapshot(mx),jev_version=version('jev27b-mlx'),mlx_version=version('mlx')),indent=2))
    try:
        with dest.open('a') as out:
            for task in pending:
                for claim in task['data']['claims']:
                    row={k:v for k,v in task.items() if k!='data'}
                    row['key']=task['key']+':'+claim['claim_id']
                    if row['key'] in done:continue
                    question=model.question('choice',instruction+'\nClaim to assess: '+claim['statement'],config['labels'],['All material facts supported in this passage.','Incompatible evidence for the same fact.','Missing or mismatched evidence.'])
                    state=json.dumps({k:v for k,v in task['data'].items() if k!='claims'},ensure_ascii=False)
                    row.update(claim=claim,before=snapshot(mx));mx.reset_peak_memory();tick=time.perf_counter()
                    try:
                        result=model.decide(state,question)
                        row.update(result=result,status='completed')
                    except Exception as error:
                        row.update(status='failed',error=f'{type(error).__name__}: {error}')
                    row.update(seconds=time.perf_counter()-tick,after=snapshot(mx))
                    gc.collect();mx.clear_cache();row['after_cleanup']=snapshot(mx)
                    out.write(json.dumps(row)+'\n');out.flush()
                    print(f"{row['key']} {row['status']} {row['seconds']:.2f}s active={row['after']['active_gb']:.2f}GB cache={row['after']['cached_gb']:.2f}GB peak={row['after']['peak_mlx_gb']:.2f}GB",flush=True)
    except KeyboardInterrupt:print('PAUSED: rerun the same command to resume.');return
    print('DONE: decisions and per-claim memory diagnostics saved.')


if __name__=='__main__':main()
