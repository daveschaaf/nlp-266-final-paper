import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
pilot=root/'data/experiments/finrank_training/pilot500-v1'
read=lambda p: json.loads(p.read_text())
runs=[root / x for x in read(pilot/'pilot_plan.json')['seed_runs']]
runs.extend(sorted((pilot/'batches').glob('batch-*')))
def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()
import argparse
parser=argparse.ArgumentParser(description='Export the frozen generation outcomes')
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
with args.output.open('x',newline='') as f:
    w=csv.writer(f)
    w.writerow(['run','case_id','passage_id','remark_text_hash','status','accepted_question_hash'])
    for run in runs:
        if not (run/'summary.json').exists():
            continue
        results={r['id']:r for r in read(run/'summary.json')['results']}
        for case in read(run/'plan.json')['cases']:
            result=results[case['id']]
            status=result['status']
            q=digest(' '.join(result['question']['question'].casefold().split())) if status=='accepted' else ''
            for remark in case['remarks']:
                w.writerow([run.name,case['id'],case['passage_id'],digest(remark),status,q])
