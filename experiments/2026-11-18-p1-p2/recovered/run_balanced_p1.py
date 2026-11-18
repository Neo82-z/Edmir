"""Prelabel P1 order control; original blocks retained, no performance-based tuning."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,json,time
import run

HERE=Path(__file__).resolve().parent

def main():
    p=json.loads((HERE/'balanced-p1-plan.json').read_text())
    guard=json.loads((HERE/'balanced-guard.json').read_text());clock=run.Clock(guard)
    if run.sha(__file__)!=guard['balanced_runner_sha256'] or run.sha(HERE/'balanced-p1-plan.json')!=guard['balanced_plan_sha256']:
        raise ValueError('balanced control not sealed')
    if run.sha(HERE/'run.py')!=guard['runner_sha256']:raise ValueError('main runner differs')
    plan,records=run.w1.verify(HERE/'bundle',HERE/'inputs')
    if run.sha(HERE/'bundle/files.json')!=guard['bundle_sha256']:raise ValueError('bundle differs')
    pred=json.loads((HERE/'preregistered-predictions.json').read_text());run.validate_plan(plan,pred)
    if not (HERE/'results/collect-complete.json').exists():raise ValueError('main collection must finish first')
    evidence=[]
    for f in sorted((HERE/'results').glob('collect-[0-9][0-9][0-9].json')):
        r=json.loads(f.read_text())
        if r['job']['shape']=='m512' and any(not x['correctness_passed'] for x in r['rows']):
            raise ValueError('a P1 numerical failure dominates all later admission')
        if r['job']['shape']=='m512' and r['job']['phase']=='check':
            if run.w1.done_raw_changed(f,HERE/'results'):raise ValueError('check changed')
            run.w1.validate_rows(r['rows'],r['job'],plan['p1_candidates'],plan['shapes']['m512'],json.loads((HERE/'results/context.json').read_text())['runtime'])
            evidence.append(r)
    if sorted(r['job']['seed'] for r in evidence)!=sorted(plan['seeds']) or any(not x['correctness_passed'] for r in evidence for x in r['rows']):
        raise ValueError('balanced control needs all P1 correctness checks')
    for a,b in zip(p['jobs'][::2],p['jobs'][1::2]):
        if a['seed']!=b['seed'] or a['order']!=list(reversed(b['order'])):raise ValueError('not paired reverse order')
    a=SimpleNamespace(build_dir=HERE/'build',bundle=HERE/'bundle',inputs=HERE/'inputs',out=HERE/'results-p1-balanced')
    context=run.preflight(a,plan,clock);context.update(balanced_plan=run.sha(HERE/'balanced-p1-plan.json'),balanced_runner=run.sha(__file__))
    a.out.mkdir(exist_ok=True)
    with (a.out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=a.out/'context.json'
        if c.exists() and json.loads(c.read_text())!=context:raise ValueError('balanced resume context changed')
        run.save(c,context);status=dict(state='running',started_epoch=time.time())
        try:
            run.collect(a,dict(plan,jobs=p['jobs']),records,pred,clock,context)
            status['state']='complete'
        except BaseException as e:status.update(state='failed',error=repr(e));raise
        finally:status['finished_epoch']=time.time();run.save(a.out/'collect-status.json',status)

if __name__=='__main__':main()
