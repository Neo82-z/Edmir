"""Authorized time extension only; reuse sealed collector and unchanged context."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

def validate_guard(g, old, now):
    if not all(g.get(k) is True for k in ('authorization_confirmed','platform_verified','time_limit_removed_by_user')):
        raise ValueError('Missing explicit time extension/platform confirmation')
    if g['instance_id'] != old['instance_id'] or g['billed_from_epoch'] != old['billed_from_epoch']:
        raise ValueError('Instance or original conservative cost start changed')
    keys=('billed_from_epoch','hourly_total_usd','budget_usd','platform_stop_epoch','recovery_reserve_seconds','platform_spend_cap_usd')
    if any(type(g.get(k)) not in (int,float) or not math.isfinite(g[k]) for k in keys):
        raise ValueError('Invalid guard value')
    start,end=g['billed_from_epoch'],g['platform_stop_epoch']
    if not start < now < end or not 0 < g['budget_usd'] <= 15 or g['hourly_total_usd'] != old['hourly_total_usd']:
        raise ValueError('Invalid time/rate/budget')
    if (end-start)*g['hourly_total_usd']/3600 > g['budget_usd']:
        raise ValueError('Conservative full elapsed cost exceeds USD15')
    if not 0 < g['platform_spend_cap_usd'] <= g['budget_usd'] or g['recovery_reserve_seconds'] < 180:
        raise ValueError('Insufficient recovery reserve or spend protection')
    cutoff=end-g['recovery_reserve_seconds']
    if not now < cutoff <= now+3600:
        raise ValueError('Invalid resumed collection deadline')
    return cutoff

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace'))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    spec=importlib.util.spec_from_file_location('sealed_window',root/'window/run_window.py')
    window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
    original=json.loads((root/'results-precision/intent.json').read_text())
    guard=json.loads((root/'extension-guard.json').read_text())
    cutoff=validate_guard(guard,original['guard'],time.time())
    bundle=root/'precision-bundle'
    plan=json.loads((root/'window/queue.json').read_text())
    source=json.loads((bundle/'plan.json').read_text())
    window.validate_plan(plan,source,bundle)
    context=original['context']
    assert plan==original['plan']
    assert sha(root/'window/run_window.py')==context['window_identity']['runner']
    assert sha(root/'window/queue.json')==context['window_identity']['queue']
    if args.dry_run:
        print(json.dumps({'guard_valid':True,'original_runner_and_queue_unchanged':True,'cutoff':cutoff}))
        return
    sys.path.insert(0,str(bundle/'source/EDM-IR/tools'))
    import run_w1_gpu as w1
    run=SimpleNamespace(bundle=bundle,inputs=root/'inputs',build_dir=root/'build-precision',out=root/'results-precision')
    clock=w1.Deadline(cutoff)
    _,records=w1.verify(bundle,run.inputs)
    assert w1.preflight(run,source,clock)=={k:v for k,v in context.items() if k!='window_identity'}, 'Execution conditions changed'
    extension=root/'session/extension-intent.json'
    if extension.exists():raise ValueError('Extension already attempted; inspect before another resume')
    w1.save(extension,{'guard':guard,'original_intent_sha256':sha(run.out/'intent.json'),
                      'resume_runner_sha256':sha(Path(__file__).resolve()),'cutoff_epoch':cutoff,
                      'started_utc':datetime.now(timezone.utc).isoformat()})
    previous=json.loads((run.out/'status.json').read_text())
    w1.save(run.out/'status-before-extension.json',previous)
    state=dict(state='resumed_running',completed_stages=list(previous['completed_stages']),traces=dict(previous['traces']))
    try:
        for stage in plan['stages']:
            if stage['name'] in state['completed_stages']:
                assert (run.out/stage['name']/'collect-complete.json').exists()
                print('Preserved complete stage:',stage['name'],flush=True)
                continue
            traces=window.run_stage(w1,run,source,records,stage,clock,context)
            state['completed_stages'].append(stage['name']);state['traces'].update(traces)
            w1.save(run.out/'extension-status.json',state)
            print('Completed resumed stage:',stage['name'],traces,flush=True)
        state['state']='timings_complete_diagnostics_require_review'
    except BaseException as error:
        state.update(state='interrupted',error=str(error));raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat()
        w1.save(run.out/'extension-status.json',state)

if __name__=='__main__':main()
