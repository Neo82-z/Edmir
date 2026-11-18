#!/usr/bin/env python3
"""Bounded acquisition using the unchanged, sealed W1 binaries and validators."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_plan(plan, source, bundle):
    if plan['source_files_sha256'] != sha(bundle / 'files.json'):
        raise ValueError('sealed source manifest changed')
    if plan['source_plan_sha256'] != sha(bundle / 'plan.json'):
        raise ValueError('sealed source plan changed')
    if plan['shapes'] != source['shapes'] or plan['seeds'] != source['seeds']:
        raise ValueError('development shapes or seeds changed')
    if [s['name'] for s in plan['stages']] != ['grouped', 'm512', 'large_n', 'm256']:
        raise ValueError('stage order changed')
    seen = []
    traces = {'grouped': ['grouped-g1', 'grouped-g2'],
              'm512': ['m512-serial', 'm512-pipeline'], 'large_n': [], 'm256': []}
    for stage in plan['stages']:
        indices = stage['original_job_indices']
        if stage['jobs'] != [source['jobs'][i] for i in indices]:
            raise ValueError('job differs from original protocol')
        if any(j['shape'] != stage['name'] for j in stage['jobs']):
            raise ValueError('wrong shape in stage')
        if stage['traces'] != traces[stage['name']]:
            raise ValueError('diagnostic cases changed')
        seen.extend(indices)
    if sorted(seen) != list(range(len(source['jobs']))):
        raise ValueError('missing or duplicated original job')


def deadline_from_guard(guard, now):
    if guard.get('platform_verified') is not True or not guard.get('instance_id'):
        raise ValueError('platform time/spend guards must be visibly verified first')
    active = guard['active_epoch']
    hourly = guard['hourly_usd']
    end = guard['auto_delete_epoch']
    cap = guard['spend_cap_usd']
    if not (0 < hourly <= 10.80 and 0 < cap <= 10.80):
        raise ValueError('additional-hour price or spend cap exceeds authorization')
    if not (active <= now < end <= active + 3300):
        raise ValueError('expired or excessive platform runtime')
    if (end - active) * hourly / 3600 > cap:
        raise ValueError('time rule exceeds spend cap')
    cutoff = min(active + 3000, end - 300)
    if cutoff <= now:
        raise TimeoutError('collection cutoff reached; recover and stop')
    return cutoff


def execute_stage(w1, args, source, records, stage, clock, context):
    out = args.out / stage['name']
    runargs = SimpleNamespace(out=out, inputs=args.inputs, build_dir=args.build_dir,
                              action='pilot', trace_case=None)
    derived = dict(source, jobs=stage['jobs'])
    # A new result directory always acquires its own pilot. No old context is copied.
    w1.collect(runargs, derived, records, clock, context)
    runargs.action = 'collect'
    w1.collect(runargs, derived, records, clock, context)
    traces = {}
    for case in stage['traces']:
        dest = out / 'diagnostic' / case
        if dest.exists():
            receipt = dest / 'receipt.json'
            report = dest / 'timeline.nsys-rep'
            valid = receipt.is_file() and report.is_file()
            if valid:
                valid = json.loads(receipt.read_text())['report_sha256'] == sha(report)
            traces[case] = 'captured' if valid else 'failed_preserved_no_overwrite'
            continue
        if clock.end - time.time() < 125:
            traces[case] = 'not_started_time_budget'
            continue
        runargs.trace_case = case
        try:
            w1.trace(runargs, derived, clock, context)
            traces[case] = 'captured'
        except Exception as error:
            traces[case] = 'failed'
            save(out / (case + '-failure.json'), {'case': case, 'error': str(error),
                 'kind': 'diagnostic_failure_not_a_timing_label'})
            # Do not continue if the profiler left GPU processes or corrupted state.
            if w1.preflight(args, source, clock) != {k: v for k, v in context.items()
                                                    if k != 'extension_identity'}:
                raise ValueError('environment changed after diagnostic failure')
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--guard', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for name in ('bundle', 'inputs', 'build_dir', 'out', 'guard'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    plan = json.loads((ROOT / 'plan.json').read_text())
    source = json.loads((args.bundle / 'plan.json').read_text())
    validate_plan(plan, source, args.bundle)
    if args.dry_run:
        print(json.dumps({'status': 'local_plan_validated_gpu_not_run',
                          'jobs': sum(len(s['jobs']) for s in plan['stages']),
                          'traces': [c for s in plan['stages'] for c in s['traces']],
                          'plan_sha256': sha(ROOT / 'plan.json')}, indent=2))
        return
    if any(getattr(args, name) is None for name in ('inputs', 'build_dir', 'out', 'guard')):
        parser.error('execution requires inputs, build-dir, out and verified platform guard')
    guard = json.loads(args.guard.read_text())
    cutoff = deadline_from_guard(guard, time.time())
    sys.path.insert(0, str(args.bundle / 'source/EDM-IR/tools'))
    import run_w1_gpu as w1
    clock = w1.Deadline(cutoff)
    _, records = w1.verify(args.bundle, args.inputs)
    context = w1.preflight(args, source, clock)
    context['extension_identity'] = {'plan': sha(ROOT / 'plan.json'),
                                     'runner': sha(Path(__file__)),
                                     'instance_id': guard['instance_id']}
    args.out.mkdir(parents=True, exist_ok=True)
    intent_path = args.out / 'intent.json'
    intent = {'plan': plan, 'context': context, 'guard': guard, 'cutoff_epoch': cutoff}
    if intent_path.exists() and json.loads(intent_path.read_text()) != intent:
        raise ValueError('resume requires identical instance, plan, binary, libraries and guard')
    save(intent_path, intent)
    status = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'completed_stages': [], 'traces': {}, 'gpu_collection_complete': False}
    try:
        for stage in plan['stages']:
            # File verification and preflight never manufacture measurements.
            if clock.end - time.time() < 30:
                raise TimeoutError('not starting another stage at collection cutoff')
            traces = execute_stage(w1, args, source, records, stage, clock, context)
            status['completed_stages'].append(stage['name'])
            status['traces'].update(traces)
            save(args.out / 'status.json', status)
            print('Completed:', stage['name'], traces, flush=True)
        status['gpu_collection_complete'] = True
        status['state'] = ('complete' if all(v == 'captured' for v in status['traces'].values())
                           and len(status['traces']) == 4 else 'timings_complete_diagnostics_incomplete')
    except BaseException as error:
        status['state'] = 'interrupted'
        status['error'] = str(error)
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(args.out / 'status.json', status)


if __name__ == '__main__':
    main()
