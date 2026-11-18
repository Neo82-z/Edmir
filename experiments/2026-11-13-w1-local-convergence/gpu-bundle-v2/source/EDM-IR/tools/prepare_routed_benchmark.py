#!/usr/bin/env python3
"""Deterministic CPU rehearsal. Never launches GPUs or exports measured results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edmir.benchmark.measurement import FIXTURE_STATS, compare_batches, digest, synthetic_batch
from edmir.benchmark.routed_reference import compare_outputs, execute_reference, make_fixture, toy_trace
from edmir.compiler.routed_contract import build_routed_traffic_ledger


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def retained_formal_audit():
    """Check retained evidence against current model/config bytes; NOT a rerun."""
    spec = ROOT/'spec/tla'
    result = []
    for folder in ('results', 'publication_results'):
        summary = json.loads((spec/folder/'summary.json').read_text())
        checks = [file_hash(spec/summary['model']) == summary['model_sha256']]
        for case in summary['cases']:
            checks.append(file_hash(spec/case['config']) == case['config_sha256'])
            raw = spec/folder/case['raw_output']
            checks.append(raw.is_file() and case['expectation_met'])
        result.append({'model':summary['model'], 'model_and_config_hashes_match':all(checks),
                       'evidence_kind':'retained_report_audit_not_fresh_tlc',
                       'original_executed_at_utc':summary['executed_at_utc']})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True, help='must not already exist')
    ap.add_argument('--jar', type=Path, help='optional official tla2tools.jar; enables fresh TLC')
    ap.add_argument('--java', help='working Java executable')
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks, plots, ledgers, traces, timings = [], [], [], [], []
    for p in (2,4,8):
        for pattern in ('balanced','hotset','empty'):
            trace = toy_trace(p,pattern)
            traces.append(trace.to_dict())
            ledgers.append(build_routed_traffic_ledger(trace).to_dict())
            for seed in (0,1,2):
                fixture = make_fixture(trace,seed)
                expected = execute_reference(fixture)
                for q in (1,2,4,7):
                    for g in (1,2):
                        actual = execute_reference(fixture,split_k=q,group_n=g,
                                                   private_order=reversed(range(len(trace.routes))))
                        checks.append({'trace_id':trace.trace_id, 'seed':seed, 'split_k':q,
                                       'group_n':g, **compare_outputs(expected,actual)})
            if pattern == 'empty':
                continue  # Empty correctness cases must not invent kernel timings.
            th = digest(trace.to_dict())
            host = synthetic_batch(p,'fixture-host',trace_hash=th)
            for family, scale in [('fixture-fast',.8),('fixture-near-tie',.997)]:
                candidate = synthetic_batch(p,family,trace_hash=th,scale=scale)
                stats = compare_batches(host,candidate,**{k:FIXTURE_STATS[k] for k in ('seed','confidence','resamples')})
                plots.append({'trace_id':trace.trace_id,'world_size':p,'pattern':pattern,**stats})
                timings.append(candidate)
            timings.append(host)
    write_json(out/'synthetic-traces.json',traces)
    write_json(out/'synthetic-ledgers.json',ledgers)
    write_json(out/'cpu-numerical-checks.json',checks)
    write_json(out/'synthetic-timings.json',timings)
    write_json(out/'synthetic-statistics.json',{'evidence_kind':'synthetic_pipeline_fixture',
               'warning':'ENGINEERED INPUTS; NOT GPU PERFORMANCE; NOT PAPER EVIDENCE',
               'statistics':FIXTURE_STATS,'comparisons':plots})
    campaign = {
        'schema':'edmir.routed-campaign-preparation.v1','status':'draft_needs_assets',
        'protocol':'D7','protocol_hash':file_hash(ROOT/'docs/ROUTED_BENCHMARK_CONTRACT.md'),
        'dtype':'bfloat16','world_sizes':[2,4,8],'same_node_required':True,
        'warmups':10,'blocks':10,'samples_per_block':20,'near_tie_repeat_iterations':2000,
        'trace_hashes':[],'environment_hashes':[],'source_hashes':[],
        'gpu_subsets_by_uuid':None,'numerical_seed_ids':None,'split_salt':None,
        'bootstrap_config':None,'normalization_definitions':None,
        'unresolved':['real trace assets and parent split manifest',
                      'candidate/baseline builds and numerical evidence',
                      'same-node GPU UUID subsets and environment snapshot',
                      'three numerical seeds, split salt, bootstrap and norm definitions',
                      'repeat timing envelope and pilot runtime budget check'],
        'note':'Template only. Do not replace missing values with synthetic measurements.'}
    write_json(out/'campaign.template.json',campaign)
    tests = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','*_test.py'],
                           cwd=ROOT,capture_output=True,text=True)
    (out/'unit-tests.log').write_text(tests.stdout+tests.stderr)
    retained = retained_formal_audit()
    fresh = []
    if args.jar:
        for script, folder in [('run_tlc.py','slot'),('run_publication_tlc.py','publication')]:
            destination = out/'formal'/folder
            command = [sys.executable,str(ROOT/'tools'/script),'--jar',str(args.jar.resolve()),
                       '--out-dir',str(destination)]
            if args.java:
                command += ['--java',args.java]
            run = subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
            (out/f'tlc-{folder}.log').write_text(run.stdout+run.stderr)
            fresh.append({'model_family':folder,'returncode':run.returncode,
                          'passed':run.returncode == 0,
                          'evidence_kind':'fresh_tlc_execution'})
    passed = (tests.returncode == 0 and all(c['passed'] for c in checks)
              and all(r['model_and_config_hashes_match'] for r in retained)
              and all(r['passed'] for r in fresh))
    sources = {str(p.relative_to(ROOT)):file_hash(p) for base in ('edmir','tools','spec/tla','tests')
               for p in sorted((ROOT/base).rglob('*'))
               if p.is_file() and p.suffix in ('.py','.tla','.cfg')}
    write_json(out/'source-manifest.json',sources)
    summary = {'schema':'edmir.offline-rehearsal.v1','offline_checks_passed':passed,
               'cpu_schedule_layout_cases':len(checks),'synthetic_traces':len(traces),
               'synthetic_timing_records':len(timings),'retained_formal_audit':retained,
               'fresh_tlc':fresh,'fresh_tlc_run':bool(fresh),'gpu_used':False,
               'ready_for_paid_experiments':False,'blockers':campaign['unresolved'],
               'warning':'CPU checks do not certify CUDA kernels, hardware memory ordering, or speed.'}
    write_json(out/'summary.json',summary)
    write_json(out/'artifact-manifest.json',{p.name:file_hash(p) for p in sorted(out.glob('*')) if p.is_file()})
    print(json.dumps(summary,indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
