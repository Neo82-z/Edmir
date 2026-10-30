"""Bounded active-route / archival timing audit. No fitting or policy edit."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'EDM-IR'),str(ROOT/'experiments/2026-10-28-dependency-structure-review')]
from edmir.compiler.prediction_explanation import explain
import review as previous
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SHORT = ((4,64,512,128),(4,64,512,256),(4,64,512,512),(4,64,512,1024),(4,128,512,512))


def historical_rows(rows):
    allowed = {(tuple(r['shape']),r['candidate_id']):r for r in rows}
    results = []
    for line in previous.diagnosis.ARCHIVE.read_text().splitlines():
        archived = json.loads(line); m = archived['measurement']
        shape = (m['world_size'],m['rows_per_peer'],m['k'],m['n']); family = m['family']
        if shape not in SHORT or family not in ('direct','splitk2') or archived['historical_role']!='development': continue
        key = (shape,family)
        if key not in allowed or archived['environment_id']!='a100-sxm4-80gb-p4-2026-08-29': continue
        source = ROOT/archived['source']['path']; source.resolve().relative_to(ROOT)
        if sha(source)!=archived['source']['sha256']: raise AssertionError('source file differs')
        p,m,k,n = shape
        logs = list((source.parent/'logs').glob(f'12_bench_*_m{m}_k{k}_n{n}_{family}.txt'))
        if len(logs)!=1 or logs[0].stat().st_size>100000: raise ValueError('bounded unique raw log required')
        text = logs[0].read_text(); fields = dict(re.findall(r'(\w+)=([^\s]+)',text))
        timing = archived['reported_timing']
        for name in ('min_ms','median_ms','p95_ms','max_ms','mean_ms'):
            if float(fields[name])!=timing[name]: raise AssertionError('raw timing label differs')
        if not math.isclose(1000*timing['median_ms'],allowed[key]['latency_us'],abs_tol=1e-9): raise AssertionError('training label differs')
        counts = {name:int(fields[name]) for name in ('ops','op_dispatch','event_record','event_wait','buffer_reuse',
            'remote_input_bytes','remote_a_replay','estimated_remote_a_load_bytes','estimated_ctas_per_op','warmup','iterations')}
        results.append({'shape':shape,'candidate':family,'source':archived['source'],
            'raw_log':str(logs[0].relative_to(ROOT)),'raw_log_sha256':sha(logs[0]),
            'counts':counts,'reported_timing':timing,'paired_raw_samples_available':False,
            'host_phase_timestamps_available':False,'device_stage_timestamps_available':False})
    if len(results)!=10: raise AssertionError('missing short-K source rows')
    paired=[]
    for shape in SHORT:
        rr={r['candidate']:r for r in results if tuple(r['shape'])==shape}
        a,b=rr['direct'],rr['splitk2']
        equal = {k:a['counts'][k]==b['counts'][k] for k in a['counts'] if k!='estimated_ctas_per_op'}
        if not all(equal.values()): raise AssertionError('unexpected outer execution count difference')
        paired.append({'shape':shape,'same_outer_counts':equal,
            'median_gap_us':1000*(b['reported_timing']['median_ms']-a['reported_timing']['median_ms']),
            'mean_gap_us':1000*(b['reported_timing']['mean_ms']-a['reported_timing']['mean_ms'])})
    return {'rows':results,'pairs':paired,'count_equality_is_time_equality':False,
            'rounded_pair_gap_max_error_us':0.1,
            'rounding_alone_can_flip_observed_short_k_signs':False}


def main(out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    design=json.loads((HERE/'design-before.json').read_text()); protected={**design['source_hashes'],**design['protected_hashes']}
    for path,h in protected.items():
        if sha(ROOT/path)!=h: raise AssertionError('input changed: '+path)
    source_paths=[Path(__file__),ROOT/'EDM-IR/edmir/compiler/prediction_explanation.py',
                  ROOT/'EDM-IR/runtime/tools/diagnose_selector_phases.cpp',ROOT/'EDM-IR/runtime/include/edmir/runtime/host_phase_timing.hpp']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    (out/'execution-before.json').write_text(json.dumps({'sources':hashes,'design_sha256':sha(HERE/'design-before.json'),'training_allowed':False},indent=2)+'\n')
    old_source=subprocess.run(['git','show','f584d7d:EDM-IR/runtime/tools/benchmark_generated.cpp'],cwd=ROOT,check=True,capture_output=True,text=True,timeout=5).stdout
    (out/'archived-benchmark-source.cpp').write_text(old_source)
    if 'std::vector<unsigned char> zeros' not in old_source or 'fraction * static_cast<double>(sorted.size() - 1)' not in old_source:
        raise AssertionError('reviewed source semantics changed')
    with previous.no_fitting() as calls:
        rows,folds,models=previous.diagnosis.load_inputs()
        saved=json.loads((previous.diagnosis.SAVED/'results.json').read_text())['variants']['intercept']['decisions']
        traces=[];route_counts=Counter();loop=[]
        for repetition in range(4):
            begin=time.perf_counter()
            for shape,fold in folds.items():
                rr=[r for r in rows if tuple(r['shape'])==shape];trace=explain(models[fold],rr)
                d=next(x for x in saved if tuple(x['shape'])==shape)
                if trace['prediction']['predicted_log_us']!=d['predicted_log_us'] or trace['prediction']['provisional_choice']!=d['provisional_choice']:
                    raise AssertionError('score changed')
                if repetition==0:
                    traces.append({'shape':shape,**trace});route_counts[trace['explanations']['host']['active_path']]+=1
            loop.append((time.perf_counter()-begin)*1000)
            if time.monotonic()-started>60: raise TimeoutError('audit deadline')
        stress=json.loads((previous.diagnosis.SAVED.parent/'audit/all_M512-intercept.json').read_text())['model']
        trajectory=[]
        for M in (64,128,256,512):
            rr=[r for r in rows if r['shape']==[4,M,4096,512]]
            traced=explain(stress,rr);h=traced['explanations']['host']
            trajectory.append({'M':M,'trace':h,'observed_host_us':next(r['latency_us'] for r in rr if r['is_host']),
                               'raw_prior_us':math.fsum(h['base_terms_us'].values())})
        a,b=trajectory[-2:]
        anchor_error=a['raw_prior_us']-a['observed_host_us']
        growth_error=(b['raw_prior_us']-a['raw_prior_us'])-(b['observed_host_us']-a['observed_host_us'])
        if not math.isclose(anchor_error+growth_error,b['raw_prior_us']-b['observed_host_us'],abs_tol=1e-9): raise AssertionError('growth identity failed')
        rr=[r for r in rows if r['shape']==[4,512,4096,512]];t=explain(stress,rr)
        best_fused=min((math.exp(v),n) for n,v in t['prediction']['predicted_log_us'].items() if n!='host')
        growth={'anchor_M256_prior_error_us':anchor_error,'predicted_increment_us':b['raw_prior_us']-a['raw_prior_us'],
                'observed_increment_us':b['observed_host_us']-a['observed_host_us'],'increment_error_us':growth_error,
                'M512_prior_error_us':anchor_error+growth_error,
                'inactive_interval_host_us':b['trace']['inactive_host_interval_us'],
                'best_predicted_fused':best_fused[1],'best_predicted_fused_us':best_fused[0],
                'inactive_host_minus_best_fused_us':b['trace']['inactive_host_interval_us']-best_fused[0],
                'inactive_path_would_fix_choice':False,'inactive_path_is_not_adopted':True}
        evidence=historical_rows(rows)
        previous_evidence=json.loads((ROOT/'experiments/2026-10-28-conditional-curves/existing-evidence.json').read_text())
        result={'host_route_counts':dict(route_counts),'full_traces':traces,'host_M_trajectory':trajectory,
                'M512_increment_account':growth,'short_K_timing_evidence':evidence,
                'existing_profile_coverage':{'M512_host_present':previous_evidence['component_inventory']['M512_host_trace_in_collected_trace_set'],
                    'short_K_profiles_present':any(v['shape'][2]==512 for v in previous_evidence['traces'].values()),
                    'profile_rows_reused_as_labels':False},
                'timing_semantics':{'archived_reported_revision':'f584d7d','archived_source_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
                    'archived_inputs':'all-zero external inputs and weights in reported source',
                    'archived_median':'sorted[floor(0.5*(iterations-1))]; 300 samples -> 1-based sample 150',
                    'current_median':'average of sorted[149] and sorted[150] for 300 samples',
                    'archived_output_rounding':'4 decimal places in milliseconds',
                    'timing_scope_both':'host execute plus synchronize',
                    'current_inputs':'fixed random FP16 inputs with CPU FP32 references',
                    'archived_binary_identity_verified':False,'different_identities_may_be_pooled':False,
                    'raw_samples_sufficient_to_recompute_same_quantile':False},
                'training_calls':len(calls),'parameter_updates':0,'old_scores_bitwise_identical':True,
                'old_wrong_choices':sum(x['provisional_regret']>0 for x in saved),'frozen_six_parsed_or_scored':False,
                'GPU_started':False,'compiler_admission':False,'explanation_traversal_ms':{'median':statistics.median(loop),'max':max(loop)}}
    for p,h in {**protected,**hashes}.items():
        if sha(ROOT/p)!=h: raise AssertionError('source changed: '+p)
    result.update(seconds=time.monotonic()-started,protected_unchanged=True)
    if result['seconds']>60: raise TimeoutError('audit deadline')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('host_route_counts','M512_increment_account','existing_profile_coverage','old_wrong_choices','training_calls','seconds','explanation_traversal_ms')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
