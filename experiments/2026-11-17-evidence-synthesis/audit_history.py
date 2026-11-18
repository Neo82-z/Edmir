#!/usr/bin/env python3
"""Recheck historical observations without merging execution identities or fitting."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def audit(root,manifest_path):
    manifest=read(manifest_path); files=manifest.get('files',manifest)
    for name,value in files.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve())
        expected=value['sha256'] if isinstance(value,dict) else value
        assert sha(p)==expected,name
        if isinstance(value,dict):assert p.stat().st_size==value['bytes']
    stages={}
    for name in files:
        if not name.endswith('/measurements.jsonl'):continue
        rows=[json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
        valid=[r for r in rows if r.get('correctness_passed') and r.get('samples_ms')]
        for r in valid:
            assert all(math.isfinite(t) and t>0 for t in r['samples_ms'])
            assert abs(st.median(r['samples_ms'])-r['median_ms'])<1e-8
        stages[name]=dict(records=len(rows),valid_timing_records=len(valid),
            samples=sum(len(r['samples_ms']) for r in valid),
            shapes=sorted({tuple(r['shape']) for r in rows}),
            failed_records=sum(r.get('correctness_passed') is False for r in rows),
            file_sha256=sha(root/name))
    return dict(files_verified=len(files),manifest_sha256=sha(manifest_path),stages=stages)

def main():
    results={}
    for date,folder,mf,manifest_in_root in [
        ('2026-10-21','2026-10-21-runpod-mechanism-collection/remote-run-01/server-backup','result-file-manifest.json',True),
        ('2026-10-25','2026-10-25-runpod-hour-01/recovered','recovery-file-manifest.json',True),
        ('2026-10-28','2026-10-28-p8-session/recovered','recovery-manifest.json',False)]:
        root=ROOT/'experiments'/folder
        results[date]=audit(root,(root if manifest_in_root else root.parent)/mf)
    ledger=ROOT/'experiments/2026-10-20-offline-predictor/data/selector_measurements.jsonl'
    observations=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
    sources={};by_env=defaultdict(list); repeat_sources={}; repeats=defaultdict(int)
    for r in observations:
        src=r['source'];p=ROOT/src['path']
        if src['path'] not in sources:
            assert sha(p)==src['sha256'];sources[src['path']]=list(csv.DictReader(p.open()))
        row=sources[src['path']][src['line']-2]
        assert abs(float(row['median_ms'])-r['reported_timing']['median_ms'])<1e-8
        assert row['family']==r['measurement']['family']
        by_env[r['environment_id']].append(r)
        for key in ('scoring_override','repeat_observation'):
            item=r[key]
            if not item:continue
            source=item['source'];path=ROOT/source['path']
            if source['path'] not in repeat_sources:
                assert sha(path)==source['sha256'];repeat_sources[source['path']]=list(csv.DictReader(path.open()))
            value=repeat_sources[source['path']][source['line']-2]
            assert abs(float(value['median_ms'])-item['median_ms'])<1e-8
            assert value['family']==r['measurement']['family']
            repeats[key]+=1
    results['archived_august']=dict(ledger_sha256=sha(ledger),source_csvs_verified=len(sources),candidate_rows=len(observations),
        environments={env:dict(rows=len(rs),shapes=len({(r['measurement']['world_size'],r['measurement']['rows_per_peer'],r['measurement']['k'],r['measurement']['n']) for r in rs})) for env,rs in by_env.items()},
        repeat_or_override_records_verified=dict(repeats),repeat_source_csvs_verified=len(repeat_sources),
        interpretation='archived CSV summaries, not raw per-iteration timings; no rerun of training; repeats/overrides not new base rows')
    component_path=ledger.with_name('component_measurements.jsonl')
    components=[json.loads(s) for s in component_path.read_text().splitlines() if s.strip()]
    component_sources={}; matched=defaultdict(dict)
    for r in components:
        src=r['source'];p=ROOT/src['path']
        if src['path'] not in component_sources:
            assert sha(p)==src['sha256'];component_sources[src['path']]=p.read_text().splitlines()
        line=component_sources[src['path']][src['line']-1]
        if r['scope']=='standalone_gemm':
            raw=json.loads(line);latency=raw['median_us']/1000
            assert all(raw[k]==r[k] for k in ('m','n','k','device','dtype'))
            matched[(r['device'],r['m'],r['k'],r['n'])][r['provider']]=r['latency_ms']
        else:
            cols=line.split();latency=float(cols[5 if r['mode']=='out_of_place' else 9])/1000
            assert int(cols[0])==r['size_bytes_per_rank_including_self']
        assert abs(latency-r['latency_ms'])<1e-10
        assert r['eligible_for_fused_training'] is False
    ratios=[r['cutlass']/r['cublas'] for r in matched.values()]
    results['historical_components']=dict(rows=len(components),source_files_verified=len(component_sources),
        gemm_rows=sum(r['scope']=='standalone_gemm' for r in components),collective_rows=sum(r['scope']=='standalone_collective' for r in components),
        paired_gemm_configurations=len(ratios),cublas_faster=sum(v>1 for v in ratios),cutlass_faster=sum(v<1 for v in ratios),
        cutlass_over_cublas_ratio_median=st.median(ratios),ratio_min=min(ratios),ratio_max=max(ratios),
        ledger_sha256=sha(component_path),
        interpretation='standalone FP16 GEMM CUDA-event times and standalone FP32 NCCL-tests; cannot be added or maxed into full-program fused labels')
    paper=ROOT.parent/'llm-serving-mlsys-latex'
    pilot=read(paper/'latex/data/2026-10-28-shape-evaluation/historical-pilot/results.json')
    results['historical_training_comparison']={name:{k:v[k] for k in ('mean_regret','max_regret','wrong_host_to_fused','wrong_fused_to_host','host_winning_shapes','all_order_invariant')} for name,v in pilot['results'].items()}
    muon=read(ROOT/'experiments/2026-10-28-muon-model-audit/results.json')
    results['historical_neural_fit_diagnostics']={name:[{k:v[k] for k in ('fold','heldout_shape','train_rows','train_shapes','affine_parameter_count','train_design_rank','nullity','train_fit','heldout_fit')} for v in vs] for name,vs in muon['p8_heldout_audits'].items()}
    (HERE/'history-audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({date:{k:v for k,v in result.items() if k in ('files_verified','candidate_rows','source_csvs_verified','environments','stages')} for date,result in results.items() if isinstance(result,dict)},ensure_ascii=False))
if __name__=='__main__':main()
