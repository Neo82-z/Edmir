"""Final fixed-design audit and descriptive replication; never fits a predictor."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv, hashlib, json, statistics as st, sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BUNDLE=REPO/'experiments/2026-11-13-w1-local-convergence/gpu-bundle-v2'
sys.path.insert(0,str(BUNDLE/'source/EDM-IR/tools'))
from run_w1_gpu import validate_rows,done_raw_changed,require_successful_pilot
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
    rec=ROOT/'recovered';out=rec/'results-balanced-large-n';manifest=read(rec/'recovery-manifest.json')['files']
    for name,expected in manifest.items():
        p=(rec/name).resolve()
        if not p.is_relative_to(rec.resolve()) or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            raise ValueError('recovery integrity: '+name)
    for name,expected in read(ROOT/'frozen-predictions-manifest.json')['files'].items():
        if sha(ROOT/name)!=expected['sha256']:raise ValueError('precollection prediction changed')
    plan=read(rec/'followup/operator-plan.json');source=read(BUNDLE/'plan.json');inputs=read(BUNDLE/'inputs.json')
    assert sha(rec/'followup/operator-plan.json')==sha(ROOT/'followup/operator-plan.json')
    assert sha(rec/'followup/run_precision_followup.py')==sha(ROOT/'followup/run_precision_followup.py')
    assert sha(BUNDLE/'files.json')==plan['source_manifest_sha256']
    context=read(out/'context.json');receipt=read(rec/'build-precision/receipt.json')
    assert context['package']==receipt['package']==plan['source_manifest_sha256']
    assert context['build']==sha(rec/'build-precision/receipt.json')
    assert all(sha(rec/f'build-precision/w1_{tag}_batch')==h for tag,h in receipt['binaries'].items())
    require_successful_pilot(out,sha(out/'context.json'),source,context)
    missing=[];failed=set();failures=[];checks=0;rows=[];blocks=[]
    for i,job in enumerate(plan['jobs']):
        p=out/f'collect-{i:03d}.json'
        if not p.exists():missing.append(i);continue
        r=read(p);assert r['job']==job and r['context']==sha(out/'context.json')
        assert r['input']==inputs['large_n'][str(job['seed'])]
        expected=[f for f in job['order'] if job['phase']=='check' or f not in failed]
        assert r['order']==expected
        assert r['excluded_by_numerical_check']==[f for f in job['order'] if f not in expected]
        assert not done_raw_changed(p,out)
        validate_rows(r['rows'],job,r['order'],source['shapes']['large_n'],context['runtime'])
        medians={}
        for position,x in enumerate(r['rows']):
            if not x['correctness_passed']:
                failed.add(x['family']);failures.append(dict(job_index=i,seed=job['seed'],**x));continue
            if job['phase']=='check':checks+=1;continue
            raw=x['samples_ms'];us=1000*st.median(raw);medians[x['family']]=us
            rows.append(dict(job_index=i,seed=job['seed'],position=position,predecessor=r['order'][position-1] if position else 'START',family=x['family'],median_us=us,first_half_us=1000*st.median(raw[:50]),last_half_us=1000*st.median(raw[50:]),samples=len(raw),process_seconds=r['process_seconds'],implementation_id=x['implementation_id'],gemm_algorithm_id=x['gemm_algorithm_id']))
        if medians:blocks.append(dict(job_index=i,seed=job['seed'],winner=min(medians,key=lambda f:(medians[f],f)),medians_us=medians))
    complete=not missing and (out/'collect-complete.json').exists()
    if complete:assert read(out/'collect-complete.json')['attempted_groups']==len(plan['jobs'])
    audit=dict(verified_files=len(manifest),planned_groups=len(plan['jobs']),completed_groups=len(plan['jobs'])-len(missing),missing=missing,correctness_passes=checks,numerical_failures=failures,timing_rows=len(rows),raw_samples=sum(x['samples'] for x in rows),complete_protocol=complete,frozen_shapes_read=False,fitting=False)
    write(ROOT/'recovery-audit.json',audit)
    if rows:
        with (ROOT/'measurements.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if not complete:raise RuntimeError('Incomplete protocol retained; no complete-result claim')
    families=sorted({r['family'] for r in rows});summary={}
    def grouped(rs,key):
        g=defaultdict(list)
        for x in rs:g[str(x[key])].append(x['median_us'])
        return {k:dict(blocks=len(v),median_us=st.median(v),min_us=min(v),max_us=max(v)) for k,v in g.items()}
    for f in families:
        rs=[x for x in rows if x['family']==f]
        summary[f]=dict(median_us=st.median(x['median_us'] for x in rs),by_seed=grouped(rs,'seed'),by_position=grouped(rs,'position'),by_predecessor=grouped(rs,'predecessor'))
    best=min(summary,key=lambda f:(summary[f]['median_us'],f));pred=read(ROOT/'frozen-predictions/preregistered.json')['predictions'];evaluations={}
    for variant,p in pred.items():
        chosen=p['chosen_among_fixed_six']
        if chosen not in summary:evaluations[variant]=dict(chosen=chosen,status='failed_new_admission');continue
        regrets=[100*(b['medians_us'][chosen]/min(b['medians_us'].values())-1) for b in blocks]
        errors=[]
        for a,b in combinations(families,2):
            ds=[x['medians_us'][a]-x['medians_us'][b] for x in blocks]
            measured=summary[a]['median_us']-summary[b]['median_us'];estimated=p['scores_us'][a]-p['scores_us'][b]
            errors.append(dict(a=a,b=b,predicted_delta_us=estimated,aggregate_delta_us=measured,aggregate_direction_wrong=measured*estimated<0,blocks_direction_wrong=sum(d*estimated<0 for d in ds),a_faster_blocks=sum(d<0 for d in ds),b_faster_blocks=sum(d>0 for d in ds),paired_median_delta_us=st.median(ds)))
        evaluations[variant]=dict(chosen=chosen,aggregate_regret_pct=100*(summary[chosen]['median_us']/summary[best]['median_us']-1),mean_block_regret_pct=st.mean(regrets),worst_block_regret_pct=max(regrets),block_regrets_pct=regrets,pairwise=errors,strict_contract_status='not_admitted_new_context; no evidence or calibration rebinding',interpretation=p['interpretation'])
    oldctx=read(REPO/'experiments/2026-11-16-shadeform-precision/recovered/results-precision/large_n/context.json')
    changed=[k for k in set(oldctx)|set(context) if oldctx.get(k)!=context.get(k)]
    result=dict(audit=audit,summary=summary,aggregate_best=best,block_winners=dict(Counter(x['winner'] for x in blocks)),blocks=blocks,frozen_rank_evaluation=evaluations,context_changed_fields=changed,limitations=['Development shape replication only; no frozen holdout evaluation','Block medians are units; serial samples are not independent trials','Candidate position/predecessor balanced only if all candidates pass','Order and process drift not fully causal; no post-hoc fit or filter','Kernel timing is not end-to-end serving latency'],predictor_iteration_closed=True)
    write(ROOT/'analysis.json',result)
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k!='numerical_failures'},best=best,winners=result['block_winners'],regrets={k:{q:v[q] for q in ['chosen','aggregate_regret_pct','mean_block_regret_pct','worst_block_regret_pct']} for k,v in evaluations.items() if 'aggregate_regret_pct'in v}),ensure_ascii=False))
if __name__=='__main__':main()
