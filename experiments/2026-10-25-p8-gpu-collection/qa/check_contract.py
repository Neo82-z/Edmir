import copy,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];b=root/'bundle'
sys.path[:0]=[str(b/'source/EDM-IR'),str(b/'source/EDM-IR/tools')]
import prepare_p8_gpu as prep
import run_p8_gpu as run
p=prep.verify_bundle(b,require_inputs=False)
assert len(p['jobs'])==351 and len(p['programs'])==101
assert {tuple(w['shape'])[0] for w in p['workloads']}=={8}
for stage in prep.STAGES:
    groups=[{j['candidate_id'] for j in p['jobs'] if j['stage']==stage and j['repetition']==i} for i in range(3)]
    assert groups[0]==groups[1]==groups[2]
for j in p['jobs']:
    q,g=(1,1) if j['family'].startswith('host') else run.parse_schedule_family(j['family'])
    alg='sm80_64x128x32'+(f'_splitk{q}' if q>1 else '')
    if j['family']=='host_merged_cublaslt':alg=''.join(f'rank{r}:0=1;workspace=0;budget=33554432|' for r in range(8))
    row={'family':j['family'],'shape':j['shape'],'metadata_mode':j['metadata_mode'],'implementation_id':j['implementation_id'],
        'dependency_policy':j['dependency_policy'],'input_readiness':'completed_upload_v1','gemm_algorithm_id':alg,
        'correctness_passed':True,'nonfinite_outputs':0,'checked_elements':64*j['shape'][1]*j['shape'][3],
        'max_abs_error':0.,'max_rel_error':0.,'atol':.02,'rtol':.02,'input_seed':20261020,
        'correctness_reference':'CPU_FP32_on_identical_FP16_inputs','warmup':20,'iterations':100,
        'samples_ms':[1.]*100,'median_ms':1.,'mean_ms':1.,'total_ms':100.,'min_ms':1.,'max_ms':1.,
        'timing_scope':p['timing_scope'],'measurement_policy':p['measurement_policy'],'runtime':{'test':'bookkeeping only'}}
    info={'input_sha256':'0'*64,'reference_sha256':'1'*64}
    run.check_row(row,j,info,p)
    for field,value in [('dependency_policy','wrong'),('input_readiness','pending'),('checked_elements',16*j['shape'][1]*j['shape'][3])]:
        try:run.check_row({**row,field:value},j,info,p)
        except ValueError:pass
        else:raise AssertionError('invalid execution identity accepted')
print(json.dumps({'jobs':351,'programs':101,'row_checks':351,'rejected_mutations':1053,'status':'passed'}))
