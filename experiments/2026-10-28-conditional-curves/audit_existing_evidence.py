"""Read-only provenance/trace/curvature review; never fit on profiled times."""
from pathlib import Path
import hashlib
import json
import sqlite3
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.conditional_curve_selection import convex_chord_diagnostic


def main():
    sources={}
    def read(path):
        p=ROOT/path;sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    data=read('experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json')['p4']['rows']
    work='experiments/2026-10-21-runpod-mechanism-collection/remote-run-01/server-backup/work/'
    previous=read(work+'trace-analysis.json')
    capture=read('experiments/2026-10-24-fused-dependency-audit/results.json')['captures']['prepared-identity-full']
    order=capture['op_order'];anc={int(k):v for k,v in capture['kernel_ancestors'].items()}
    traces={};start=time.monotonic()
    for label in ('m512q2','m512q4','largenq2'):
        db=ROOT/(work+f'trace-{label}/timeline.sqlite')
        digest=hashlib.sha256(db.read_bytes()).hexdigest();sources[str(db.relative_to(ROOT))]=digest
        if digest!=previous['traces']['trace-'+label]['sqlite_sha256']:raise ValueError('trace hash mismatch')
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:
            conn.execute('PRAGMA query_only=ON')
            conn.set_progress_handler(lambda:int(time.monotonic()-start>15),10000)
            rows=conn.execute('SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE ? ORDER BY k.start LIMIT 10001',('%lsa_cutlass_gemm<%',)).fetchall()
        bydev={d:[r for r in rows if r[2]==d] for d in range(4)}
        if len(rows)!=1936 or any(len(v)!=484 for v in bydev.values()):raise ValueError('unexpected trace iteration structure')
        local=[];remote=[];spans=[];chain_spans=[];fractions=[];remainders=[];violations=0
        mapping={d:[i for i,_,dest in order if dest==d] for d in range(4)}
        # Existing verified capture: 1 correctness + 20 warmups + 100 timed
        # iterations, each four GEMMs per device; no cross-run subtraction.
        for iteration in range(100):
            times={i:bydev[d][84+4*iteration+j][:2] for d in range(4) for j,i in enumerate(mapping[d])}
            finish={}
            for i,s,d in order:
                duration=(times[i][1]-times[i][0])/1000
                (local if s==d else remote).append(duration)
                violations+=sum(times[a][1]>times[i][0] for a in anc[i])
                finish[i]=duration+max((finish[a] for a in anc[i]),default=0.)
            span=(max(t[1] for t in times.values())-min(t[0] for t in times.values()))/1000
            chain=max(finish.values());spans.append(span);chain_spans.append(chain)
            fractions.append(chain/span);remainders.append(span-chain)
        if violations:raise ValueError('observed execution violates graph')
        traces[label]={'shape':previous['traces']['trace-'+label]['job']['shape'],
            'metadata_mode':'prepared','dependency_policy':'resource_queues_v1','profiler':True,
            'iterations':100,'local_kernel_median_us':statistics.median(local),
            'peer_kernel_median_us':statistics.median(remote),
            'gpu_envelope_median_us':statistics.median(spans),
            'observed_duration_weighted_chain_median_us':statistics.median(chain_spans),
            'weighted_chain_over_envelope_median':statistics.median(fractions),
            'envelope_minus_weighted_chain_median_us':statistics.median(remainders),
            'precedence_checks':100*sum(map(len,anc.values())), 'violations':violations,
            'a100_archival_scorer_training_eligible':False,
            'not_identified':['standalone compute time','standalone peer transfer time','semaphore stalls','CPU-only gaps','resource contention inflation']}
    p=ROOT/'experiments/2026-10-20-offline-predictor/data/component_measurements.jsonl'
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    components=[json.loads(line) for line in p.read_text().splitlines()]
    inventory={scope:{'count':sum(r['scope']==scope for r in components),
                      'environments':sorted({r['environment_id'] for r in components if r['scope']==scope})}
               for scope in sorted({r['scope'] for r in components})}
    inventory['M512_same_shape_standalone_gemm_count']=sum(r['scope']=='standalone_gemm' and
                   (r.get('m'),r.get('k'),r.get('n'))==(512,4096,512) for r in components)
    inventory['M512_host_trace_in_collected_trace_set']=any(v['job']['shape']==[4,512,4096,512] and
                    v['job']['family']=='host' for v in previous['traces'].values())
    points=sorted((r['shape'][2],r['latency_us']) for r in data if r['is_host'] and r['shape'][1]==512 and r['shape'][3]==512)
    output={'sources':sources,'curvature':{'points_K_us':points,**convex_chord_diagnostic(points),
             'proof_scope':'fixed M,N original NNLS host basis is a+bK+cK^2 with c>=0; not a restriction on nonlinear residual network'},
            'traces':traces,'component_inventory':inventory,
            'cross_context_training':False,'missing_matched_components_prevent_causal_decomposition':True,
            'frozen_six_used':False,'new_gpu_cost':0,'seconds':time.monotonic()-start}
    (HERE/'existing-evidence.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'curvature':output['curvature'],'components':inventory,'trace_summaries':traces},indent=2))


if __name__=='__main__':main()
