"""Audit all eight devices; trace overlap is temporal coexistence, not hidden bytes."""
from pathlib import Path
import json,sqlite3

HERE=Path(__file__).resolve().parent

def merged(intervals):
    out=[]
    for a,b in sorted(intervals):
        if not out or a>out[-1][1]:out.append([a,b])
        else:out[-1][1]=max(out[-1][1],b)
    return out

def duration(intervals):return sum(b-a for a,b in merged(intervals))

def intersect(a,b):
    a=merged(a);b=merged(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total

def main():
    plan=json.loads((HERE/'bundle/plan.json').read_text());cases=[]
    for name,(tag,family) in plan['p1_trace_cases'].items():
        folder=HERE/'recovered/results/diagnostic'/name
        x=dict(case=name,family=family,diagnostic_only=True)
        db=folder/'timeline.sqlite'
        if not db.exists():x['status']='missing_sqlite';cases.append(x);continue
        c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        rows=lambda sql:[dict(r) for r in c.execute(sql)]
        tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
        if not {'CUPTI_ACTIVITY_KIND_KERNEL','NVTX_EVENTS','StringIds'}<=tables:
            x['status']='missing_trace_tables';cases.append(x);c.close();continue
        kernels=rows('select k.*,s.value kernel from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id')
        nvtx=rows('select * from NVTX_EVENTS where end>start')
        # NVTX text may be inline or interned, depending on Nsight version.
        strings=dict(c.execute('select id,value from StringIds'))
        for r in nvtx:r['resolved_text']=r.get('text') or strings.get(r.get('textId'),'')
        phase=[r for r in nvtx if r['resolved_text'].startswith('edm.phase:runtime.execute')]
        ops=[r for r in nvtx if r['resolved_text'].startswith('edm.op:')]
        expected_ops=3*len(json.loads((HERE/'bundle'/tag/family/'program.json').read_text())['ops'])
        diagnostics=rows('select * from DIAGNOSTIC_EVENT') if 'DIAGNOSTIC_EVENT' in tables else []
        details=rows('select * from ANALYSIS_DETAILS') if 'ANALYSIS_DETAILS' in tables else []
        stderr=(folder/'stderr').read_text() if (folder/'stderr').exists() else ''
        stdout=(folder/'stdout').read_text() if (folder/'stdout').exists() else ''
        per=[]
        for device in range(8):
            ks=[r for r in kernels if r['deviceId']==device]
            comm=[(r['start'],r['end']) for r in ks if 'nccl' in r['kernel'].lower()]
            gemm=[(r['start'],r['end']) for r in ks if 'cutlass' in r['kernel'].lower() or 'gemm' in r['kernel'].lower()]
            configs={}
            for r in ks:
                key=(r['kernel'],r['gridX'],r['gridY'],r['gridZ'],r['blockX'],r['blockY'],r['blockZ'],r['registersPerThread'],r['staticSharedMemory'],r['dynamicSharedMemory'])
                configs[key]=configs.get(key,0)+1
            per.append(dict(device=device,kernel_count=len(ks),nccl_kernels=sum('nccl' in r['kernel'].lower() for r in ks),gemm_kernel_count=len(gemm),nccl_interval_union_us=duration(comm)/1000,gemm_interval_union_us=duration(gemm)/1000,nccl_gemm_simultaneous_us=intersect(comm,gemm)/1000,kernel_configurations=[dict(kernel=k[0],grid=k[1:4],block=k[4:7],registers_per_thread=k[7],static_shared_bytes=k[8],dynamic_shared_bytes=k[9],launches=n) for k,n in configs.items()]))
        complete=len(phase)==3 and len(ops)==expected_ops and all(r['kernel_count']>0 for r in per) and all(k['end']>k['start'] for k in kernels)
        x.update(status='coverage_complete_requires_diagnostic_review' if complete else 'coverage_incomplete',per_device=per,execution_ranges=[r['resolved_text'] for r in phase],operator_ranges=len(ops),expected_operator_ranges=expected_ops,diagnostics=diagnostics,analysis_details=details,stderr=stderr,diagnostic_stdout=stdout)
        cases.append(x);c.close()
    result=dict(cases=cases,all_four_coverage_complete=len(cases)==4 and all(x['status']=='coverage_complete_requires_diagnostic_review' for x in cases),interpretation=['Unprofiled paired blocks supply performance labels.','NCCL kernel intervals can include synchronization; intersection is temporal coexistence only.','Fused peer kernels have no separate NCCL interval: absence cannot imply absence of communication.','Cross-device sums are not wall-clock latency, physical bandwidth, or causal communication hiding.','Register/shared-memory declarations are per-kernel resource requirements, not achieved occupancy.'],manual_diagnostic_review_required=True)
    (HERE/'trace-analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'all_four_coverage_complete':result['all_four_coverage_complete'],'cases':[{k:x[k] for k in ('case','status','operator_ranges','expected_operator_ranges') if k in x} for x in cases]}))

if __name__=='__main__':main()
