from pathlib import Path
import json,sys,sqlite3,statistics,hashlib,shutil,tempfile,subprocess,os
from replay import read_capture,replay,paths,closure

here=Path(__file__).resolve().parent;root=here.parents[1]
sys.path.insert(0,str(root/'EDM-IR'))
from edmir.compiler.model import A2AGemmProblem
from edmir.compiler.schedule import build_fused_program
from edmir.compiler.dependence import derive_dependencies
from edmir.compiler import execution_ledger as ledger_module
import importlib
ledger_module=importlib.import_module('edmir.compiler.execution_ledger')
from edmir.compiler.host_baseline import HOST_BASELINES

def main():
    p=build_fused_program(A2AGemmProblem(4,512,4096,512),chunk_rows=512)
    dep=derive_dependencies(p);dep.require_valid();ids={o.id:i for i,o in enumerate(p.ops)}
    edges=[(ids[e.src],ids[e.dst]) for e in dep.dependencies]
    captures={};closures={}
    for f in sorted(here.glob('*-*-*.txt')):
        order,sections=read_capture(f)
        if not order:continue
        a,counts=replay(sections['ITER 0'],order);b,counts2=replay(sections['ITER 1'],order)
        assert a==b and counts==counts2
        selected=edges if f.stem.endswith('full') else [(i,j) for i,j in zip(range(12),range(4,16))]
        # Compute-only order is verified against actual source/destination indices.
        if not f.stem.endswith('full'):
            selected=[];last={}
            for i,_,d in order:
                if d in last:selected.append((last[d],i))
                last[d]=i
        ir=closure(order,selected)
        assert all(ir[k]<=a[k] for k in a),'runtime dropped an IR precedence'
        extra=sorted((i,j) for j in a for i in a[j]-ir[j])
        captures[f.stem]={'calls':counts,'ir_chain':paths(order,ir),
            'runtime_chain':paths(order,a),'extra_transitive_precedences':extra,
            'kernel_ancestors':{str(k):sorted(v) for k,v in a.items()},
            'op_order':order,'two_iterations_identical':True,
            'trace_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        closures[f.stem]=a
    for mapping in ('identity','reverse'):
        _,l=read_capture(here/f'legacy-{mapping}-full.txt');_,r=read_capture(here/f'prepared-{mapping}-full.txt')
        assert l==r
    assert closures['prepared-identity-full']==closures['prepared-reverse-full']

    traces={};work=root/'experiments/2026-10-21-runpod-mechanism-collection/remote-run-01/server-backup/work'
    order=captures['prepared-identity-full']['op_order']
    anc=closures['prepared-identity-full']
    for label in ('m512q2','m512q4','largenq2'):
        db=work/f'trace-{label}/timeline.sqlite';conn=sqlite3.connect(db)
        rows=conn.execute('SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE ? ORDER BY k.start',('%lsa_cutlass_gemm<%',)).fetchall()
        bydev={d:[r for r in rows if r[2]==d] for d in range(4)}
        assert all(len(v)==484 for v in bydev.values())
        device_order={d:[i for i,_,dest in order if dest==d] for d in range(4)}
        violations=[];overlaps=[];chain_fraction=[]
        for iteration in range(100):
            times={i:bydev[d][84+iteration*4+k][:2] for d in range(4) for k,i in enumerate(device_order[d])}
            for j in anc:
                for i in anc[j]:
                    if times[i][1]>times[j][0]:violations.append((iteration,i,j))
            duration=sum(b-a for a,b in times.values());span=max(b for a,b in times.values())-min(a for a,b in times.values())
            overlaps.append(duration/span)
            finish={}
            for i,_,_ in order:
                finish[i]=max((finish[j] for j in anc[i]),default=0)+(times[i][1]-times[i][0])
            # Descriptive bound using durations from this very trace. It is
            # not a prediction of runtimes after removing dependency edges.
            assert max(finish.values())<=span
            chain_fraction.append(max(finish.values())/span)
        assert not violations
        first=min(r[0] for r in rows)
        uploads=conn.execute("SELECT m.end FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId JOIN StringIds s ON s.id=r.nameId WHERE m.copyKind=1 AND s.value LIKE 'cudaMemcpy_%'").fetchall()
        traces[label]={'all_precedence_checks_pass':True,'iterations':100,
            'precedence_pairs_per_iteration':sum(map(len,anc.values())),
            'median_sum_kernel_durations_over_global_envelope':statistics.median(overlaps),
            'median_observed_duration_weighted_chain_over_envelope':statistics.median(chain_fraction),
            'scope':'Four devices; an overlap indicator under profiling, not hardware utilization.',
            'sqlite_sha256':hashlib.sha256(db.read_bytes()).hexdigest()}
        traces[label]['initial_upload_observation']={
            'synchronous_api_htod_activity_count_including_initialization':len(uploads),
            'last_upload_end_before_first_gemm_us':(first-max(r[0] for r in uploads))/1000,
            'scope':'Observed under profiling; not a general readiness guarantee.'}

    # Isolated mutation: no repository file or measured package is modified.
    base=ledger_module.execution_ledger((4,512,4096,512),'splitk2',environment_id='audit')
    with tempfile.TemporaryDirectory(prefix='edmir-audit-') as td:
        t=Path(td);shutil.copytree(root/'EDM-IR/edmir',t/'EDM-IR/edmir',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for path in ledger_module.SOURCE_PATHS.values():
            dst=t/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/path,dst)
        model=t/'EDM-IR/edmir/compiler/model.py';text=model.read_text()
        old='return (*resources, f"rank{self.rank}.compute")';new='return (f"rank{self.rank}.compute",)'
        assert text.count(old)==1;model.write_text(text.replace(old,new))
        code="from edmir.compiler.execution_ledger import execution_ledger; import json; x=execution_ledger((4,512,4096,512),'splitk2',environment_id='audit'); print(json.dumps({'identity':x['identity_sha256'],'edges':len(x['source_facts']['program_dependencies'])}))"
        x=subprocess.run([sys.executable,'-c',code],cwd=t,env={**os.environ,'PYTHONPATH':str(t/'EDM-IR')},capture_output=True,text=True,timeout=15,check=True)
        changed=json.loads(x.stdout)
    mutation={'original_identity':base['identity_sha256'],'original_edges':len(base['source_facts']['program_dependencies']),
              'mutated':changed,'same_identity':base['identity_sha256']==changed['identity'],
              'reason':'model.py defines Op.resources but is absent from SOURCE_PATHS and REVIEWED_SOURCE_HASHES',
              'scope':'execution ledger only; whole GPU package has a separate full source seal'}
    baseline_support={}
    for name in HOST_BASELINES:
        try:ledger_module.execution_ledger((4,512,4096,512),'host',environment_id='audit',baseline_id=name)
        except ValueError as e:baseline_support[name]={'accepted':False,'reason':str(e)}
        else:baseline_support[name]={'accepted':True}
    result={'gpu_runs':0,'trained_models':0,'production_source_changed':False,'captures':captures,
            'trace_consistency':traces,'source_guard_mutation':mutation,'ledger_baseline_support':baseline_support,
            'limitation':'Replay models happens-before constraints only; no CUDA numerical or performance validation of alternatives.'}
    (here/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in captures.items():print(k,'IR',len(v['ir_chain']),'runtime',len(v['runtime_chain']),'extra',len(v['extra_transitive_precedences']))
    print('mutation',mutation);print('traces',traces)

if __name__=='__main__':main()
