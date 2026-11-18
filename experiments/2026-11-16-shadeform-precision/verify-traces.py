"""Check trace coverage without treating profiled durations as timing labels."""
import json
from pathlib import Path
import sqlite3

root=Path(__file__).resolve().parent
data=root/'recovered'
queue=json.loads((data/'window/queue.json').read_text())
cases=[]
for stage in queue['stages']:
    for name in stage['traces']:
        folder=data/'results-precision'/stage['name']/'diagnostic'/name
        db=folder/'timeline.sqlite'
        entry={'case':name,'stage':stage['name'],'diagnostic_only':True}
        if not db.exists():
            entry['status']='sqlite_missing';cases.append(entry);continue
        c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
        per_gpu=c.execute('SELECT deviceId,count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId').fetchall()
        assert [row[0] for row in per_gpu]==list(range(8)),name
        assert all(row[1]>0 for row in per_gpu)
        assert c.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end<=start').fetchone()[0]==0
        kernels=c.execute('''SELECT s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                                    k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,count(*)
                             FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
                             GROUP BY s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                                      k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory''').fetchall()
        op_ranges=c.execute("SELECT count(*) FROM NVTX_EVENTS WHERE text LIKE 'edm.op:%' AND end>start").fetchone()[0]
        assert op_ranges>0,name
        entry.update(status='eight_gpu_kernels_and_operator_ranges_present',kernels_per_gpu=per_gpu,
                     completed_operator_ranges=op_ranges,kernel_configurations=[{
                         'kernel':row[0],'grid':row[1:4],'block':row[4:7],
                         'registers_per_thread':row[7],'static_shared_bytes':row[8],
                         'dynamic_shared_bytes':row[9],'launches':row[10]} for row in kernels])
        c.close();cases.append(entry)
report={'cases':cases,'all_seven_covered':len(cases)==7 and all(x['status']=='eight_gpu_kernels_and_operator_ranges_present' for x in cases),
        'limitations':['Profiler durations are excluded from timing labels.',
                       'CUDA event completion tracing was enabled; event-based dependency arrows can be perturbed.',
                       'Kernel coverage/configuration does not directly measure HBM/NVLink traffic or prove within-kernel overlap.']}
(root/'trace-coverage.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'trace_cases':len(cases),'all_seven_covered':report['all_seven_covered']}))
raise SystemExit(not report['all_seven_covered'])
