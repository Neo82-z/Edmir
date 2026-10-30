"""Describe separate Nsight traces; never emit primary timing labels."""
from pathlib import Path
import json, sqlite3, statistics, hashlib, sys

work = Path(sys.argv[1])
med = statistics.median
result = {}
for db in sorted(work.glob('trace-*/timeline.sqlite')):
    diagnostic = json.loads((db.parent/'diagnostic.json').read_text())
    assert diagnostic['state'] == 'complete' and diagnostic['profiler'] is True
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    all_kernels = [dict(r) for r in c.execute(
        'SELECT k.*,s.value AS kernel_name FROM CUPTI_ACTIVITY_KIND_KERNEL k '
        'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
    kernels = [r for r in all_kernels if 'lsa_cutlass_gemm<' in r['kernel_name']
               or ('cutlass::' in r['kernel_name'] and 'nccl' not in r['kernel_name'])]
    devices = sorted(set(r['deviceId'] for r in kernels))
    assert len(devices) == 4
    by_device = {d: [r for r in kernels if r['deviceId'] == d] for d in devices}
    # Audited benchmark: one correctness execute, 20 warmups, 100 timed
    # executes; each destination's single compute stream executes four GEMMs.
    assert all(len(v) == 484 for v in by_device.values()), {d:len(v) for d,v in by_device.items()}
    assert all(len(set(r['streamId'] for r in v)) == 1 for v in by_device.values())
    timed = {d: [v[84+i*4:84+(i+1)*4] for i in range(100)] for d,v in by_device.items()}
    for groups in timed.values():
        assert all(all(a['end'] <= b['start'] for a,b in zip(g,g[1:])) for g in groups)
    windows = [(min(timed[d][i][0]['start'] for d in devices),
                max(timed[d][i][-1]['end'] for d in devices)) for i in range(100)]
    assert all(a[1] <= b[0] for a,b in zip(windows,windows[1:]))
    per_device = {}
    for d,groups in timed.items():
        per_device[d] = {
            'median_sum_of_four_gemm_us': med(sum(r['end']-r['start'] for r in g)/1000 for g in groups),
            'median_gemm_span_us': med((g[-1]['end']-g[0]['start'])/1000 for g in groups),
            'median_kernel_us_by_launch_ordinal': [med((g[j]['end']-g[j]['start'])/1000 for g in groups) for j in range(4)],
        }
    runtime = [dict(r) for r in c.execute(
        'SELECT r.start,r.end,s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
        'JOIN StringIds s ON s.id=r.nameId WHERE r.start>=? AND r.end<=?',
        (windows[0][0],windows[-1][1]))]
    api = {}
    for r in runtime: api.setdefault(r['name'],[]).append((r['end']-r['start'])/1000)
    text = (db.parent/'profiled.stdout.txt').read_text()
    measurements = []
    for line in text.splitlines():
        if line.startswith('{'):
            try: obj=json.loads(line)
            except json.JSONDecodeError: continue
            if 'samples_ms' in obj: measurements.append(obj)
    assert len(measurements)==1 and measurements[0]['correctness_passed']
    fields = ('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread',
              'staticSharedMemory','dynamicSharedMemory','localMemoryPerThread')
    resources = sorted(set(tuple(r[k] for k in fields) for r in kernels))
    result[db.parent.name] = {
        'job': diagnostic['job'], 'total_gemm_kernels':len(kernels),
        'profiled_wall_median_us': measurements[0]['median_ms']*1000,
        'median_all_device_gemm_envelope_us':med((end-start)/1000 for start,end in windows),
        'median_between_gemm_envelopes_us':med((b[0]-a[1])/1000 for a,b in zip(windows,windows[1:])),
        'per_device':per_device, 'launch_resources':[dict(zip(fields,r)) for r in resources],
        'runtime_api_within_gemm_window':{name:{'count':len(v),'sum_us':sum(v),'median_us':med(v)} for name,v in api.items()},
        'sqlite_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),
    }
output = {'profiler':True, 'primary_dataset':False, 'compiler_admission':False,
          'scope':'Single trace per configuration; ordinal slicing verified against kernel counts and stream serialization. GEMM envelopes exclude host communication before GEMMs. API window is bounded by GEMMs, not exact CPU iteration markers. Do not subtract these times from unprofiled wall times or infer physical traffic / semaphore stalls.',
          'traces':result}
(work/'trace-analysis.json').write_text(json.dumps(output,indent=2)+'\n')
for name,obj in result.items():
    print(name, 'profiled_wall',round(obj['profiled_wall_median_us'],2),
          'gpu_envelope',round(obj['median_all_device_gemm_envelope_us'],2),
          'four_gemms',[round(v['median_sum_of_four_gemm_us'],2) for v in obj['per_device'].values()])
