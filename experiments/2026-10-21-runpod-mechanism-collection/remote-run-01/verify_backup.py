"""Verify the downloaded archive, every file, and supplemental measurements."""
from pathlib import Path
import hashlib, json, subprocess, sys, tarfile

here=Path(__file__).resolve().parent
root=here.parents[2]
sys.path[:0]=[str(root/'EDM-IR'),str(root/'EDM-IR/tools')]
from run_mechanism_gpu import check_row
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
archive=here/'completed-run-01.tar.gz'
receipt=json.loads((here/'completed-run-01.json').read_text())
expected='0c28679a0a0052131c46ef86718f7e0f91cb743b19f0abcd031aae813f8f4856'
assert receipt['sha256']==expected==sha(archive)
assert receipt['bytes']==archive.stat().st_size
out=here/'server-backup'
out.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(out,filter='data')
manifest=json.loads((out/'result-file-manifest.json').read_text())
for name,entry in manifest.items():
    f=out/name
    assert f.is_file() and f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],name
assert len(manifest)==receipt['files']

work=out/'work'
for stage in ('anchors','mechanisms','geometry'):
    assert sha(work/(stage+'-01')/'measurements.jsonl')==sha(here/'results'/(stage+'-01')/'measurements.jsonl')
p=json.loads((here/'revision-01/package.json').read_text())['package']
infos=json.loads((here/'revision-01/inputs.json').read_text())['workloads']
diag=work/'short-host-diagnostic-01'
design=json.loads((diag/'design.json').read_text())
summary=json.loads((diag/'summary.json').read_text())
rows=[json.loads(line) for line in (diag/'measurements.jsonl').read_text().splitlines()]
assert len(rows)==len(design['jobs'])==24
assert sha(diag/'measurements.jsonl')==summary['raw_sha256']
context=json.loads((diag/'context.json').read_text())
for row,job in zip(rows,design['jobs']):
    trial={**p,'warmup':job['diagnostic_warmup']}
    assert row==check_row(row,job,infos[job['workload_id']],trial)
    assert row['runtime']==context['runtime']

for f in work.glob('trace-*/diagnostic.json'):
    d=json.loads(f.read_text())
    assert d['state']=='complete' and d['profiler'] is True
    for name,hash_value in d['reports'].items():assert sha(f.parent/name)==hash_value
server_analysis=json.loads((work/'trace-analysis.json').read_text())
run=subprocess.run([sys.executable,str(here/'analyze_traces.py'),str(work)],capture_output=True,text=True,timeout=60,check=True)
assert server_analysis==json.loads((work/'trace-analysis.json').read_text())
primary=json.loads((here/'validated-analysis.json').read_text())
result={'archive_sha256':expected,'verified_files':len(manifest),
        'primary_processes':sum(v['processes'] for v in primary['stages'].values()),
        'primary_samples':sum(v['timed_samples'] for v in primary['stages'].values()),
        'supplemental_processes':len(rows),'supplemental_samples':sum(len(r['samples_ms']) for r in rows),
        'profiled_diagnostics':len(server_analysis['traces']),
        'all_numerical_checks_passed':True,'server_local_primary_summaries_equal':True,
        'server_local_trace_summaries_equal':True,'new_training_runs':0,'compiler_admission':False,
        'pod_stopped':False,'backup_complete':True}
(here/'backup-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
