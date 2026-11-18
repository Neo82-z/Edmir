"""Post-collection CPU export only; does not run or remeasure any candidate."""
from pathlib import Path
import subprocess,json,datetime,os,signal,time
root=Path('/workspace')
nsys='/opt/edmir-nsys/opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys'
entries=[]
export_deadline=json.loads((root/'extension-guard.json').read_text())['platform_stop_epoch']-90
for report in sorted((root/'results-precision').rglob('timeline.nsys-rep')):
    out=report.parent/'cpu-analysis'
    if out.exists():
        entries.append({'report':str(report.relative_to(root)),'status':'preserved_existing_export'});continue
    if export_deadline-time.time()<5:
        entries.append({'report':str(report.relative_to(root)),'status':'raw_report_preserved_recovery_reserve'});continue
    out.mkdir(exist_ok=False)
    cmd=[nsys,'stats','--report','cuda_gpu_kern_sum,nvtx_sum','--format','csv',str(report)]
    item={'report':str(report.relative_to(root)),'command':cmd,'diagnostic_only':True}
    with (out/'stats.stdout').open('w') as o,(out/'stats.stderr').open('w') as e:
        process=subprocess.Popen(cmd,stdout=o,stderr=e,start_new_session=True)
        try:
            item['exit_code']=process.wait(timeout=min(50,max(1,export_deadline-time.time())))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            item['error']='CPU export timeout; raw report preserved'
    entries.append(item)
(root/'session/trace-export-resume.json').write_text(json.dumps({'exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entries':entries},indent=2)+'\n')
print(json.dumps(entries))
