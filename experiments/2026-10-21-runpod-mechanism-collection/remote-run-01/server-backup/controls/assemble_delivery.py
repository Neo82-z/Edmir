"""Seal local verification and transport archive. Never accesses a GPU."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from prepare_mechanism_gpu import verify_bundle
from prepare_boundary_gpu import verify_bundle as verify_old
from edmir.compiler.measurement_dataset import sha256
from independent_selector_validation import digest


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=verify_bundle(HERE/'bundle')
    tests=(HERE/'python-tests.log').read_text()
    final=(HERE/'final-package-tests.log').read_text()
    cpp=(HERE/'cpu-tests.log').read_text()
    if not tests.rstrip().endswith('OK') or not final.rstrip().endswith('OK') or '100% tests passed, 0 tests failed out of 128' not in cpp:
        raise RuntimeError('local checks did not pass')
    old=verify_old(ROOT/'experiments/2026-10-21-boundary-gpu-package/bundle')
    previous=ROOT/'experiments/2026-10-21-execution-evidence-delivery'
    protected=json.loads((previous/'protected-before.json').read_text())
    if any(sha256(ROOT/rel)!=value for rel,value in protected.items()):raise RuntimeError('protected artifact changed')
    before=previous/'execution-evidence-review.zip'
    if sha256(before)!='4841f51af25ed5f9a776d548be874085bfc8da28ab8142ddd04266e5b43bfad2':raise RuntimeError('previous delivery changed')
    refs=json.loads((HERE/'input-checks.json').read_text())
    inputs=json.loads((HERE/'bundle/inputs.json').read_text())
    info={'status':'local_ready_gpu_validation_pending','package_sha256':digest(p),
          'frozen_programs':len(p['programs']),'stages':p['stages'],
          'python_tests':int(re.search(r'Ran (\d+) tests',tests).group(1)),
          'final_package_recheck_tests':int(re.search(r'Ran (\d+) tests',final).group(1)),
          'cpp_cpu_checks':128,'cpp_check_scope':'layout and API test doubles; not CUDA kernels',
          'input_reference_bytes':sum(v['input_bytes']+v['reference_bytes'] for v in inputs['workloads'].values()),
          'integer_dot_elements_checked':refs['checked_reference_elements'],
          'sampled_fp16_max_abs_error':max(e['max_abs'] for r in refs['records'] for e in r['sampled_serial_fp16_errors'].values()),
          'protected_files_unchanged':len(protected),'previous_boundary_package_sha256':digest(old),
          'training_runs':0,'gpu_runs':0,'cuda_compilation_verified':False,
          'gpu_correctness_verified':False,'independent_validation':False,'compiler_admission':False}
    write(HERE/'verification.json',info)
    top_files=('README.md','server.py','trace-candidates.json','verify_inputs.py','input-checks.json',
               'verification.json','python-tests.log','final-package-tests.log','cpu-tests.log',
               'cpu-build.log','cmake-configure.log','prepare-inputs.log','assemble_delivery.py')
    paths=[HERE/name for name in top_files]
    for folder in ('bundle','dependencies'):
        paths += [f for f in (HERE/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    manifest={str(f.relative_to(HERE)):sha256(f) for f in sorted(paths)}
    write(HERE/'MANIFEST.json',manifest)
    archive=HERE/'runpod-mechanism-collection.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        # Detached shallow Git snapshots still require empty refs directories.
        # File-only packing drops them and makes Git reject the extracted tree.
        directories={parent for f in paths for parent in f.parents if parent!=HERE and HERE in parent.parents}
        directories.update(d for d in (HERE/'dependencies').rglob('*') if d.is_dir())
        for directory in sorted(directories):
            tar.add(directory,arcname='runpod-mechanism-collection/'+str(directory.relative_to(HERE)),recursive=False)
        for f in sorted(paths)+[HERE/'MANIFEST.json']:
            tar.add(f,arcname='runpod-mechanism-collection/'+str(f.relative_to(HERE)),recursive=False)
    # Verify transport in a different path and without site packages.
    with tempfile.TemporaryDirectory(prefix='edmir-transfer-check-') as temporary:
        dest=Path(temporary)
        with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
        extracted=dest/'runpod-mechanism-collection'
        for rel,value in manifest.items():
            if sha256(extracted/rel)!=value:raise RuntimeError('transport hash mismatch: '+rel)
        output=subprocess.run([sys.executable,'-S',str(extracted/'server.py'),'verify'],check=True,
                              capture_output=True,text=True,timeout=60)
        dep_commits={}
        for name in ('cutlass','nccl'):
            dep=extracted/'dependencies'/name
            dep_commits[name]=subprocess.check_output(['git','-C',dep,'rev-parse','HEAD'],text=True).strip()
            if subprocess.check_output(['git','-C',dep,'status','--porcelain','--untracked-files=no'],text=True).strip():
                raise RuntimeError('transport changed tracked dependency files')
    checksum=sha256(archive)
    (HERE/(archive.name+'.sha256')).write_text(checksum+'  '+archive.name+'\n')
    write(HERE/'delivery.json',{'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum,
        'verified_files':len(manifest),'detached_stdlib_verification':json.loads(output.stdout),
        'dependency_commits_after_transport':dep_commits,'gpu_validated':False})
    print(json.dumps({'verification':info,'archive_bytes':archive.stat().st_size,'sha256':checksum},indent=2))


if __name__=='__main__':main()
