#!/usr/bin/env python3
"""Entry point after extracting the delivery on an existing server."""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'bundle'
sys.path[:0]=[str(PACKAGE/'source/EDM-IR'),str(PACKAGE/'source/EDM-IR/tools')]
import run_mechanism_gpu as runner
from prepare_mechanism_gpu import verify_bundle,CUTLASS_REVISION,NCCL_REVISION
from independent_selector_validation import digest


def dependencies(work,jobs):
    nccl=HERE/'dependencies/nccl';cutlass=HERE/'dependencies/cutlass'
    for path,expected in ((cutlass,CUTLASS_REVISION),(nccl,NCCL_REVISION)):
        if runner.base.git_commit(path)!=expected: raise ValueError('dependency revision changed')
        if runner.base.command(['git','-C',path,'status','--porcelain','--untracked-files=no'],capture_output=True).stdout.strip():
            raise ValueError('dependency has tracked modifications')
    cuda=shutil.which('nvcc')
    if not cuda: raise ValueError('nvcc missing; choose a CUDA development image')
    toolchain=runner.base.command([cuda,'--version'],capture_output=True).stdout
    cuda_home=Path(cuda).resolve().parent.parent
    work.mkdir(parents=True,exist_ok=True)
    args=['make','-C',nccl,'src.build',f'-j{jobs}',f'CUDA_HOME={cuda_home}',
          'NVCC_GENCODE=-gencode=arch=compute_80,code=sm_80','NVTX=0']
    with (work/'nccl-build.stdout.txt').open('w') as out,(work/'nccl-build.stderr.txt').open('w') as err:
        runner.base.command(args,timeout=1200,stdout=out,stderr=err)
    library=nccl/'build/lib/libnccl.so'
    runner.base.write_json(work/'dependency-build.json',{'command':[str(v) for v in args],
        'nvcc':toolchain,'nccl_commit':NCCL_REVISION,'cutlass_commit':CUTLASS_REVISION,
        'library_sha256':runner.sha256(library)})


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=('verify','inventory','build-deps','build','preflight','run','trace'))
p.add_argument('--work',type=Path,default=HERE/'work')
p.add_argument('--jobs',type=int,default=4)
p.add_argument('--stage',choices=('anchors','mechanisms','geometry'))
p.add_argument('--budget-seconds',type=int,default=900)
p.add_argument('--resume',action='store_true')
p.add_argument('--job-index',type=int)
p.add_argument('--label',default='01')
a=p.parse_args()
if not 1<=a.jobs<=16:p.error('jobs must be 1..16')
if not a.label.isalnum():p.error('label must be alphanumeric')
work=a.work.resolve();work.mkdir(parents=True,exist_ok=True);build=work/'build'
if a.action=='verify': print(json.dumps({'package_sha256':digest(verify_bundle(PACKAGE))}))
elif a.action=='inventory':runner.inventory(work/f'inventory-{a.label}.json')
elif a.action=='build-deps':dependencies(work,a.jobs)
elif a.action=='build':
    runner.build(PACKAGE,build,cutlass=HERE/'dependencies/cutlass',nccl_source=HERE/'dependencies/nccl',
        nccl_include=HERE/'dependencies/nccl/build/include',nccl_library=HERE/'dependencies/nccl/build/lib/libnccl.so',jobs=a.jobs)
elif a.action=='preflight':
    out=work/f'preflight-{a.label}.json'
    if out.exists():raise ValueError('choose a fresh --label')
    _,ctx,env=runner.preflight(PACKAGE,build);runner.base.write_json(out,{'context':ctx,'environment':env})
elif a.action=='run':
    if not a.stage:p.error('run requires --stage')
    if not runner.collect(PACKAGE,build,work/(a.stage+'-'+a.label),stage=a.stage,
                          resume=a.resume,budget_seconds=a.budget_seconds):raise SystemExit(75)
else:
    if a.job_index is None:p.error('trace requires --job-index from bundle/plan.json')
    runner.trace(PACKAGE,build,work/f'trace-{a.label}',job_index=a.job_index,nsys='nsys')
