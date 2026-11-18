#!/usr/bin/env python3
"""CPU-only Linux link of new descriptors against unchanged precompiled runtime."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import time

ROOT=Path('/workspace/p1p2')
BUNDLE=ROOT/'bundle'
BUILD=ROOT/'build'
ARCHIVE=Path('/archive-build')
OLD_SOURCE=Path('/archive-source')
DEPS=Path('/workspace/dependencies')
CUDA=Path('/usr/local/cuda-12.8/targets/x86_64-linux')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not list(Path('/dev').glob('nvidia[0-9]*'))
    BUILD.mkdir(exist_ok=False)
    old_receipt=json.loads((ARCHIVE/'receipt.json').read_text())
    source=BUNDLE/'source/EDM-IR/runtime'
    for p in source.rglob('*'):
        if p.suffix in ('.cpp','.cu','.cuh','.hpp'):
            if sha(p)!=sha(OLD_SOURCE/p.relative_to(source)):raise ValueError('runtime source differs: '+str(p))
    for name,h in old_receipt['dependencies'].items():
        if sha(name)!=h:raise ValueError('dependency changed')
    for name in ['libedmir_runtime.a','libedmir_cuda_runtime.a']:shutil.copyfile(ARCHIVE/name,BUILD/name)
    plan=json.loads((BUNDLE/'plan.json').read_text())
    def compile(tag):
        command=['c++','-O3','-DNDEBUG','-std=gnu++20','-DEDMIR_SELECTOR_BATCH=1',
           '-I'+str(BUNDLE/tag),'-I'+str(source/'include'),'-I'+str(DEPS/'nccl/build/include'),
           '-isystem',str(CUDA/'include'),str(source/'tools/benchmark_selector.cpp'),
           '-o',str(BUILD/f'w1_{tag}_batch'),str(BUILD/'libedmir_cuda_runtime.a'),str(BUILD/'libedmir_runtime.a'),
           '-L'+str(CUDA/'lib'),'-L'+str(DEPS/'nccl/build/lib'),
           '-Wl,-rpath,'+str(DEPS/'nccl/build/lib')+':'+str(CUDA/'lib'),
           '-lcudart','-lcublas','-lcublasLt','-lnccl','-lcudadevrt','-lcudart_static','-lrt','-lpthread','-ldl']
        with (BUILD/f'{tag}.txt').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        print('CPU compiled',tag,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile,plan['shapes']))
    receipt=dict(package=sha(BUNDLE/'files.json'),compiler=subprocess.check_output(['c++','--version'],text=True).strip(),
        nvcc=subprocess.check_output(['nvcc','--version'],text=True).strip(),
        dependencies=old_receipt['dependencies'],nccl_library=old_receipt['nccl_library'],
        runtime_archive_sha256={n:sha(BUILD/n) for n in ['libedmir_runtime.a','libedmir_cuda_runtime.a']},
        precompiled_runtime_receipt=sha(ARCHIVE/'receipt.json'),runtime_source_verified_identical=True,
        binaries={tag:sha(BUILD/f'w1_{tag}_batch') for tag in plan['shapes']},GPU_used=False,
        completed_epoch=time.time())
    (BUILD/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
