#!/usr/bin/env python3
"""Generate real compiler manifests and run bounded CPU API contract tests.

Requires local official CUDA/cuBLAS/NCCL/NVTX headers, not GPUs or their libraries.
Every output directory is new; historical experiment bundles are never modified.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler import A2AGemmProblem, CppManifestBackend, Target, TopologySnapshot
from edmir.compiler.schedule import build_fused_program, build_sequential_program
from edmir.compiler.execution_policy import with_dependency_policy, UNIFORM_PEER_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--api-header-dir', action='append', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    generated = out/'generated'
    generated.mkdir()
    problem = A2AGemmProblem(4, 512, 4096, 512)
    topology = TopologySnapshot(4, ('cpu-contract-fixture',)*4, ((True,)*4,)*4)
    backend = CppManifestBackend()
    for fused in (True, False):
        program = (with_dependency_policy(build_fused_program(problem, chunk_rows=512), UNIFORM_PEER_POLICY)
                   if fused else build_sequential_program(problem))
        target = Target('cpu-contract-fixture', topology, arch='sm_80', gemm_provider='cutlass',
                        gemm_kernel='sm80_64x128x32_splitk2' if fused else 'sm80_64x128x32',
                        communication_api='nccl_device' if fused else 'host_nccl')
        artifact = backend.emit(program, target)
        header = artifact.files['edmir_program.hpp']
        name = 'edmir_program.hpp' if fused else 'edmir_host_program.hpp'
        if not fused:
            header = header.replace('namespace edmir_generated', 'namespace edmir_host_generated')
        (generated/name).write_text(header)
        (generated/('fused-program.json' if fused else 'host-program.json')).write_text(
            json.dumps(program.to_dict(), indent=2)+'\n')
    commands = [
        ['cmake', '-S', str(root/'runtime'), '-B', str(out/'build'), '-DCMAKE_BUILD_TYPE=Release',
         '-DEDMIR_CUDA_API_TRACE_TESTS=ON', '-DEDMIR_ENABLE_CUDA=OFF',
         '-DEDMIR_CUDA_API_HEADER_DIRS='+';'.join(str(Path(d).resolve()) for d in args.api_header_dir),
         '-DEDMIR_GENERATED_DIR='+str(generated)],
        ['cmake', '--build', str(out/'build'), '--parallel', '2'],
        ['ctest', '--test-dir', str(out/'build'), '--output-on-failure', '-V'],
    ]
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    for i, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out/f'{i}.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            print(result.stdout+result.stderr, file=sys.stderr)
            return result.returncode
    print((out/'2.log').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
