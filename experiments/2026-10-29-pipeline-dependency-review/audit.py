"""Pinned template trace, not a statement about an unarchived executable."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUTLASS = ROOT / 'experiments/2026-10-28-paired-gpu-readiness/dependencies/cutlass'
COMMIT = 'dcf215af68a2d08d305076c152a06f201728cd53'


def audit():
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=CUTLASS,text=True,capture_output=True,check=True,timeout=5).stdout.strip()
    if revision != COMMIT:
        raise AssertionError('pinned dependency changed')
    requirements = {
        'gemm/device/gemm.h': ['SharedMemoryClearOption::kNone'],
        'gemm/threadblock/default_mma.h': ['((sizeof_bits<ElementA>::value * kAlignmentA) == 128)',
                                         '? cutlass::arch::CacheOperation::Global', 'Stages, SharedMemoryClear>'],
        'gemm/threadblock/default_mma_core_sm80.h': ['layout::RowMajor, ElementB_, layout::RowMajor, ElementC_',
                                                  'kCacheOpA = CacheOpA', 'PitchLinearWarpRakedThreadMap'],
        'gemm/threadblock/mma_base.h': ['(WarpGemm::kK / Operator::Policy::MmaShape::kK)'],
        'gemm/warp/default_mma_tensor_op.h': ['cutlass::arch::Mma<InstructionShape_, 32, ElementA'],
        'transform/pitch_linear_thread_map.h': ['struct PitchLinearWarpRakedThreadMap',
                                              'Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided'],
        'gemm/threadblock/mma_multistage.h': ['stage < Base::kStages - 1', 'cp_async_wait<Base::kStages - 2>()',
                                           'warp_mma_k + 2 == Base::kWarpGemmIterations',
                                           'gemm_k_iterations > (-Base::kStages + 1)', 'cp_async_wait<0>()'],
        'gemm/kernel/gemm.h': ['semaphore.wait(threadblock_tile_offset.k())', 'semaphore.release(lock)',
                              'mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators)'],
        'arch/memory_sm80.h': ['cp.async.cg.shared.global', 'cp.async.wait_group %0', 'cp.async.wait_all;'],
    }
    evidence = {}
    for rel, markers in requirements.items():
        path = CUTLASS / 'include/cutlass' / rel
        content = path.read_text()
        for marker in markers:
            if marker not in content:
                raise AssertionError('source contract changed: ' + rel + ' / ' + marker)
        # Confirm reviewed headers themselves, not only checkout HEAD.
        tracked = subprocess.run(['git','show',COMMIT+':include/cutlass/'+rel],cwd=CUTLASS,capture_output=True,check=True,timeout=5).stdout
        if path.read_bytes() != tracked:
            raise AssertionError('modified dependency header: ' + rel)
        evidence[str(path.relative_to(ROOT))] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'markers': {marker: content[:content.index(marker)].count('\n')+1 for marker in markers}}
    kernel = (CUTLASS/'include/cutlass/gemm/kernel/gemm.h').read_text()
    if not kernel.index('mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators)') < kernel.index('semaphore.wait(threadblock_tile_offset.k())') < kernel.index('epilogue(output_op, iterator_D, accumulators, iterator_C)') < kernel.index('semaphore.release(lock)'):
        raise AssertionError('mainloop/epilogue dependency changed')
    local_sources = [ROOT/'experiments/2026-10-28-shape-transfer-review/source-audit'/name
                     for name in ('cutlass_sm80_gemm.cuh','nccl_device_fused_cutlass.cu')]
    for path in local_sources:
        evidence[str(path.relative_to(ROOT))] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    header = local_sources[0].read_text()
    if 'GemmShape<16, 8, 16>' not in header or '    3,\n    8,\n    8,\n    SplitKSerial>' not in header:
        raise AssertionError('local instruction/stage/alignment template changed')
    return {'CUTLASS_revision':COMMIT,'sources':evidence,
            'source_template':{'stages':3,'warp_shape':[32,64,32],'instruction_shape':[16,8,16],
                               'warp_k_groups':2,'alignment_elements':8,'element_bits':16,
                               'cache_policy_A_B':'Global -> cp.async.cg','clear_policy':'kNone',
                               'prologue_copy_groups':2,'mainloop_copy_subgroups':2,
                               'wait_location':'after first warp-k MMA and both copy subgroups; before second warp-k MMA',
                               'last_drain':'explicit commit then wait_all (implicit empty commit), then CTA barrier',
                               'mainloops_serialized_by_semaphore':False},
            'unverified':['actual historical binary and PTX/SASS','registers and launch occupancy',
                          'per-warp issue time','cache residency and peer placement','full-program service times'],
            'official_reference':'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all',
            'compiler_admission':False}


if __name__ == '__main__':
    result = audit()
    (HERE/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['source_template'],indent=2))
