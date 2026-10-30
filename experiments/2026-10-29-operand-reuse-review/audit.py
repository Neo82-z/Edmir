"""Verify buffer ownership and summarize the fixed, conditional reuse sweep."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    generated = ROOT / 'experiments/2026-10-28-paired-gpu-readiness/p4/generated/pair_p4_w000_splitk8/edmir_program.hpp'
    runtime = ROOT / 'experiments/2026-10-28-shape-transfer-review/source-audit/cuda_backend.cpp'
    source = generated.read_text()
    buffers = re.findall(r'\{"([^"]+)", (\d+), BufferRole::k(\w+), (\d+)ull\}', source)
    ops = re.findall(r'\{"(fused_transfer_gemm[^\"]+)", OpKind::kFusedTransferGemm, (\d+), (\d+), QueueKind::k(\w+),', source)
    section = source.split('kAccesses{{', 1)[1].split('}};', 1)[0]
    accesses = [(int(op), int(buf), write == 'true') for op, buf, write in re.findall(r'\{(\d+), (\d+), (true|false)\}', section)]
    if (len(buffers), len(ops), len(accesses)) != (36, 16, 48):
        raise AssertionError('generated contract changed; review the source')
    per_rank = []
    for rank in range(4):
        indices = [i for i, (_, dst, _, _) in enumerate(ops) if int(dst) == rank]
        weights = [(buf, buffers[buf][0]) for op, buf, write in accesses
                   if op in indices and buffers[buf][2] == 'Weight' and not write]
        if len(weights) != 4 or len(set(weights)) != 1:
            raise AssertionError('source GEMMs do not share one destination weight')
        sources = [int(ops[i][2]) for i in indices]
        if sources != [(rank + delta) % 4 for delta in range(4)]:
            raise AssertionError('local-first cyclic order changed')
        if any(ops[i][3] != 'Compute' for i in indices):
            raise AssertionError('destination compute stream changed')
        per_rank.append({'rank': rank, 'op_indices': indices, 'sources_in_header_order': sources,
                         'weight_buffer_index': weights[0][0], 'weight_name': weights[0][1],
                         'weight_bytes': int(buffers[weights[0][0]][3])})
    backend = runtime.read_text()
    for marker in ('for (const auto op_index : order_)', 'order_ = topological_order(program_)',
                   'fused_cutlass_workspaces_[op_index],\n        ranks_[op.rank].compute)',
                   'cudaEventRecord(events_[op_index][op.rank], ranks_[op.rank].compute)'):
        if marker not in backend:
            raise AssertionError('runtime stream/dependency source changed')

    result = json.loads((HERE / 'run-01/results.json').read_text())
    differences = result['short_K_pair_differences']
    if len(differences) != 80:
        raise AssertionError('incomplete short-K pairs')
    operand_equality = all(all(d['warm_q2_minus_q1_read_miss_bytes'][x] == 0
                              for x in ('A_local', 'A_remote', 'B')) for d in differences)
    output_nonnegative = all(d['warm_q2_minus_q1_read_miss_bytes']['C'] >= 0
                             and d['warm_q2_minus_q1_write_bytes'] > 0 for d in differences)
    collisions = [r for r in result['scenarios'] if r['shape_P_M_K_N'] in
                  ([4, 64, 4096, 2048], [4, 256, 4096, 512])]
    examples = []
    for r in collisions:
        if r['q'] == 2 and r['scenario']['tile_order'] == 'm_fast' and r['scenario']['remote_A_cached']:
            examples.append({'shape_P_M_K_N': r['shape_P_M_K_N'], 'capacity_bytes': r['scenario']['capacity_bytes'],
                             'warm_read_miss_bytes': {x: v['read_miss_bytes'] for x, v in r['passes'][1]['operands'].items()}})
    facts = {
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (generated, runtime)},
        'destination_weight_sharing': per_rank,
        'runtime_revision': 'f584d7d (source snapshot)',
        'generated_program_is_recent_readiness_packet': True,
        'archived_executed_binary_verified': False,
        'short_K_pairs': len(differences),
        'all_short_K_warm_AB_read_miss_deltas_zero': operand_equality,
        'all_short_K_C_read_miss_deltas_nonnegative_and_writes_increase': output_nonnegative,
        'equal_work_collision_examples': examples,
        'interpretation': 'Capacity/order scenarios distinguish some equal-work shapes, but provide no short-K q2 A/B miss reduction in this grid. Not proof of physical cache behavior or measured latency.',
        'scoring_change_admitted': False,
        'next_local_target': 'Derive CTA/stage pipeline and semaphore dependency constraints from pinned source; retain unidentified service times as explicit variables, not fitted labels.',
        'references': [{'url': 'https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/',
                        'section': 'A100 L2 cache; original technical body, not page AI summary',
                        'facts': ['40 MB L2', 'two partitions', 'residency controls', 'data compression'],
                        'inference_limit': 'Total capacity does not establish full residency, peer-read cache placement, or physical traffic in these runs.'}],
    }
    (HERE / 'run-01/audit.json').write_text(json.dumps(facts, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: facts[k] for k in ('short_K_pairs', 'all_short_K_warm_AB_read_miss_deltas_zero',
                                          'all_short_K_C_read_miss_deltas_nonnegative_and_writes_increase',
                                          'scoring_change_admitted')}, indent=2))


if __name__ == '__main__':
    main()
