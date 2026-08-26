# Fused All2All-to-GEMM Program contract

This document fixes the smallest compiler meaning of communication/compute
fusion before an implementation is benchmarked. It prevents a faster sequence
of NCCL and GEMM launches from being mislabeled as one fused kernel.

## Semantic boundary

The Dataflow IR remains:

```text
activation -- all_to_all -- gemm -- output
                              ^
                              |
                            weight
```

Fusion is a lowering choice, not a new frontend operation. For each
`(source rank, destination rank, chunk)` tile, `lower_fused_a2a_gemm()` emits
one `fused_transfer_gemm` Program op with this contract:

- `rank` is the destination and `peer_rank` is the source;
- reads are exactly the source-rank input tile and destination-rank weight;
- the only write is the destination-rank output tile;
- no staging buffer exists;
- resources are source communication, destination communication, and
  destination compute;
- program order on any shared resource is a derived dependency.

The ordinary `transfer -> staging -> gemm` lowering stays unchanged and is the
fallback.

## Capability and backend matrix

| Target/backend | Unfused Program | Fused Program |
|---|---:|---:|
| `host_nccl` + JSON | yes | rejected |
| `host_nccl` + C++ runtime | yes | rejected |
| `nccl_device` + JSON | yes | semantic prototype only |
| `nccl_device` + C++ runtime v0.1 | rejected | rejected |

The fused target additionally requires full peer-to-peer connectivity and the
CUTLASS provider. cuBLAS is an independent library launch and cannot satisfy
the one-kernel semantic contract. The
current C++ ABI and runtime intentionally do not contain a placeholder fused
opcode. Adding an enum that throws at launch time would be weaker than the
current compile-time capability error.

## Why the cost model rejects it

The existing model is calibrated from independent transfer, GEMM, event, and
dispatch measurements. Summing or overlapping those terms does not measure an
in-kernel communication/GEMM pipeline. `simulate_program()` therefore rejects
the fused op rather than assigning it an invented duration. Fused selection can
begin only after a profile contains measurements from the exact fused kernel
identity and launch path.

## Runtime experiment gate

Before extending the C++ ABI, capture the following on one fixed machine:

1. environment provenance: GPU UUIDs, topology, driver, CUDA, NCCL, clocks,
   persistence state, and Device API symbols available in the installed headers;
2. a minimal NCCL Window/LSA or CuTe DSL kernel that reads a remote tile and
   produces a numerically checked destination result;
3. warm steady-state latency for identical `(world, M, N, K, dtype, chunk)`
   points under unfused and fused implementations;
4. Nsight evidence that communication and tensor-core work occur inside the
   intended kernel boundary, including synchronization and resource occupancy;
5. held-out shape and chunk points used only for direction accuracy and
   schedule-regret evaluation.

The minimum admission criterion is not a single speedup. It is: zero numerical
failures, an explicit runtime capability check, reproducible kernel identity,
held-out direction accuracy, and lower measured runtime for at least one
declared workload region. Until then, the compiler continues to emit the
measured host-NCCL plus GEMM-provider schedule.
