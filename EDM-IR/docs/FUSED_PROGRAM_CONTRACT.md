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
| `nccl_device` + JSON | yes | semantic artifact |
| `nccl_device` + C++ / `cuda_reference` | rejected | guarded SM80 FP16 reference |
| `nccl_device` + C++ / `cutlass` | rejected | experimental SM80 tensor-core path |

The fused target additionally requires full peer-to-peer connectivity and the
`cuda_reference` or CUTLASS provider. cuBLAS is an independent library launch
and cannot satisfy the one-kernel semantic contract. `cuda_reference` resolves
to `lsa_naive_f16`. The CUTLASS path resolves the remote A pointer inside a
custom device wrapper, substitutes it into `GemmKernel::Params`, and invokes
the ordinary predicated tensor-core mainloop and epilogue in the same launch.

## Why the decomposed cost model still rejects it

The existing model is calibrated from independent transfer, GEMM, event, and
dispatch measurements. Summing or overlapping those terms does not measure an
in-kernel communication/GEMM pipeline. `simulate_program()` therefore rejects
the fused op rather than assigning it an invented duration. Fused selection is
instead performed by a measured, environment-fingerprinted decision profile;
the two models are not mixed.

## Runtime experiment and admission gate

The ABI/reference-runtime gate and first fixed-machine measurements are
complete. Any new profile must still capture:

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
measured host-NCCL plus GEMM-provider schedule. The retained A100/NV12 profile
meets that criterion only for its explicit 45-point envelope. Missing opt-in,
insufficient evidence, an environment mismatch, or an unmeasured shape emits
the host-NCCL plus cuBLAS Program and records the rejection in
`selection.json`.

Run the admission probe with:

```bash
python -m edmir probe-runtime \
  --out runtime-capabilities.json --require-fused
```

The command records GPUs, topology, CUDA compiler, NCCL version/header set,
required Device API symbols, and a compile-link-runtime check against the
selected NCCL prefix. Set `NCCL_HOME=/path/to/nccl/build` for an isolated build.
Exit status 2 means the fused gate is closed while still leaving the complete
JSON report.

## Reference-kernel synchronization contract

The first implementation registers one symmetric NCCL window per fused tile.
The destination kernel resolves the source allocation with
`ncclGetLsaPointer`, reads FP16 input directly, multiplies by its local FP16
weight, accumulates in FP32, and writes FP16 output. It intentionally uses a
naive CUDA loop rather than tensor cores.

No device barrier is needed under this narrow runner contract: external inputs
and weights are populated by synchronous host copies before `execute()`, remain
immutable during execution, and the registered windows outlive all launches.
A future producer-generated or mutable tile must add explicit release/acquire
publication; this reference must not be generalized by assumption.

## Measured traffic boundary

The first CUTLASS implementation assigns one CTA to each `(M tile, N tile)`.
Every N tile therefore reloads the same remote A tile. For a kernel with
threadblock width `T_N`, the compiler-visible replay factor is

```text
remote_a_replay = ceil(N / T_N)
estimated_remote_a_load_bytes = remote_input_bytes * remote_a_replay
```

The grouped-N schedule contract generalizes that accounting. Let `g` be
`n_tiles_per_cta`, the number of adjacent N tiles computed while one CTA keeps
the remote A tile live:

```text
n_tiles = ceil(N / T_N)
remote_a_replay = ceil(n_tiles / g)
estimated_remote_a_load_bytes = remote_input_bytes * remote_a_replay
estimated_ctas_per_op = ceil(M / T_M) * remote_a_replay * split_k_slices
```

This is a two-sided compiler contract, not a speedup formula. Increasing `g`
reduces the modeled remote bytes but also removes CTA-level parallelism;
increasing split-K restores CTA supply without reducing those bytes and adds a
reduction cost. The analysis is valid only for a kernel that actually retains
the A tile across its grouped N work. The first source implementation uses
`g=2`: each supported CUTLASS family doubles its N-threadblock width so one CTA
shares each K-stage A load across twice the output columns. Factors above two
remain JSON-analysis-only. The `g=2` path is not eligible for automatic
selection until it compiles and passes numerical and held-out latency
validation on the pinned A100 stack.

This estimate counts only non-local source/destination pairs. It is not a
latency model, because SM-initiated LSA reads and copy-engine transfers have
different fixed and bandwidth costs. On 4x A100, direct fusion won for the tiny
`M8 K64 N64` validation shape but lost for every `M64 K4096` N-sweep point,
including replay one. Adding serial split-K=8 at `M64 K4096 N64` kept bytes and
replay fixed, increased estimated CTAs per op from one to eight, and changed the
result from 82.3% slower to 11.8% faster than the host baseline. Those features
are retained in every admitted selection report; they are descriptive traffic
terms, not an invented latency prediction.

A fixed-`M64 N64` K sweep subsequently isolated the concurrency boundary from
N-tile replay. Direct fusion won for K=256, 512, and 1024; split-K=8 won for
K=2048 and 4096. At K=2048, direct reached host parity while split-K remained
24.1% below host. This is evidence for distinct kernel candidates, not a global
K threshold: the default selector stays closed until held-out M/N shapes
preserve the choice direction.

The follow-up completed the full `M={32,64,128}` by `N={64,128,256}` grid.
Training on M64/N64 and holding out the other eight M/N pairs produced 100%
direction accuracy and zero selection regret over 40 held-out problems. This
admits the boundary as a measured A100/NV12 profile, but not as a portable
compiler default. `compile-fused` requires an exact environment fingerprint,
the exact measured M/N/K set, full NV12 P2P topology, the held-out evidence
gate, and `--allow-experimental-fused`. The selector chooses direct through
K=1024 and split-K8 from K=2048 only after all guards pass; otherwise it emits
the host fallback.
