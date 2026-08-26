# Edmir compiler architecture

Edmir v0.1 is a deliberately small layered compiler for one problem:

```text
static, equal-size, single-node All2All -> GEMM
```

It borrows two design instincts without copying either project:

- from **tinygrad**: a compact typed graph, ordered rewrite/lowering passes, and ordinary Python data structures that remain easy to print and test;
- from **CUTLASS**: an explicit target contract and a static backend descriptor that separates operation selection from the C++/CUDA runtime executing it.

It does not introduce MLIR dialect machinery, a general tensor compiler, or a second schedule graph.

## Pipeline

```text
A2AGemmProblem / GraphBuilder
              │
              ▼
┌──────────────────────────────┐
│ Frontend                     │  Python API or edmir frontend
│ checks immediately knowable │
│ shape and dtype constraints  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Dataflow IR                  │  TensorType, Value, Node, Module
│ pure tensor semantics        │  all_to_all, gemm
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Graph passes                 │  canonicalize, verify
│ typed SSA and pattern rules  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Pattern lowering             │  match All2All -> GEMM
│ derive A2AGemmProblem        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Schedule search/lowering     │  sequential and pipelined candidates
│ profile + topology + options │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Program IR                   │  Buffer, Op
│ baseline + fused reference   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Dependency lowering         │  queue + RAW/WAR/WAW
│ verifier / simulator         │  one semantic implementation
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Backend                      │  JSON or C++ ProgramView
│ target-specific artifact     │  shared runtime ABI
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ C++ runtime                  │  validate / allocate / execute
│ CUDA + NCCL                  │  host baseline or Device API reference
└──────────────────────────────┘
```

## Layer contracts

### 1. Frontend

Files: `compiler/frontend.py`, `compiler/dataflow.py`

The frontend turns a user-facing problem or builder calls into typed SSA-like dataflow. It knows tensor shape, dtype, layout and per-rank distribution. It does not know chunks, staging buffers, streams, NCCL algorithms or kernel tiles.

Current public entry points:

```python
module = build_a2a_gemm_module(problem)

builder = GraphBuilder("example", world_size=4)
x = builder.input("x", TensorType((256, 4096), dtype="float16"))
w = builder.input(
    "w",
    TensorType((4096, 4096), dtype="float16", distribution="replicated"),
)
y = builder.all_to_all(x, rows_per_peer=64)
z = builder.gemm(y, w)
builder.output(z)
module = builder.build()
```

### 2. Dataflow IR

Schema: `edmir.dataflow.v1`

The high-level IR has four concepts:

- `TensorType`: local shape, dtype, layout and distribution;
- `Value`: a typed SSA value;
- `Node`: a pure operation with input/output value IDs and attributes;
- `DataflowModule`: ordered nodes, module inputs/outputs and world size.

Only `all_to_all` and `gemm` exist in v0.1. Adding an operation requires type rules, verification, a lowering rule and tests; adding an enum value alone is not support.

### 3. Passes

File: `compiler/passes.py`

`PassManager` currently runs:

1. `CanonicalizeDataflowPass`: fills explicit peer order, transpose and epilogue attributes;
2. `VerifyDataflowPass`: checks SSA definition order, op arity, shape/dtype rules and module outputs.

Passes return a new immutable module and a `PassTrace`. There is no hidden global graph or mutable compiler context.

### 4. Pattern lowering

File: `compiler/lowering.py`

`match_a2a_gemm()` is intentionally strict. It accepts exactly one `all_to_all` whose result is the GEMM lhs, a frontend weight input, and the GEMM result as the module output. It derives the existing `A2AGemmProblem` from types and attributes.

This is the semantic boundary between tensor dataflow and schedule search. Unsupported graphs fail here with a concrete reason instead of silently falling back to an unrelated implementation.

### 5. Program IR and schedule lowering

Files: `compiler/model.py`, `compiler/schedule.py`, `compiler/compile.py`

`Program(Buffer, Op)` is the only low-level schedule IR. It represents physical staging and execution order:

- `transfer`: one source/destination communication tile;
- `gemm`: one destination-local tile GEMM;
- `barrier`: readiness token used by the sequential baseline.
- `fused_transfer_gemm`: a staging-free, device-initiated source-tile read and
  destination-local GEMM. It occupies source/destination communication and
  destination compute resources as one indivisible Program op.

The candidate generator varies strategy, chunk rows and staging buffer count.
The calibrated simulator ranks legal baseline candidates. The fused strategy
has formal lowering, verification, a guarded executable correctness reference,
and a CUTLASS tensor-core research kernel, but remains excluded from selection
until the measured workload-region gate is complete. Dataflow nodes never
contain these choices.

Fused-family evaluation is a separate offline layer. It may fit and validate a
direct-versus-split-K boundary from measured CSVs, but it cannot mutate
`CalibrationProfile` or enter normal candidate selection. The emitted report
records train/held-out direction accuracy, regret, host wins, and
`default_compiler_admission=false`. This keeps a reproducible path from
measurement to future admission without treating one A100 profile as an
architecture law.

### 6. Dependency and legality lowering

Files: `compiler/dependence.py`, `compiler/verify.py`, `compiler/simulate.py`

Program files do not store dependency edges. Dependencies are derived from:

- resource order;
- read-after-write readiness;
- write-after-read buffer reuse;
- write-after-write serialization.

The verifier, simulator and C++ backend all call the same implementation. This is the most important invariant in the compiler: schedule legality has one owner.

### 7. Target and profile

Files: `compiler/target.py`, `compiler/calibration.py`

`Target` owns CUDA target identity, architecture label, communication API,
GEMM provider, resolved kernel policy and the measured topology snapshot.
`host_nccl` selects the executable transfer/GEMM baseline; `nccl_device` is
required by the fused reference. `CalibrationProfile` owns replaceable timing information, including
optional per-`(src,dst,size)` NCCL curves.

Topology describes what is legal; the profile estimates cost. Neither is embedded into the frontend graph.

### 8. Backends

Directory: `compiler/backends/`

- `JsonBackend`: emits an `edmir.program.v1` for inspection and reproducibility.
- `CppManifestBackend`: emits the same JSON plus a C++20 header containing static buffer, op, access and fully derived dependency descriptors.

The generated header uses the shared structs in `runtime/include/edmir/runtime/program.hpp`; it does not define a second private ABI. The host runtime validates that descriptor and its dependency DAG. With `EDMIR_ENABLE_CUDA=ON`, the CUDA runtime allocates rank-local buffers, creates streams/events and dispatches grouped NCCL send/recv. GEMM is then sent to the statically selected cuBLAS or narrow SM80 CUTLASS provider without changing the Program. With the additional `EDMIR_ENABLE_NCCL_DEVICE=ON` gate, an SM80 FP16 fused op registers symmetric NCCL windows and launches either the `lsa_naive_f16` device reference or an LSA-wrapped CUTLASS kernel.

Calibration profiles carry the provider and kernel policy that produced their
GEMM samples. The pipeline validates that identity against the target before
lowering, preventing cuBLAS timings from silently scoring a CUTLASS artifact or
an explicit kernel from consuming measurements for another kernel.

## Why Python above and C++ below

Frontend construction, graph rewriting, finite schedule search and diagnostics change frequently and benefit from Python's small iteration loop. GPU resource ownership, CUDA Graph capture, NCCL communicator lifetime, stream/event execution and CUTLASS kernel launch belong in C++.

The split is therefore:

```text
Python: semantic IR -> legal selected Program -> static descriptor
C++:    ProgramView -> validation -> resource binding -> host or Device NCCL execution
```

Moving the optimizer to C++ now would add build friction without improving the research claim. Writing generated CUDA source strings now would hide runtime errors behind code generation. The static descriptor keeps the boundary testable.

## End-to-end CLI

```bash
python -m edmir frontend examples/a2a_gemm_problem.json \
  --out /tmp/a2a-gemm.dataflow.json

python -m edmir compile-graph /tmp/a2a-gemm.dataflow.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --target cuda --arch sm_89 --gemm-provider cublas --backend cpp \
  --out-dir /tmp/edmir-artifact
```

The output directory contains:

```text
program.json       selected low-level Program for audit/replay
target.json        architecture and measured topology contract
edmir_program.hpp  compile-time C++ runtime descriptor
```

## Extension rules

1. New frontend syntax must lower to existing Dataflow IR or add explicit typed IR semantics.
2. New dataflow ops must have verifier and pattern/lowering tests.
3. New scheduling choices belong in Program lowering, not frontend attributes.
4. New hardware facts belong in `Target`; measured timing belongs in `CalibrationProfile`.
5. Backends consume verified Programs and must not reinterpret dataflow semantics.
6. No backend may maintain a private dependency implementation.

## Runtime completion boundary

The minimal executable backend now exists in `runtime/`. cuBLAS remains the
general correctness baseline, while a guarded SM80 FP16 CUTLASS family proves
that the backend can substitute and statically select kernels without changing
IR or schedule. `auto` is resolved before artifact emission, so target JSON and
the C++ ABI contain a reproducible concrete kernel. Both paths are covered by
host compilation tests and 4x A100 numerical/latency experiments.

Provider-specific GEMM and runtime calibration now preserves the measured
oracle on all five A100 validation problems, including both held-out groups.
Optional NVTX3 op ranges and the Nsight trace validator additionally cover all
32 non-barrier ops and validate all 52 dependency-device constraints on both
held-out Programs without a violation. Kernel identity is preserved in target
JSON, C++ descriptors, runtime output, experiment archives, and CLI diagnostics.

The fused communication/compute milestone now includes executable kernels:
NCCL 2.28.9 was built in an isolated prefix, the official four-rank Device API
AllReduce passed, and Edmir's LSA remote-read plus naive FP16 GEMM matched its
CPU interpreter on 4x A100. A custom device wrapper now substitutes the LSA A
pointer into CUTLASS `GemmKernel::Params`, then reuses CUTLASS's predicated
tensor-core mainloop and epilogue inside the same kernel.

The first identical-shape measurements establish a workload boundary rather
than a universal speedup. Tiny `M8 K64 N64` lowered latency by 42.6% versus the
host-NCCL/CUTLASS Program, while `M64 K4096` was slower for every measured N.
For the standard 128-column tile, direct remote A reads are replayed
`ceil(N/128)` times; even replay one loses once A is 512 KiB. An admitted fused
model must therefore include remote A bytes and N-tile replay, and a future
persistent kernel must explicitly trade A reuse against CTA occupancy. A first
serial split-K=8 kernel already demonstrates the missing concurrency axis: at
`M64 K4096 N64` it turns the direct fused path from 82.3% slower into 11.8%
faster than the host baseline without changing remote bytes or replay.
The contract and experiment gate are in
[`FUSED_PROGRAM_CONTRACT.md`](FUSED_PROGRAM_CONTRACT.md).

See `runtime/README.md` for the server build and smoke runner.
