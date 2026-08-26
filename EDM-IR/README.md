# EDM-IR

EDM-IR deliberately keeps compilation and analysis different:

```text
Frontend -> Dataflow IR -> passes/lowering -> Program(Buffer, Op)
                                              |
                                              +-> JSON/C++ backend
                                              +-> derived dependencies -> EventGraph -> ECR/EDMR
profiler trace -------------------------------------------------------> EventGraph -> ECR/EDMR
```

The compiler IR is an ordered, executable Tile/communication program. It does
not store dependency edges. An operation declares only its queue plus the
physical buffers it reads and writes; queue order and RAW/WAR/WAW hazards are
derived in one place. The resulting graph is consumed by both the legality
checker and the deterministic cost simulator.

The current artifact generates a compileable C++20 `ProgramView`. The optional
C++ runtime executes it with CUDA streams/events, grouped NCCL send/recv, and a
statically selected cuBLAS or CUTLASS GEMM provider. Both provider paths have
been compiled, numerically validated, and benchmarked on 4x A100 SXM; the
retained baselines are under `experiments/2026-08-25-runpod-4xa100-sxm` and the
controlled provider substitution is under
`experiments/2026-08-26-a100-cutlass-provider`; measured static kernel selection
is under `experiments/2026-08-26-a100-cutlass-kernel-family`; provider-specific
cost-model validation is under
`experiments/2026-08-26-a100-cutlass-cost-model`; measured dependency-order
validation is under `experiments/2026-08-26-a100-nsys-trace-validation`. The
complete layer contract is documented in
[`docs/COMPILER_ARCHITECTURE.md`](docs/COMPILER_ARCHITECTURE.md).

## Minimal compiler model

The core types are:

```python
Buffer(name, rank, role, size_bytes)
Op(id, kind, rank, queue, reads, writes, tile, rows, peer_rank)
Program(problem, strategy, chunk_rows, buffer_count, buffers, ops)
```

[`examples/minimal_program.py`](examples/minimal_program.py) constructs a
complete two-rank program directly and prints every automatically derived
dependency.

The first version has only three executable op kinds:

- `transfer`: move one All2All tile from `peer_rank` to `rank` on both ranks'
  communication resources;
- `gemm`: consume a received staging tile on the destination compute queue;
- `barrier`: create a readiness token for the sequential baseline.

`ProgramBuilder` constructs those ops directly. `derive_dependencies()` then
applies five rules:

1. operations sharing a hardware resource keep program order;
2. the last writer precedes every reader (RAW/data readiness);
3. prior readers precede a buffer overwrite (WAR/buffer reuse);
4. consecutive writers are ordered (WAW);
5. a transfer occupies both source and destination communication resources.

All derived buffer edges carry release/acquire semantics. There is no second
edge list for schedule generation, verification, and simulation to disagree on.

## CPU-only quick start

```bash
python -m unittest discover -s tests -p '*_test.py' -v

python -m edmir compile examples/a2a_gemm_problem.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --out /tmp/edmir-program.json

python -m edmir verify /tmp/edmir-program.json
python -m edmir inspect /tmp/edmir-program.json --dependencies
python -m edmir simulate /tmp/edmir-program.json \
  --profile examples/synthetic_profile.json
```

Run the complete frontend-to-backend path:

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

The output schema is `edmir.program.v1`. Compiler-generated `edmir.plan.v1`
files from the first prototype remain readable and are rebuilt as Programs;
their serialized edge lists are intentionally discarded.

The bundled profile and topology are synthetic fixtures, not experimental
results. Use `python -m edmir probe --out topology.json` on the target machine
and attach measured pairwise NCCL curves before drawing any performance
conclusion:

```bash
python -m edmir fit-nccl \
  --base-profile examples/synthetic_profile.json \
  --logs /path/to/nccl/logs \
  --out /tmp/measured-comm-profile.json
```

The fitter reads only `pair_<src>_<dst>_default_sendrecv.txt` logs and installs
the measured curve in both directions. Add exact-runtime GEMM samples and fit
the host/runtime residual separately:

```bash
python -m edmir fit-gemm \
  --base-profile /tmp/measured-comm-profile.json \
  --log /path/to/gemm_measurements.jsonl \
  --gemm-provider cublas --gemm-kernel library_default \
  --out /tmp/device-profile.json

python -m edmir fit-copy \
  --base-profile /tmp/device-profile.json \
  --log /path/to/local_copy_measurements.jsonl \
  --out /tmp/device-profile-with-copy.json

python -m edmir fit-runtime \
  --base-profile /tmp/device-profile-with-copy.json \
  --measurements /path/to/runtime_measurements.csv \
  --out-profile /tmp/calibrated-profile.json \
  --out-report /tmp/model-evaluation.json
```

`fit-runtime` learns non-negative coefficients only from rows labeled `train`.
Rows labeled `heldout` are used for MAPE, direction-accuracy, and
schedule-regret reporting without refitting. A profile records the GEMM
provider and kernel policy that produced its samples; the full compiler rejects
a profile/target mismatch instead of silently comparing unlike runtimes. The
complete cuBLAS and CUTLASS 4x A100 experiments are retained in
`experiments/2026-08-26-runpod-4xa100-cost-model` and
`experiments/2026-08-26-a100-cutlass-cost-model`.

## Code map

```text
compiler/model.py       Buffer, Op, Program, problem/profile/topology schemas
compiler/calibration.py NCCL and exact-shape GEMM parsers/calibration
compiler/evaluation.py train/heldout runtime fit and selection metrics
compiler/dataflow.py    typed SSA-like tensor Dataflow IR
compiler/frontend.py    small graph builder and A2AGemm frontend
compiler/passes.py      canonicalization, verification, pass trace
compiler/lowering.py    strict All2All->GEMM pattern lowering
compiler/schedule.py    direct ProgramBuilder and finite candidate construction
compiler/dependence.py  the only queue and buffer dependency implementation
compiler/verify.py      All2All->GEMM semantic checks
compiler/simulate.py    ready-time/resource interpreter and EventGraph emission
compiler/compile.py     candidate enumeration, selection, and fallback
compiler/pipeline.py    end-to-end compiler orchestration
compiler/target.py      CUDA topology, architecture, and GEMM-provider contract
compiler/backends/      JSON and compileable C++20 manifest backends
runtime/                ProgramView validator and CUDA/NCCL/cuBLAS/CUTLASS executor
analysis.py             critical path and counterfactual exposure on EventGraph
```

Runtime build and generated-program smoke instructions are in
[`runtime/README.md`](runtime/README.md).

## v0.1 boundary

- static world size and shapes;
- row-major equal-sized All2All blocks;
- cyclic peer order;
- one destination-local GEMM per received row tile;
- one communication, compute, and control resource model per rank;
- sequential fallback when P2P connectivity or predicted gain is absent.

Dynamic MoE routing, ragged tokens, Grouped GEMM, multi-node networking, true
collective rendezvous, and GPU code generation remain outside this version.
