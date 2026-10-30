# EDM-IR

## Offline routed-benchmark preparation

The [offline readiness guide](docs/ROUTED_OFFLINE_READINESS.md) connects the
[D7 routing contract](docs/ROUTED_BENCHMARK_CONTRACT.md) to a small BF16/FP32
CPU reference, strict timing-record validation, paired-block statistics, and
fresh bounded TLC checks. It requires no GPU. Synthetic fixtures are kept out
of measured paper exports; a passing rehearsal is **not** GPU readiness.

```bash
cd EDM-IR
python3 tools/prepare_routed_benchmark.py --out-dir build/routed-rehearsal
# Add --jar /absolute/path/tla2tools.jar for fresh TLC checks.
```

The output directory must be new. No server access, model download, paid
resource provisioning, or modifications to historical measurements occur.

## Compiler overview

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

Correctness and performance claims are deliberately separated. The
[`spec/ASSURANCE_MODEL.md`](spec/ASSURANCE_MODEL.md) boundary gives host,
pipelined, direct-LSA, split-K, and grouped-N schedules one executable tensor
relation, while [`compiler/assurance.py`](edmir/compiler/assurance.py) checks
small Programs against that relation. Its target-bound evidence gate requires
matching problem, environment, code, protocol, and raw-source hashes; failed
numerics, insufficient samples, or touching confidence intervals select host.
The accompanying [`spec/RISK_REGISTER.md`](spec/RISK_REGISTER.md) says which
new concurrency mechanisms are modeled and which mature library internals are
kept behind contracts. The compiler-owned slot lifecycle is additionally
model-checked at two finite TLA+/TLC bounds, with deliberate early-read,
early-reuse, and incomplete-release mutations producing counterexamples. This
does not prove the CUDA runtime or the NCCL/CUTLASS internals; those remain
separate trace, numerical, and library-contract obligations. Reproduction
instructions and hashed TLC output are under [`spec/tla/`](spec/tla/).
Mutable producer-generated remote inputs have a second bounded epoch protocol.
Its four mutation witnesses cover early acquire, early consume, incomplete
retirement, and premature overwrite. The executable IR currently admits only
the versioned immutable-prelaunch contract; declaring the mutable mode fails
verification until its synchronization events become Program-visible.

Fused kernels use a separate, deliberately conservative selection path. A
versioned profile binds a measured shape envelope and direct/split-K rule to an
exact GPU, topology, CUDA, NCCL, and CUTLASS fingerprint. Selection requires an
explicit opt-in and refuses every extrapolation; the ordinary host NCCL +
cuBLAS Program is the successful fallback, not an error path.

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
fused-runtime software/hardware gate is captured under
`experiments/2026-08-26-a100-runtime-capabilities`. The first executable NCCL
Device API reference kernel and its numerical evidence are under
`experiments/2026-08-26-a100-nccl-device-fused-reference`. The executable
CUTLASS fusion boundary, including the negative direct-access sweep and the
split-K concurrency result, is under
`experiments/2026-08-26-a100-nccl-device-cutlass-fusion`. Its fixed-replay
K/remote-byte follow-up is under
`experiments/2026-08-26-a100-fused-k-boundary`; its complete held-out M/N grid
and guarded evaluation are under
`experiments/2026-08-26-a100-fused-heldout`. The
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

The Program IR has four op kinds:

- `transfer`: move one All2All tile from `peer_rank` to `rank` on both ranks'
  communication resources;
- `gemm`: consume a received staging tile on the destination compute queue;
- `barrier`: create a readiness token for the sequential baseline.
- `fused_transfer_gemm`: directly read a source tile and destination weight,
  write the destination output, and occupy both communication endpoints plus
  destination compute. It has no staging buffer. Guarded SM80 FP16 NCCL
  Device API implementations provide a scalar correctness reference and a
  CUTLASS tensor-core research path.

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

For four-RTX-4090 portability debugging, use the non-destructive protocol in
[`experiments/2026-08-31-4090-debug/`](experiments/2026-08-31-4090-debug/).
It keeps host NCCL + cuBLAS validation separate from the SM80 fused research
path, refuses occupied GPUs, and labels every timing as diagnostic rather than
paper evidence.

Generate the guarded fused reference artifact:

```bash
python -m edmir lower-fused /tmp/a2a-gemm.dataflow.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16 --arch sm_80 \
  --gemm-provider cutlass --backend cpp \
  --out-dir /tmp/edmir-fused-reference
```

This command emits a verified `fused` Program and a
`communication_api=nccl_device` target. A runtime built with NCCL 2.28+,
`EDMIR_ENABLE_NCCL_DEVICE=ON`, and optionally `EDMIR_ENABLE_CUTLASS=ON`
executes the selected fused kernel. The calibrated simulator still rejects the
fused op because decomposed transfer/GEMM costs do not describe an in-kernel
implementation. `lower-fused` is therefore an explicit force-lowering/debug
entry point, not an automatic selector. See
[`docs/FUSED_PROGRAM_CONTRACT.md`](docs/FUSED_PROGRAM_CONTRACT.md).

Run the measured, profile-gated selector locally with:

```bash
python -m edmir compile-fused examples/a100_fused_problem.json \
  --topology examples/a100_fused_topology.json \
  --profile examples/a100_fused_selection_profile.json \
  --environment examples/a100_fused_environment.json \
  --allow-experimental-fused --backend cpp \
  --out-dir /tmp/edmir-fused-selected
```

The retained example now emits the cuBLAS host fallback even with explicit
opt-in: its speed evidence used per-source CUTLASS, while this compiler's
fallback uses per-source cuBLAS. `reference_host_baseline_id` must match the
actual fallback before a profile can admit fused execution. Missing baseline
identity also rejects admission; old measurement files are not relabelled.
Changing any environment field, removing the NV12 link labels, or using a shape
outside the explicit 45-point envelope also emits a host fallback.
`selection.json` records the reference and fallback host identities, chosen family, every
candidate, the guard result, exact reason, resolved target, and static traffic
features. The supplied fingerprint is a retained experiment fixture; it must
be replaced by the observed server environment on a new run.

The output schema is `edmir.program.v1`. Compiler-generated `edmir.plan.v1`
files from the first prototype remain readable and are rebuilt as Programs;
their serialized edge lists are intentionally discarded.

The bundled profile and topology are synthetic fixtures, not experimental
results. Use `python -m edmir probe --out topology.json` and
`python -m edmir probe-runtime --out runtime-capabilities.json` on the target
machine, then attach measured pairwise NCCL curves before drawing any
performance conclusion:

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

Fused kernel-family evidence is evaluated separately so it cannot silently
enter the host-NCCL simulator:

```bash
python -m edmir evaluate-fused \
  --train /path/to/train/results.csv \
  --heldout /path/to/heldout/*/results.csv \
  --out /tmp/fused-boundary-evaluation.json
```

The report fits the observed direct/split-K boundary and computes held-out
direction accuracy and selection regret. Its schema explicitly leaves default
compiler admission disabled. `compiler/fused_selection.py` consumes that
evidence through an environment-fingerprinted profile and still requires
explicit experimental admission at each compile.

Evaluate the replacement two-stage selector on complete declared sweep groups:

```bash
python tools/evaluate_two_stage_selector.py \
  --world-size 4 \
  --out /tmp/two-stage-selector.json \
  /path/to/sweep-*/results.csv
```

The first model predicts host admission separately from the fused `(g,q)`
family. Its interpretable feature surface contains FLOPs, unique remote and
local bytes, `rho_N`, CTA supply `Gamma`, `K/q`, and split reduction work. The
tool leaves each complete input directory out in turn and runs no-admission,
no-Gamma, no-rho, no-reduction, and K-only ablations. Failing the registered
direction/regret/host-regression gates produces evidence but cannot open the
default compiler path.

The next selector/baseline experiment design and retrospective interaction
comparison are recorded in
[`experiments/2026-10-19-selector-baseline-design`](../experiments/2026-10-19-selector-baseline-design/README.md).
`tools/evaluate_selector_interactions.py` compares fixed feature alternatives
on archived data without GPU execution or compiler admission. Evaluation v2
counts every selected latency above host as a point-estimate host regression;
the previous host-oracle-only count is retained as `host_admission_errors`.

The audited offline dataset and experimental relative-cost predictor are in
[`experiments/2026-10-20-offline-predictor`](../experiments/2026-10-20-offline-predictor/README.md).
`tools/prepare_selector_dataset.py` retains timing scope, environment, source
hashes, and historical roles; standalone GEMM/NCCL records stay separate.
`tools/evaluate_relative_selector.py` compares the original models, a fixed
schedule chosen on training workloads, and within-workload cost ratios.
This retrospective comparison does not enable compiler admission.

Code-path mapping, input-contract fixes, regression checks, and a feature-level
trace of the largest remaining prediction error are documented in
[`experiments/2026-10-20-predictor-debug`](../experiments/2026-10-20-predictor-debug/README.md).
`tools/debug_relative_selector.py` verifies historical decisions before producing
that trace. Passing these checks does not establish prediction accuracy on new workloads.

The subsequent [predictor contract audit](../experiments/2026-10-20-predictor-contracts/README.md)
adds explicit host identities, paired grouped-N support by M/q, a diagnostic
three-ratio factorization, and frozen predictions for new A100 measurements.
The factorized fit performs worse on historical data and is not promoted.
`tools/independent_selector_validation.py` validates and scores complete new
runs; it does not implement or execute the missing merged-host collectors.

Plan the grouped-N kernel space:

```bash
python tools/plan_grouped_n_sweep.py \
  --problem examples/a2a_gemm_problem.json \
  --topology examples/full_p2p_topology.json \
  --kernels sm80_64x128x32,sm80_64x128x32_splitk8 \
  --n-tiles-per-cta 1,2,4 \
  --out /tmp/grouped-n-schedule-space.json
```

The report exposes total N tiles, effective remote-A replay, split-K slices,
and estimated CTA supply for every point. The source backend implements the
first grouped point, `n_tiles_per_cta=2`, as a doubled CUTLASS N-threadblock;
larger factors remain analysis-only and are rejected by C++ emission. The new
kernel is experimental, not profile-admitted: it must compile and pass numeric
validation on the pinned A100 stack before any timing claim or automatic
selection.

## Code map

```text
compiler/model.py       Buffer, Op, Program, problem/profile/topology schemas
compiler/calibration.py NCCL and exact-shape GEMM parsers/calibration
compiler/evaluation.py train/heldout runtime fit and selection metrics
compiler/fused_evaluation.py guarded fused-family boundary and held-out metrics
compiler/fused_selection.py exact-envelope admission, selection, and fallback
compiler/two_stage_selection.py host admission, fused schedule fit, group holdout
compiler/host_baseline.py explicit host implementation/partition/timer identities
compiler/factorized_selection.py experimental three-ratio fits and grouped-N support
compiler/additive_latency_selection.py bounded nested fit of host/fused absolute latency
compiler/dataflow.py    typed SSA-like tensor Dataflow IR
compiler/frontend.py    small graph builder and A2AGemm frontend
compiler/passes.py      canonicalization, verification, pass trace
compiler/lowering.py    strict All2All->GEMM match and fused-reference lowering
compiler/schedule.py    direct ProgramBuilder and finite candidate construction
compiler/schedule_space.py static direct/split-K x grouped-N traffic/CTA frontier
compiler/dependence.py  the only queue and buffer dependency implementation
compiler/verify.py      All2All->GEMM semantic checks
compiler/assurance.py   executable semantics and target-bound evidence gate
compiler/slot_protocol.py host-staging correspondence to the TLA+ lifecycle
compiler/remote_publication.py publication trace correspondence and closed gate
compiler/routed_contract.py versioned routed rows, canonical order, and D7 ledger
compiler/simulate.py    ready-time/resource interpreter and EventGraph emission
compiler/compile.py     candidate enumeration, selection, and fallback
compiler/pipeline.py    end-to-end compiler orchestration
compiler/target.py      CUDA topology, communication API, and GEMM contract
compiler/backends/      JSON and compileable C++20 manifest backends
runtime/                ProgramView validator and host-/device-NCCL CUDA executor
analysis.py             critical path and counterfactual exposure on EventGraph
```

Runtime build and generated-program smoke instructions are in
[`runtime/README.md`](runtime/README.md).

The controlled split-K revision, retrospective results, four explicit host
baselines, and prepared GPU pilot are documented in
[`2026-10-20-splitk-local-revision`](../experiments/2026-10-20-splitk-local-revision/README.md).
The revision did not pass its replacement criteria. The merged host source
and collector are available, but GPU build, correctness, and timing remain
unverified; they do not change the compiler's fallback.

The latest [additive latency selector repair](../experiments/2026-10-21-additive-selector-repair/README.md)
passes the existing local selection gates, including all four priority shapes,
using nested training-only parameter selection. It estimates host and fused
milliseconds separately, keeps grouped-N within paired M64 support, and retains
the CUTLASS baseline identity. New independent GPU validation and compiler
admission remain pending. Use `tools/predict_additive_schedule.py` with the frozen
model to inspect decisions without loading measurements or fitting.

## v0.1 boundary

- static world size and shapes;
- row-major equal-sized All2All blocks;
- cyclic peer order;
- one destination-local GEMM per received row tile;
- one communication, compute, and control resource model per rank;
- sequential fallback when P2P connectivity or predicted gain is absent.
- fused selection requires the exact versioned A100/NV12 measured envelope
  and evidence against the same host implementation as the fallback.

Dynamic MoE routing, ragged tokens, Grouped GEMM, multi-node networking, true
collective rendezvous, and GPU code generation remain outside this version.
The next routed extension is frozen in
[`docs/ROUTED_BENCHMARK_CONTRACT.md`](docs/ROUTED_BENCHMARK_CONTRACT.md); its
schema and analytical D7 checks are implemented, but its CUDA execution path
is not yet claimed.
