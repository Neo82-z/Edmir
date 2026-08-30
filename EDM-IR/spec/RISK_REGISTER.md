# Assurance risk register

Risk determines modeling depth. New concurrent boundaries receive explicit
semantics and checks; mature library internals remain abstract contracts.

| ID | Failure | Consequence | Priority | Control |
|---|---|---|---|---|
| R1 | Source/destination/chunk offset mismatch | Silent wrong expert output | Critical | `SEM-1/2`, executable reference, rank-permutation tests |
| R2 | Consumer observes an unpublished remote or staging tile | Nondeterministic corruption | Critical | `MEM-1`, dependency validation, future TLA+ model |
| R3 | Slot is reused before the previous consumer finishes | Rare data race under load | Critical | `MEM-2`, buffer-reuse hazards, trace validation, mutation test |
| R4 | Split-K omits or duplicates a partition | Wrong output with plausible magnitude | Critical | `RED-1/2`, partition property tests, numerical ledger |
| R5 | Barrier/semaphore cycle | Multi-rank deadlock | Critical | Dependency-cycle check, future TLA+ liveness check |
| R6 | Grouped-N aliases output columns or reuses mutable A | Wrong result or stale read | High | `REUSE-1`, immutable-input contract, grouped equivalence tests |
| R7 | Evidence from another driver/topology/code revision is reused | Performance regression or unsupported execution | Critical | `ENV-1`, SHA-256 binding, deterministic host fallback |
| R8 | Near-tie noise admits a slower fused family | SLO regression | High | Confidence-interval gate, minimum sample count, safety margin |
| R9 | Uniform synthetic M hides real routing imbalance | Invalid inference claim | High | Router trace, fixed trace replay, balanced/imbalanced controls |
| R10 | P4/P8 comparison changes node and software with world size | Confounded scale conclusion | High | Same-node four-GPU subset and eight-GPU controlled run |
| R11 | Integer-only or easy numerical fixtures hide reduction error | False correctness confidence | High | Non-integer/adversarial inputs and independent seeds |
| R12 | Performance model is tuned on held-out outcomes | Reward-hacked selector claim | Critical | Frozen protocol hash, append-only raw data, new version after any rule change |

## Abstracted components

| Component | Required contract | Why it is abstracted |
|---|---|---|
| CUTLASS/cuBLAS | Legal GEMM shape computes its documented operation | Mature provider; EDMIR verifies shape and invocation, not MMA generation |
| NCCL transport | Registered communication operation obeys the selected API contract | Transport protocol is outside the compiler contribution |
| vLLM/FlashInfer | Produces a versioned request/router/KV trace | Used to generate realistic workloads, not reimplemented |
| Router | Supplies token-to-expert assignments and declared ordering | Model quality and top-k implementation are outside scope |
| CUDA runtime | Provides documented stream, event, and memory-order primitives | EDMIR verifies the dependency protocol built on them |

An abstraction is not an assumption that failures are impossible. Version,
capability, shape, and invocation checks remain inside EDMIR; only the mature
component's internal algorithm is excluded.
