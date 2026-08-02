# Communication-Compute Fusion: Prior Art and Claim Boundary

Last verified: 2026-08-01.

Purpose: constrain Edmir v0.1 before implementation. This is a prior-art map, not a claim that every listed system solves the same problem.

## Terminology

Use these labels consistently:

| Label | Meaning |
|---|---|
| operator overlap | communication and computation remain separate kernels/operators and overlap through streams or scheduling |
| wait-logic fusion | a compute kernel contains tile-ready waits/signals, while data movement is performed by host, copy engine, or another kernel |
| in-kernel fusion | device-side communication and computation execute as cooperating work inside one kernel |
| semantic fusion | communication layout/routing and the consumer computation are jointly transformed, rather than merely co-scheduled |

Do not infer one category from another.

## P0: Directly adjacent work

| Work | What it actually demonstrates | Boundary for Edmir |
|---|---|---|
| [CoCoNet / Breaking the Computation and Communication Abstraction Barrier](https://www.microsoft.com/en-us/research/publication/breaking-the-computation-and-communication-abstraction-barrier-in-distributed-machine-learning-workloads/) (ASPLOS 2022) | DSL and compiler where communication and computation are first-class; supports fusion and overlap transformations | Edmir cannot claim that a joint communication-computation IR is new |
| [FLUX](https://arxiv.org/abs/2406.06858) ([code](https://github.com/bytedance/flux)) | CUTLASS-based fused/overlapped `AllGather → GEMM` and `GEMM → ReduceScatter`; ReduceScatter is decomposed into an All2All communication component plus local reduction, and the All2All write path may be fused into the GEMM epilogue | This is not a general compiler for arbitrary `All2All → GEMM`, but it invalidates broad “first communication/GEMM fusion” claims |
| [TileLink](https://arxiv.org/abs/2503.20313) | Compiler with tile-centric signal/data primitives, static and dynamic rank mapping, memory-consistency lowering, and examples including `GEMM → ReduceScatter`, `AllGather → MoE`, and `AllGather KV → attention` | Closest abstraction-level competitor to a tile schedule IR and legality-aware lowering |
| [Triton-distributed](https://arxiv.org/abs/2504.19442) ([code](https://github.com/ByteDance-Seed/Triton-distributed)) | Extends Triton with OpenSHMEM-like primitives, symmetric memory and signals; demonstrates overlapped `AllGather → GEMM`/`GEMM → ReduceScatter`; implements EP All2All dispatch/combine as communication kernels | Edmir cannot claim that Python/Triton-level distributed primitives or compiler-assisted overlap are new; its paper distinguishes All2All-only kernels from fused All2All consumers |
| [AutoOverlap](https://arxiv.org/abs/2601.20595) | Source-to-source compiler that combines a local Triton kernel with a chunk schedule and aligns computation with chunk availability inside a fused kernel | Direct competitor to automatic chunk/tile overlap generation |
| [Comet](https://openreview.net/pdf?id=fGgQS5VW09) | Fine-grained communication-computation overlap for MoE using dependency analysis and task rescheduling | Dynamic MoE overlap is not an empty design space and remains outside v0.1 |
| [Lancet](https://arxiv.org/abs/2404.19429) | Whole-graph computation-communication overlap for MoE training | Edmir v0.1 must not make whole-graph or MoE claims |

## P0: Communication substrate and current NVIDIA direction

| Reference | Why it matters |
|---|---|
| [NCCL collective API: `ncclAlltoAll`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclalltoall) | Canonical host-launched sequential baseline in NCCL 2.30+. Older NCCL versions need an equivalent grouped send/recv implementation. |
| [NCCL Device-Initiated Communication documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) | Canonical semantics and requirements for symmetric windows, LSA, Multimem, GIN, barriers and version compatibility |
| [GPU-Initiated Networking for NCCL](https://arxiv.org/abs/2511.15076) | Architecture and motivation for the NCCL Device API/GIN stack |
| [NCCL EP](https://arxiv.org/abs/2603.13606) | Unified MoE dispatch/combine API built on NCCL Device API; low-latency design adapted from DeepEP and high-throughput design adapted from Hybrid-EP |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) and [DeepEP V2 announcement](https://github.com/deepseek-ai/DeepEP/pull/605) | Real EP implementation; V2 moves its main networking backend from NVSHMEM to NCCL GIN and requires Hopper/SM90 for its current main path |
| [NCCL Q2 2026 roadmap](https://github.com/NVIDIA/nccl/issues/2090) | Tracks CuTe DSL integration, cost-model changes, JIT support and the still-under-consideration SM-initiated Copy Engine path |
| [CUTLASS / CuTe DSL](https://github.com/NVIDIA/cutlass) | NVIDIA's low-level compute and Python DSL stack for high-performance GEMM and custom kernels |

The 2026-05-23 Hangzhou vLLM Meetup talk by Wen Ke and Qiang Yicheng, *Symmetric Memory + NCCL Device APIs: A New Path Towards Multi-GPU Fused Kernels*, showed a progression from a standard NCCL All2All followed by a CuTe DSL kernel toward a future SM-initiated Copy Engine design on 8×B200. The public [event agenda](https://www.linkedin.com/posts/jiaju-zhang-06868314_vllm-meetup-hangzhou-accelerating-inference-activity-7462646858117435393-LXH1) confirms the session, but the photographed slides are context rather than a citable paper. In particular, a slide labeled “Ver. 6 in the future” must not be treated as a released implementation.

## P1: Collective compiler and synthesis lineage

| Work | Main abstraction |
|---|---|
| [SCCL: Synthesizing Optimal Collective Algorithms](https://arxiv.org/abs/2008.08708) | SMT synthesis of latency/bandwidth Pareto-optimal collectives for a topology |
| [TACCL](https://arxiv.org/abs/2111.04867) | Topology profile plus designer-provided communication sketches |
| [GC3](https://www.microsoft.com/en-us/research/publication/gc3-an-optimizing-compiler-for-gpu-collective-communication/) | Data-oriented DSL and compiler for programmable AllReduce, AllToAll and application-specific collectives |
| [MSCCLang](https://doi.org/10.1145/3575693.3575724) | Collective DSL, compiler/runtime and pre-execution correctness checking |
| [MSCCL++](https://arxiv.org/abs/2504.09014) | Low-level portable communication primitives, higher-level DSL and synchronization/consistency abstractions |
| [PCCL](https://arxiv.org/abs/2606.07019) | Process-group-aware, scalable synthesis for arbitrary collective patterns |

Edmir v0.1 does not synthesize a new collective algorithm. Its fixed All2All is an input to a communication-consumer schedule.

## What is and is not already fused

The defensible answer is:

- Special cases of communication/GEMM fusion already exist.
- `AllGather → GEMM` and `GEMM → ReduceScatter` are well covered by FLUX, TileLink and Triton-distributed.
- FLUX explicitly fuses the All2All communication substep of ReduceScatter into a GEMM epilogue in supported paths.
- Triton-distributed implements All2All dispatch/combine, but presents it as communication-only EP kernels rather than a generic fused `All2All → GEMM` compiler.
- DeepEP and NCCL EP optimize MoE dispatch/combine; expert GEMM remains a distinct semantic component in their public interfaces.
- The NVIDIA meetup shows that direct `All2All → GEMM` in-kernel fusion is an active target, not that an arbitrary automatic compiler is a finished public product.

Therefore the v0.1 example is valid, but the broad fusion idea is not a novelty claim.

## Candidate claim under investigation

The only current candidate is deliberately narrow:

> Edmir investigates legality-checked lowering of a static communication-consumer schedule to NCCL Device API mechanisms.

This candidate survives only if it can be distinguished from TileLink's memory-consistency compilation, AutoOverlap's chunk transformation, Triton-distributed's primitives/lowering, and MSCCL++'s consistency abstractions. Until then it is a question, not a contribution.

## Reading order

1. TileLink: extract tile primitives, mapping functions and memory-consistency rules.
2. Triton-distributed: extract symmetric memory, signal semantics, task model and All2All boundary.
3. AutoOverlap: extract compiler input, chunk schedule ownership and transformations.
4. FLUX: extract the exact division between data transfer, signals, GEMM prologue/epilogue and autotuning.
5. NCCL Device API: map LSA/barrier/window semantics to possible lowering operations.
6. MSCCLang/MSCCL++: extract verifier obligations and runtime contracts.

For each work, record: input IR, schedule unit, backend, synchronization model, automated decisions, generated artifact, hardware, and unsupported cases.
