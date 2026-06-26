# MLSys 2027 Working Draft

## Working Title

**When Does Overlap Help in Multi-GPU LLM Serving?**

Subtitle candidates:

- A critical-path model for overlap-aware inference.
- From communication rooflines to serving latency.
- Explaining when topology, KV movement, collectives, and overlap become user-visible.

## Core Question

Multi-GPU LLM serving performance is not determined by FLOPS, NCCL bandwidth, KV cache hit rate, or whether an overlap switch is enabled.

The research question is:

> When does overlap reduce user-visible latency in multi-GPU LLM serving, and when does it only transform one hidden cost into another exposed cost?

This should be framed as a problem-understanding paper, not as a bigger serving system.

## Central Hypothesis

The missing variable between low-level communication measurements and end-to-end serving latency is **exposed data movement**:

```text
hardware fabric
  -> collective / KV-transfer cost
  -> workload phase and dependency graph
  -> overlap window
  -> exposed communication or exposed KV movement
  -> TTFT / ITL / P99
```

Communication matters only when it is on, or delays, the critical path.

Overlap is not a binary optimization. It is a transformation of the serving dependency graph:

```text
request / batch
  -> microbatch split
  -> stream scheduling
  -> collective placement
  -> metadata / KV / synchronization updates
  -> new critical path
```

An overlap policy helps only if:

```text
removed exposed communication
  > introduced split + stream + metadata + contention overhead
```

This is the new center of the draft. DBO-like mechanisms are case studies for the model, not the source of the paper's authority.

## Narrative Stance

This project should not stitch together a collection of recent serving papers.

Systems such as PD disaggregation, chunked prefill, Green Context / PD multiplexing, remote KV cache, and MoE communication should be treated as **samples of a shared trade-off space**, not as components to copy.

The paper's own explanatory axis is:

```text
LLM serving optimization is critical-path management across
compute, memory, communication, state locality, and runtime scheduling.
```

The goal is to build a small explanation system:

```text
raw bandwidth / FLOPS / logical KV hit rate
  -> are insufficient
exposed data movement on the serving critical path
  -> explains when an optimization affects TTFT / ITL / P99
```

## Current Research Position

Industrial practice already knows many of the individual facts:

- NVLink / NVSwitch / IB / RoCE / RDMA matter.
- All-to-all and KV movement can dominate at scale.
- NCCL microbenchmarks and total communication time do not directly predict serving latency.
- Prefill and decode have different resource profiles.
- KV cache hit rate is not enough if transfer and materialization are slow.
- DBO-like overlap is difficult because it changes the dependency graph, not just the collective timeline.

The gap is not "industry has never done this." The gap is:

> The public literature lacks a small, reproducible, phase-aware model that connects topology, collective cost, KV movement, overlap policy, runtime overhead, and user-visible serving metrics.

## Proposed Contributions

1. **A measurement method** that calibrates communication cost per hardware fabric, collective, placement, and message size.
2. **A trace decomposition** that separates compute, communication, overlap, runtime scheduling, and KV movement.
3. **An exposed-communication model** that predicts when communication affects TTFT / ITL / P99.
4. **An overlap-gain model** that predicts when DBO-like overlap helps, fails, or hurts.
5. **An effective KV-hit model** that distinguishes logical KV hits from useful hits.
6. **A decision procedure** for rejecting bad TP placement / TP degree / overlap / KV-transfer choices before full serving rollout.

## Non-Goals

- Do not build a new full serving engine.
- Do not compete with industrial-scale systems on cluster size.
- Do not claim universal fixed thresholds across all hardware.
- Do not start by writing custom NCCL / CUDA communication kernels.
- Do not make DeepEP, RDMA, KV storage, and TP planning one giant first paper.

## First Minimum Viable Paper Scope

Start narrow:

```text
4x4090 PCIe-only warmup
  + 8xH100/NVSwitch if rented
  + vLLM dense TP serving
  + NCCL roofline
  + trace-based exposed communication model
  + no-overlap / naive-overlap / critical-path-aware overlap comparison
```

Only add MoE / DeepEP / remote KV after the dense TP story is measurable and the model has a working calibration path.

## Files

- [`positioning.md`](positioning.md): narrative stance, defensive scope, and review-risk framing.
- [`scope.md`](scope.md): problem formulation, boundary, and paper taste.
- [`model.md`](model.md): variables, formulas, and measurement procedures.
- [`overlap.md`](overlap.md): DBO-like overlap as a critical-path question.
- [`claims-and-experiments.md`](claims-and-experiments.md): claims, falsification tests, and experiment matrix.
