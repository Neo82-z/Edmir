# MLSys 2027 Working Draft

## Working Title

**When Does Communication Matter in Multi-GPU LLM Serving?**

Subtitle candidates:

- A critical-path model for communication-aware inference.
- From communication rooflines to serving latency.
- Explaining when topology, KV movement, and collectives become user-visible.

## Core Question

Multi-GPU LLM serving performance is not determined by FLOPS, NCCL bandwidth, or KV cache hit rate alone.

The research question is:

> When does data movement become exposed on the critical path of LLM serving, and when does optimizing communication fail to improve TTFT / ITL / P99?

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
- NCCL microbenchmarks do not directly predict serving latency.
- Prefill and decode have different resource profiles.
- KV cache hit rate is not enough if transfer and materialization are slow.

The gap is not "industry has never done this." The gap is:

> The public literature lacks a small, reproducible, phase-aware model that connects topology, collective cost, KV movement, overlap, and user-visible serving metrics.

## Proposed Contributions

1. **A measurement method** that calibrates communication cost per hardware fabric, collective, placement, and message size.
2. **A trace decomposition** that separates compute, communication, overlap, runtime scheduling, and KV movement.
3. **An exposed-communication model** that predicts when communication affects TTFT / ITL / P99.
4. **An effective KV-hit model** that distinguishes logical KV hits from useful hits.
5. **A decision procedure** for rejecting bad TP placement / TP degree / KV-transfer choices before full serving rollout.

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
```

Only add MoE / DeepEP / remote KV after the dense TP story is measurable and the model has a working calibration path.

## Files

- [`positioning.md`](positioning.md): narrative stance, defensive scope, and review-risk framing.
- [`scope.md`](scope.md): problem formulation, boundary, and paper taste.
- [`model.md`](model.md): variables, formulas, and measurement procedures.
- [`claims-and-experiments.md`](claims-and-experiments.md): claims, falsification tests, and experiment matrix.
