# MLSys 2027 Working Draft

## Working Title

**EDM-IR: Exposed Data-Movement Analysis for Multi-GPU LLM Serving**

Subtitle candidates:

- A trace-derived IR for overlap-aware serving analysis.
- From communication rooflines to serving latency.
- Explaining when topology, KV movement, collectives, and overlap become user-visible.

## Core Question

Multi-GPU LLM serving performance is not determined by FLOPS, NCCL bandwidth, KV cache hit rate, or whether an overlap switch is enabled.
The missing object is a representation that connects low-level trace events to request-level critical paths.

The research question is:

> How can raw multi-GPU serving traces be turned into a representation that explains when data movement is exposed to user-visible latency?

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

The representation-level center is **EDM-IR**:

```text
raw serving trace
  -> serving-level uops
  -> exposed data-movement graph
  -> critical-path attribution
  -> ECR / overlap-gain analysis
```

EDM-IR borrows the taste of tinygrad-style uops, but it is not a compiler IR.
It represents observed serving execution for diagnosis and prediction.

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

> The public literature lacks a small, reproducible, phase-aware representation that connects topology, collective cost, KV movement, overlap policy, runtime overhead, and user-visible serving metrics.

## Proposed Contributions

1. **EDM-IR**, a trace-derived analysis IR that turns serving traces into uops and dependency graphs.
2. **Critical-path exposure analysis**, which computes exposed communication and ECR from EDM graphs.
3. **An overlap-gain model** that predicts when DBO-like overlap helps, fails, or hurts.
4. **A minimal analyzer artifact** that starts from synthetic traces and later supports real profiler frontends.
5. **A decision procedure** for rejecting bad TP placement / TP degree / overlap / KV-transfer choices before full serving rollout.

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
  + EDM-IR analyzer
  + trace-based exposed communication analysis
  + controlled overlap probe
```

Only add MoE / DeepEP / remote KV after the dense TP story is measurable and the model has a working calibration path.

## Files

- [`positioning.md`](positioning.md): narrative stance, defensive scope, and review-risk framing.
- [`scope.md`](scope.md): problem formulation, boundary, and paper taste.
- [`model.md`](model.md): variables, formulas, and measurement procedures.
- [`edm-ir.md`](edm-ir.md): trace-derived IR design note.
- [`overlap.md`](overlap.md): DBO-like overlap as a critical-path question.
- [`claims-and-experiments.md`](claims-and-experiments.md): claims, falsification tests, and experiment matrix.
- [`reviewer-checklist.md`](reviewer-checklist.md): compressed reviewer expectations and artifact plan.
- [`progress-log.md`](progress-log.md): public-safe milestone notes.
