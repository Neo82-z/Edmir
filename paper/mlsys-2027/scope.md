# Problem Scope

## The Thing To Explain

Multi-GPU LLM serving is often discussed as if adding GPUs gives more compute, and faster communication gives faster serving. In practice this is frequently false.

Examples of failure modes:

- A collective is slow in isolation but hidden by compute in serving.
- NCCL bandwidth looks healthy, but decode P99 is bad because small communication events are exposed.
- Total communication time is high, but overlap does not help because the exposed portion is small.
- DBO-like overlap hides some communication but adds microbatch, stream, metadata, or contention overhead.
- KV cache hit rate is high, but remote transfer or materialization makes the hit useless.
- TP degree increases theoretical compute capacity but increases synchronization and tail latency.
- A topology is fast for all GPUs together but poor for a partial GPU group.
- Storage / KV traffic competes with EP / PD / TP communication on the same fabric.

The paper should explain these as one family of problems: **data movement only helps or hurts end-to-end latency through dependency and exposure.**

## What Is New Enough?

The paper should not claim that networking or communication is newly important. That is industrial common sense and is present in many systems papers.

The intended contribution is narrower:

> A calibrated model and measurement workflow that says when communication or KV movement is exposed to user-visible LLM serving latency.

This turns engineering experience into a reproducible decision procedure.

## Core Boundary

Include:

- topology and placement;
- collective type and message size;
- prefill vs decode phase;
- compute / communication overlap;
- DBO-like microbatch overlap and its overhead;
- KV cache transfer vs recomputation;
- runtime scheduling and small-op overhead when visible in trace;
- TTFT, ITL / TPOT, throughput, P50/P90/P99.

Exclude from first scope:

- custom kernel design as the main contribution;
- full MoE runtime implementation;
- complete RDMA storage system;
- universal cluster-level scheduler;
- proof-only paper without calibration and decisions.

## Why This Can Be Academic Despite Being Industry-Known

Industrial systems often report deployment rules, benchmark tables, or large-scale speedups. A research paper can contribute by asking why:

- Why does a communication optimization transfer to serving latency in some regimes but not others?
- Why does overlap help in some regimes, do nothing in others, and sometimes hurt?
- Which part of communication is exposed rather than overlapped?
- Which workload phase is sensitive to which communication primitive?
- When does a logical cache hit become a useful hit?
- What small set of microbenchmarks and traces is sufficient to reject a bad configuration?

The paper should be about problem structure, not just a solution point.

## Working Claim Shape

Bad version:

> We implement a topology-aware planner and improve vLLM performance.

Better version:

> We show that total communication time, raw bandwidth, and whether overlap is enabled are poor predictors of serving latency. A small set of exposed data-movement metrics explains and predicts when topology, TP degree, overlap policy, and KV transfer choices affect TTFT / ITL / P99.

## Defensive Narrowing

The hardest risk is that "LLM serving infra" is too large.
Kernel quality, runtime scheduling, CUDA Graphs, memory allocation, NCCL, RDMA, KV cache placement, queueing, topology, NUMA, and workload mix can all be valid objections.

The paper should therefore defend a narrow claim:

> Given a topology, parallelism placement, serving phase, and request shape, when does raw communication or KV movement become exposed user-visible latency?

This is not a complete serving planner.
It is a calibrated explanation layer for one recurring bottleneck class.

The first scope should be dense tensor parallelism, topology, NCCL roofline, and trace-based exposed communication.
Controlled overlap ablations can be part of the first scope if the serving stack allows it.
KV movement should be introduced only after this dense-TP case is measurable.
MoE, DeepEP, RDMA, and remote KV storage are later extensions.

## Own Explanation System

The paper should not be a collage of recent systems.
Existing systems should be used as design points in a shared trade-off space:

- PD disaggregation trades interference isolation for KV migration and communication cost.
- Chunked prefill trades KV locality for chunk-size sensitivity and decode jitter.
- Green Context / PD multiplexing trades migration cost for shared-resource contention.
- Remote KV systems trade recomputation for lookup, transfer, and materialization latency.

The paper's own coordinate system is:

```text
compute / memory / communication / state locality / runtime scheduling
  -> critical-path exposure
  -> TTFT / ITL / P99
```

DBO-like overlap fits this coordinate system as:

```text
microbatch / stream / collective scheduling
  -> changed critical path
  -> removed exposed communication or introduced exposed overhead
```

## Hardware Reality

This project cannot compete with industrial-scale clusters. The hardware story should be honest:

- **4x4090 PCIe-only**: local methodology warmup and topology-sensitive case study.
- **8xH100 NVSwitch**: rented or borrowed validation point for modern single-node inference fabric.
- **2-node H100 / RDMA**: optional later validation for remote KV / inter-node communication, not required for the first model.

The paper should emphasize calibration rather than fixed thresholds:

```text
same model form
different alpha / beta / overlap / tail parameters per platform
```

## Research Taste Check

Each experiment should answer a "why" question, not merely show a faster number.

Good experiment:

> This configuration has worse NCCL bandwidth but unchanged ITL because communication is hidden by prefill compute.

Weak experiment:

> Configuration A is 12% faster than configuration B.

The evaluation should map the solution space into regimes:

- hidden communication;
- exposed communication;
- overlap-helpful;
- overlap-neutral;
- overlap-harmful;
- topology-sensitive placement;
- KV-transfer-useful;
- KV-transfer-dangerous;
- scheduler / small-op dominated;
- storage or network contention dominated.
