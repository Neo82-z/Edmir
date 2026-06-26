# Research Positioning

## What This Paper Is Trying To Be

The paper should feel like an MLSys paper because it explains a recurring systems trade-off, not because it combines many fashionable systems.

The intended claim is:

> LLM serving needs a critical-path-aware data movement model because raw bandwidth, FLOPS, total communication time, and logical KV hit rate cannot predict user-visible latency by themselves.

This gives the work its own coordinate system.
Existing systems can be discussed as design points inside this coordinate system, but they should not become the contribution.

## The Explanatory Axis

The central abstraction is:

```text
total data movement
  != exposed data movement
  != user-visible latency
```

The paper should ask whether communication, KV movement, or runtime overhead is exposed on the request critical path.
If it is hidden by compute or scheduler slack, improving it may not improve TTFT, ITL, or P99.
If it delays a dependent operation, even a small communication event can become user-visible.

## How To Use Existing Systems Without Stitching Them Together

Recent systems should be treated as examples of trade-offs:

- **PD disaggregation**: separates prefill and decode interference, but introduces KV migration and communication risk.
- **Chunked prefill**: keeps KV local, but chunk size trades prefill progress against decode ITL jitter.
- **Green Context / PD multiplexing**: partitions compute resources while keeping KV locality, but still shares HBM, L2, memory capacity, and runtime paths.
- **Remote KV systems**: increase logical cache capacity and reuse, but only help if lookup, transfer, and materialization beat recomputation on the critical path.
- **MoE / DeepEP-style communication**: makes all-to-all and dispatch/combine traffic central, but should enter the first paper only if it sharpens the exposed-data-movement model.

These are not modules to assemble.
They are evidence that many serving designs are negotiating the same axes:

```text
compute partition
memory locality
communication path
KV state placement
runtime scheduling
tail-latency exposure
```

## Defensive Scope

The paper should not claim:

- to explain all LLM infra performance problems;
- to build a full serving planner;
- to predict every TTFT / ITL / P99 value exactly;
- to solve KV cache, RDMA, MoE, and scheduling in one system;
- to compete with industrial clusters by scale.

The paper can claim:

- to model when raw communication cost becomes user-visible serving latency;
- to predict whether a topology or TP placement choice is likely to matter;
- to show when an NCCL improvement transfers to serving latency and when it does not;
- to separate logical KV hits from useful KV hits;
- to provide calibrated decision boundaries rather than universal thresholds.

## First Fight To Win

The first paper should fight one contained battle:

```text
dense TP
  + topology
  + NCCL roofline
  + vLLM trace
  + exposed communication ratio
```

KV movement should be the second layer, not the thing that makes the first paper impossible to finish.
MoE, DeepEP, RDMA, and remote KV storage are extensions after the dense TP model is measurable.

## Reviewer Risks

**"This is obvious."**

Answer: raw bandwidth, total communication time, and logical cache hit rate are common but insufficient predictors.
The contribution is a calibrated exposure model that predicts when those metrics transfer to serving latency.

**"This is just profiling."**

Answer: the goal is not a one-off profile.
The model should predict across placement, TP degree, message size, phase, and request shape.

**"The hardware scale is too small."**

Answer: the paper validates a mechanism, not a leaderboard.
Small PCIe systems can stress topology; an 8-GPU H100/NVSwitch run can validate the modern single-node case.

**"Existing systems already know this."**

Answer: existing systems are design points.
This paper contributes a public explanation framework and measurement workflow.

**"The contribution is not engineering enough."**

Answer: the model must output decisions:

- whether a TP group should cross a weak fabric;
- whether a communication optimization is worth doing;
- whether to fetch KV or recompute it;
- whether prefill and decode should co-locate or disaggregate;
- whether latency is communication-exposed or dominated by another bottleneck.

