# Reviewer Checklist For The Overlap Paper

This note compresses the reviewer-facing expectations for the working paper:

> **When Does Overlap Help in Multi-GPU LLM Serving?**

Core thesis:

```text
Overlap helps only when it removes data movement from the serving critical path
faster than it introduces split, stream, metadata, and contention overhead.
```

## One-Line Positioning

Do not frame the paper as:

> We implement DBO-like overlap in a serving stack and report speedups.

Frame it as:

> We turn communication overlap in multi-GPU LLM serving from an empirical optimization into a critical-path diagnosis and decision methodology.

The explanation chain should be:

```text
raw communication time
  -> total communication time
  -> exposed communication time
  -> critical-path change
  -> TTFT / ITL / P99
```

## What Reviewers Should See

### 1. A Reusable Abstraction

The paper should repeatedly return to:

```text
Communication matters only when it is exposed on the request critical path.
Overlap is useful only when it removes more exposed communication than the overhead it introduces.
```

The central diagnostic equation is:

```text
Delta_T_overlap ~=
  Delta_T_exposed_comm
- T_split
- T_stream
- T_metadata
- T_new_contention
```

Each term should be approximated by trace attribution, instrumentation, or controlled experiments.

### 2. A Measurement System, Not Screenshots

The paper needs a reusable analyzer rather than a few profiler screenshots.

Desired artifact:

```text
ECR Analyzer
  input:
    - Nsight Systems trace
    - torch profiler trace
    - Ascend profiler trace
    - serving runtime event log

  output:
    - phase-level critical path
    - total communication time
    - exposed communication time
    - ECR = T_exposed_comm / T_phase
    - split overhead
    - stream/event overhead
    - metadata overhead
    - contention indicators
    - predicted overlap direction
```

The key evidence should look like a trace-attribution table:

| Config | Total Comm | Exposed Comm | ECR | Split | Stream | Metadata | Contention | Predicted | Observed |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| TP4, long prefill | high | high | 0.31 | low | low | low | low | helps | helps |
| TP4, short decode | medium | low | 0.04 | medium | medium | low | low | neutral / hurts | hurts |
| TP8, weak placement | high | high | 0.42 | low | low | low | high | risky | mixed |

### 3. Controlled Interventions

Correlation between ECR and latency is not enough. The evaluation should actively change one variable while controlling others.

Intervention A: change communication cost with minimal compute change.

```text
same model
same prompt / output length
same concurrency
same TP degree
only change placement or fabric path
```

Desired conclusion:

```text
Topology penalty matters only when the affected communication is exposed.
```

Intervention B: change the overlap window.

```text
compute stream: controlled GEMM-like window W
comm stream: NCCL / HCCL collective
controls: message size, W, event placement, microbatch count, start offset
```

The probe should produce:

```text
overlap-helpful
overlap-neutral
overlap-harmful
```

Intervention C: compare overlap policies.

```text
P0: no overlap
P1: naive overlap
P2: critical-path-aware overlap
```

If the paper only compares no-overlap and DBO, it will look like a normal optimization paper.
The stronger result is showing that naive overlap can hurt, while critical-path-aware overlap avoids bad regimes.

### 4. Held-Out Prediction

The model should predict unseen configurations, at least directionally:

```text
overlap helps / neutral / hurts
communication-sensitive / insensitive
placement acceptable / risky
```

Compare predictors:

```text
A: raw NCCL / HCCL bandwidth
B: total communication time
C: ECR + overlap-gain model
```

The goal is not exact millisecond prediction first. The goal is better regime classification.

### 5. Negative Results As Evidence

The paper should include negative cases:

```text
communication time is large, but overlap does not help
overlap visually hides communication, but P99 gets worse
microbenchmark improves, but serving latency does not move
KV logical hit rate is high, but useful hit rate is low
topology microbenchmark differs, but serving metric is insensitive
```

These cases support the main claim:

```text
total work != critical-path work
```

### 6. One Deep Systems Case Study

The main paper should stay narrow:

```text
dense TP + topology + ECR + overlap model
```

Then add one deep case study, not many shallow ones.

Candidate case:

```text
vllm-ascend DBO / HCCL / AIC-AIV resource contention
```

The case study should explain:

```text
request split
attention metadata split
query_start_loc / seq_lens / position / rotary metadata
collective stream placement
event record / wait / yield dependencies
HCCL AIV / AI CPU mode impact
AIC / AIV resource limits and T_new_contention
```

The point is not only speedup. The point is:

```text
what exposed communication was removed
what overhead was introduced
why the policy helps in some regimes and fails in others
```

## What Reviewers Should Not See

Avoid:

- a paper that combines vLLM, DBO, DeepEP, MoE, remote KV, RDMA, Ascend, NCCL, HCCL, and scheduling without depth;
- only mean TTFT or QPS;
- profiler screenshots without trace attribution;
- open-source PRs used as paper evidence;
- "critical path" used as a slogan without graph construction details;
- speedup-only reporting with no neutral or harmful overlap regimes;
- a model that only explains after the fact and cannot predict held-out configurations.

## Critical-Path Specificity

If the paper claims critical-path analysis, it must define how the graph is built.

Approximate phase DAG:

```text
nodes:
  compute / collective / KV / runtime / sync / idle

edges:
  per-stream order
  explicit event record/wait
  collective completion to dependent compute
  request phase boundaries
  scheduler/runtime dependencies when visible
```

When edges are missing, use conservative exposure attribution:

```text
count communication as exposed only if it delays the next dependent event
or lies on the selected phase critical path
```

## Strong-Accept Shape

A strong version of the paper should convince reviewers of five things:

1. Raw bandwidth and total communication time are insufficient predictors.
2. The ECR / critical-path analyzer is reproducible.
3. Controlled interventions validate the mechanism.
4. Held-out prediction works better than raw bandwidth or total communication.
5. One system case study is deep enough to expose real implementation constraints.

## Next Artifact Plan

Most valuable next step:

```text
ECR Analyzer + controlled overlap probe + dense TP trace suite
```

Recommended repo structure:

```text
tools/ecr_analyzer/
  trace_schema.py
  parse_nsight.py
  parse_torch_profiler.py
  parse_ascend_profiler.py
  build_phase_dag.py
  critical_path.py
  exposed_comm.py
  overlap_gain.py
  report.py

benchmarks/overlap_probe/
  nccl_overlap_probe.cc
  hccl_overlap_probe.cc
  torch_overlap_probe.py

experiments/
  000_platform_snapshot/
  001_comm_roofline/
  002_dense_tp_trace/
  003_controlled_overlap/
  004_dbo_case_study/
```

