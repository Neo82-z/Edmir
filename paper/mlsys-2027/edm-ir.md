# EDM-IR: Exposed Data-Movement Intermediate Representation

Status: working design note.

Scope: trace-derived analysis IR for multi-GPU LLM serving.

Non-goal: not a serving engine, not a compiler backend, not a full overlap scheduler.

## Core Idea

The current paper should center on a missing representation:

```text
raw serving trace
  -> serving-level uops
  -> exposed data-movement graph
  -> critical-path attribution
  -> ECR / overlap-gain analysis
  -> visualization and diagnosis
```

The point is not to implement DBO first.
The point is to make communication exposure measurable and reproducible.

Working question:

```text
When does overlap remove data movement from the request critical path,
and when does it only introduce new exposed overhead?
```

## Why IR

Raw profiler timelines are precise but too low-level for serving-level reasoning.
They contain CUDA / NPU kernels, collective kernels, runtime launches, stream waits, metadata operations, and idle gaps, but they do not directly answer:

```text
Did this communication lie on the request critical path?
Was it hidden behind useful compute?
Did overlap shorten the critical path?
Did overlap introduce split / stream / metadata / contention overhead?
```

EDM-IR borrows the taste of tinygrad-style uops, but the direction is different:

```text
tinygrad:
  tensor program -> uops -> schedule / lowering / codegen

EDM-IR:
  serving trace -> serving uops -> dependency graph -> diagnosis / prediction
```

EDM-IR represents observed execution. It is not used to generate kernels or schedules.

## UOp Classes

The first version should use a compact set of serving-level uops:

```text
phase.begin
phase.end
request.enqueue
request.finish
batch.form
batch.commit

ubatch.split
ubatch.merge
padding.adjust

metadata.build
metadata.split
metadata.copy
metadata.materialize

kv.lookup
kv.transfer
kv.materialize
kv.prefetch
kv.evict

compute.qkv
compute.attn
compute.mlp
compute.moe_expert
compute.norm
compute.logits
compute.other

comm.all_reduce
comm.all_gather
comm.reduce_scatter
comm.all_to_all
comm.broadcast
comm.p2p
comm.kv_transfer

event.record
event.wait
stream.yield.compute_to_comm
stream.yield.comm_to_compute
stream.synchronize

runtime.launch
runtime.launch_gap
runtime.scheduler_step
runtime.cpu_wait
runtime.graph_capture
runtime.graph_replay

contention.region
resource.limit.enter
resource.limit.exit
```

The purpose is to put compute, communication, metadata, KV, synchronization, runtime, and contention into one analyzable logic instead of treating every library call as an isolated fact.

## Graph Edges

EDM-IR becomes useful when dependencies are explicit.

Minimum edge types:

```text
stream_order
data_dependency
event_dependency
collective_dependency
scheduler_dependency
kv_readiness
phase_boundary
resource_contention
```

Conservative rule:

```text
A communication uop is exposed only if it lies on the reconstructed critical path
or delays a dependent uop on that path.
```

This avoids claiming that every long collective is user-visible.

## Analysis Passes

EDM-IR should support a small pass pipeline:

```text
Pass 0: raw trace import
Pass 1: event canonicalization
Pass 2: phase segmentation
Pass 3: uop construction
Pass 4: dependency linking
Pass 5: critical-path analysis
Pass 6: exposure attribution
Pass 7: overlap-gain analysis
Pass 8: report and visualization
```

Core metrics:

```text
T_phase
T_total_comm
T_exposed_comm
ECR = T_exposed_comm / T_phase
T_metadata
T_stream
T_new_contention
predicted_overlap_direction
```

Overlap-gain estimate:

```text
Delta_T_phase ~=
  Delta_T_exposed_comm
- T_split
- T_stream
- T_metadata
- T_new_contention
```

Classification target:

```text
overlap_helpful
overlap_neutral
overlap_harmful
```

## Minimal Artifact

The first artifact should be tiny:

```text
synthetic_hidden_comm.json
synthetic_exposed_comm.json
synthetic_overlap_harmful.json
```

Expected output:

```text
uop dump
critical path
T_total_comm
T_exposed_comm
ECR
overlap direction
DOT graph or text graph
```

Recommended layout:

```text
tools/edm_ir/
  schema.py
  uops.py
  graph.py
  critical_path.py
  exposure.py
  overlap_gain.py
  report.py
  visualize.py

examples/edm_ir/
  synthetic_hidden_comm.json
  synthetic_exposed_comm.json
  synthetic_overlap_harmful.json
```

Only after this demo works should real profiler parsers be added.

## Paper Positioning

Suggested contribution wording:

> We introduce EDM-IR, an IR-like, trace-derived representation for diagnosing exposed data movement in multi-GPU LLM serving. EDM-IR canonicalizes profiler events into serving-level uops, reconstructs phase dependency graphs, and supports critical-path attribution of communication, KV movement, metadata, runtime overhead, and contention.

Defensive wording:

```text
EDM-IR is not a compiler IR for code generation.
It is an analysis IR for serving traces.
```

```text
DBO is not the source of authority for the paper.
DBO is one graph transformation that EDM-IR can analyze.
```

## Stopping Point

For now, stop expanding the system scope.

The first concrete milestone is:

```text
synthetic JSON -> EDM Graph -> critical path -> ECR -> report
```

Not:

```text
full DBO scheduler
DeepEP integration
remote KV system
vLLM rewrite
custom communication kernels
```

