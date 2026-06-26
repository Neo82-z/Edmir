# Claims And Experiments

This matrix keeps the project problem-centered. Each claim should teach something about the structure of multi-GPU LLM serving.

## Claim Matrix

| ID | Claim | Why It Matters | Evidence Needed | Possible Counterexample |
|---|---|---|---|---|
| C1 | NCCL microbenchmarks alone do not predict serving latency. | Low-level bandwidth can be hidden by compute or irrelevant to the serving phase. | Compare NCCL roofline with vLLM TTFT / ITL under TP degree and workload-shape changes. | NCCL communication time predicts all latency changes without trace decomposition. |
| C2 | Exposed communication explains latency better than total communication time. | The critical path, not total work, determines user-visible latency. | Trace compute, communication, overlap, and exposed segments; correlate ECR with ITL/P99. | Total NCCL time and ECR have similar predictive power. |
| C3 | Overlap is conditional rather than universally beneficial. | DBO-like methods change the dependency graph and can add split, stream, metadata, or contention overhead. | Compare no-overlap, naive-overlap, and critical-path-aware overlap under matched workloads. | Enabling overlap always improves latency whenever total communication is large. |
| C4 | Topology changes active fabric, not just raw bandwidth. | GPU count alone is misleading; placement and group size activate different links. | Compare near/cross NUMA on 4x4090 and TP group sizes on H100/NVSwitch if available. | Placement differences vanish after controlling for message size and compute. |
| C5 | Prefill and decode have different communication sensitivity. | Serving decisions should be phase-aware. | Prefill-heavy, decode-heavy, and mixed workloads with same hardware and model. | One phase-independent model predicts all cases equally well. |
| C6 | Logical KV hit rate can overstate system benefit. | Remote KV or storage hits can hurt tail latency if transfer/materialization is exposed. | Compare recompute vs local hit vs remote hit / simulated transfer; measure useful hit margin. | Logical hit rate alone predicts TTFT/P99 across workloads. |
| C7 | Runtime/scheduler/small-op overhead can be a first-class bottleneck. | Not every bottleneck is FLOPS or NCCL. | Trace small event density, launch gaps, dependency fan-in, and idle gaps. | Removing/fusing/reordering small events does not change tail or utilization. |

## Experiment 0: Sanity Platform Snapshot

Goal:

> Make later numbers interpretable.

Collect:

```text
nvidia-smi
nvidia-smi topo -m
lscpu
numactl --hardware
driver / CUDA / NCCL / PyTorch / vLLM versions
cpuset / cgroup / container constraints
GPU process occupancy
```

Observation to record:

- Are GPUs idle and clean?
- Are CPUs pinned sanely?
- Is there NUMA asymmetry?
- Is the hardware suitable for the claim being tested?

## Experiment 1: Communication Roofline

Goal:

> Calibrate alpha / beta per collective and placement.

Run:

```text
all_reduce
all_gather
reduce_scatter
all_to_all
```

Vary:

```text
message size: 1KB -> 1GB
GPU group: near pair, cross pair, full group
optional: NCCL_P2P_DISABLE=1
optional: NCCL_ALGO / NCCL_PROTO
```

Report:

- latency curve;
- bus bandwidth curve;
- latency-bound / bandwidth-bound regions;
- topology penalty.

## Experiment 2: vLLM Dense TP Trace

Goal:

> Test whether communication is exposed in a real serving workload.

Workloads:

```text
decode-heavy: short input, long output
prefill-heavy: long input, short output
mixed: medium input and output
```

Vary:

```text
TP degree
concurrency
placement if possible
```

Collect:

- TTFT;
- ITL / TPOT;
- tokens/s;
- P50 / P90 / P99;
- Nsight or profiler trace;
- NCCL event durations and message sizes;
- compute/communication overlap.

Expected learning:

- Prefill has larger compute overlap windows.
- Decode is more sensitive to small exposed events.
- Some communication is hidden and should not be over-interpreted.

## Experiment 3: Held-Out Prediction

Goal:

> Make the model a decision tool rather than a post-hoc explanation.

Procedure:

1. Fit communication cost and overlap behavior from a subset of configs.
2. Predict high-risk and low-risk configs not used in calibration.
3. Run only those held-out configs.
4. Compare predicted direction and error.

Held-out examples:

- new concurrency;
- new prompt/output length;
- new TP degree;
- new placement;
- optional different model size.

Success criterion:

> The model predicts which configs have communication exposed enough to affect ITL/P99 better than raw NCCL bandwidth or GPU count.

## Experiment 4: Overlap / DBO-Like Case Study

Goal:

> Test whether overlap reduces exposed communication or merely introduces new exposed overhead.

Compare:

```text
no overlap
naive overlap
critical-path-aware overlap
```

For each configuration, measure:

- total collective time before and after overlap;
- exposed collective time before and after overlap;
- microbatch split overhead;
- stream/event synchronization overhead;
- attention / KV / request metadata overhead;
- new communication-compute or communication-memory contention;
- TTFT / ITL / P50 / P90 / P99.

Expected learning:

- Overlap helps when it removes exposed communication from the critical path.
- Overlap fails when the original communication was already hidden.
- Overlap can hurt when split / stream / metadata / contention overhead becomes exposed.

Important framing:

> DBO-like systems and open-source attempts are useful engineering context, but the paper evidence must come from controlled traces and ablations in this repo.

## Experiment 5: KV Usefulness Probe

Goal:

> Separate logical hit rate from useful hit rate.

Start small, even if remote KV is simulated.

Cases:

```text
recompute
local KV hit
remote KV hit or simulated transfer
remote KV hit under injected contention
```

Measure:

- lookup time;
- transfer time;
- materialization / H2D / layout time;
- TTFT impact;
- P99 transfer tail;
- effective hit rate.

Question:

> Does a cache hit actually reduce user-visible latency?

## Experiment 6: Interference Probe

Goal:

> Test whether KV / storage / PD / EP traffic competes with serving communication.

First version can be synthetic:

- Run NCCL collective alone.
- Run KV-like transfer alone.
- Run both together.
- Compare isolated vs co-run latency and bandwidth.

Record:

- shared NIC / PCIe / NUMA / storage path;
- P50/P90/P99 degradation;
- which flow gets hurt.

## What Not To Do Yet

- Do not build a full cluster scheduler first.
- Do not implement custom CUDA kernels before trace says the bottleneck is kernel-level.
- Do not make MoE / DeepEP the first required success path.
- Do not overfit to one rented machine without a calibration story.
- Do not report only average latency.

## Paper Evaluation Shape

The evaluation should be claim-driven:

```text
1. show raw communication regimes;
2. show serving traces where raw communication is insufficient;
3. show exposed communication improves explanation;
4. show overlap succeeds/fails according to exposed gain;
5. show held-out prediction;
6. optionally show KV useful-hit and interference cases.
```

If a result does not support, refine, or falsify a claim, it should probably not be in the main paper.
