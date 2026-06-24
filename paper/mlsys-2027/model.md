# Exposed Data-Movement Model

This file records the first model skeleton. The goal is not to invent a large new theory. The goal is to combine known ideas from communication models, roofline reasoning, critical paths, and LLM serving phases into a measurable decision model.

## Request And Phase Decomposition

For a request:

```text
T_request =
  T_queue
+ T_prefill
+ N_decode * T_decode
+ T_tail_effects
```

For one serving phase:

```text
T_phase =
  T_compute
+ T_exposed_comm
+ T_kv
+ T_runtime
+ T_contention
```

Where:

- `T_compute`: visible compute on the critical path.
- `T_exposed_comm`: communication not hidden by compute or other useful work.
- `T_kv`: KV lookup, transfer, materialization, or recomputation.
- `T_runtime`: launch, scheduling, graph/runtime overhead, small-op overhead.
- `T_contention`: queueing, shared NIC/switch/storage contention, OS/runtime interference.

## Communication Cost

Per collective / placement / message size:

```text
T_comm(c, m, g, topo) = alpha(c, g, topo) + beta(c, g, topo) * m + epsilon
```

Definitions:

- `c`: collective or communication type, e.g. all-reduce, all-gather, reduce-scatter, all-to-all, p2p, KV transfer.
- `m`: message size.
- `g`: GPU group / placement.
- `topo`: hardware fabric, e.g. PCIe-only, NVSwitch, IB / RoCE.
- `alpha`: latency and launch/protocol cost.
- `beta`: inverse effective bandwidth.
- `epsilon`: noise, tail, contention, implementation effects.

This should be calibrated from `nccl-tests`, P2P tests, and serving traces.

## Exposed Communication

The core metric:

```text
T_exposed_comm = max(0, T_comm - T_overlap_window)
```

Trace-based definition:

```text
T_exposed_comm = duration of communication events that lie on, or extend, the critical path of the serving phase
```

Ratio:

```text
ECR = T_exposed_comm / T_phase
```

Interpretation:

- `ECR ~= 0`: communication optimizations are unlikely to improve end-to-end latency.
- high `ECR`: communication, placement, or overlap changes may affect TTFT / ITL / P99.

The exact boundary should be calibrated per platform and workload. Do not hard-code universal thresholds.

## Topology Penalty

For a candidate placement:

```text
P_topo(c, m, g) =
  T_comm(c, m, g) / min_g T_comm(c, m, g)
```

This captures whether a GPU group activates a poor portion of the fabric.

This is useful for cases where fewer GPUs can be slower because they use less aggregate fabric bandwidth or a worse peer-to-peer path.

## Effective Bandwidth

```text
BW_eff(c, m, g) = bytes_moved(c, m) / T_comm(c, m, g)
```

Bandwidth efficiency:

```text
eta_bw = BW_eff / BW_peak_or_best_observed
```

This should be plotted against message size to find latency-bound and bandwidth-bound regimes.

## KV Cache Usefulness

Logical cache hit rate is not enough. A KV hit is useful only if using it beats recomputation and does not violate tail-latency budget.

```text
T_kv_hit =
  T_lookup
+ T_transfer
+ T_materialize
```

```text
KV_useful_margin = T_recompute - T_kv_hit
```

Useful hit condition:

```text
KV_useful_margin > 0
```

Tail condition:

```text
P99(T_kv_hit) < SLA_budget_for_phase
```

Effective hit rate:

```text
effective_hit_rate =
  useful_hits / reusable_prefixes
```

This distinguishes:

- cache exists but is too far away;
- remote hit helps TTFT;
- remote hit hurts P99;
- recomputation is cheaper than transfer.

## Traffic Interference

KV transfer, EP all-to-all, PD handoff, NCCL collectives, storage backup, and runtime control traffic can share bottleneck links.

Simple model:

```text
T_flow_observed = T_flow_isolated * contention_factor
```

Or:

```text
BW_effective = BW_isolated / contention_factor
```

The model does not need to solve full network queueing first. It should at least detect when co-running flows share the same NIC, switch queue, PCIe path, or storage backend.

## Decision Rule

For a candidate serving configuration:

```text
score(config) =
  predicted_TTFT_or_ITL
+ tail_risk_penalty
+ memory_or_capacity_penalty
```

Reject or warn if:

- `ECR` is high for decode;
- `P_topo` is high for selected GPU group;
- KV logical hit rate is high but `KV_useful_margin <= 0`;
- KV / EP / PD traffic shares a bottleneck and tail risk is high;
- runtime overhead dominates because the graph has many small dependent events.

## How To Measure

Minimum evidence per platform:

1. `nvidia-smi topo -m`.
2. NCCL roofline for all-reduce, all-gather, reduce-scatter, all-to-all.
3. P2P bandwidth / latency if available.
4. vLLM trace with prefill-heavy, decode-heavy, and mixed workloads.
5. TTFT / ITL / throughput / P50 / P90 / P99.
6. Trace decomposition into compute, communication, overlap, KV movement, runtime overhead.

## What Would Falsify The Model?

- Total communication time predicts latency as well as exposed communication.
- ECR is high but changing communication cost does not affect latency.
- ECR is low but communication changes strongly affect latency.
- KV useful margin predicts benefit poorly across held-out workloads.
- Topology penalty does not correlate with observed serving differences when other variables are controlled.
