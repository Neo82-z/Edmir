# Roadmap

## Phase 0: Build The First Research Workbench

Goal: produce reproducible baseline slices, not final optimizations.

- Read tinygrad-notes and tensor puzzles.
- Reproduce primitive composition with small Python code.
- Run NCCL collectives on 4x4090.
- Record topology, commands, raw logs, and first plots.

Expected output:

- tinygrad primitive notes.
- NCCL all-reduce, all-gather, reduce-scatter, all-to-all baseline.
- One README per experiment slice.

## Phase 1: Communication Roofline

Goal: establish an empirical roofline for the current hardware.

- Compare near pairs and cross-NUMA pairs.
- Compare 2-GPU and 4-GPU collectives.
- Identify latency-bound and bandwidth-bound regions.
- Add Nsight Systems traces for selected sizes.

## Phase 2: vLLM Trace Alignment

Goal: connect communication primitives to real serving workloads.

- Run TP/PP/EP serving workloads.
- Identify collectives in prefill and decode.
- Compare observed collective time with NCCL microbench roofline.
- Decide whether bottlenecks are communication volume, topology, launch overhead, or scheduling.

## Phase 3: Small Optimization

Goal: implement one scoped optimization or decision policy.

Possible directions:

- Topology-aware parallelism selection.
- Message-size-aware collective diagnosis.
- MoE all-to-all profiling and backend selection.
- KV cache transfer and recomputation boundary analysis.
