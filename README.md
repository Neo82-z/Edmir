# LLM Serving Research Notes

This repository is a research log for multi-GPU LLM serving, with a focus on communication bottleneck diagnosis and optimization.

## Main Thread

**Communication bottleneck diagnosis and optimization for multi-GPU LLM inference.**

Allowed side quests:

- tinygrad: tensor IR, primitive composition, and collective semantics.
- CUDA C++: kernels, memory hierarchy, streams, and profiling.
- DeepGEMM / CuTe / CUTLASS: compute-side granularity and GEMM/MoE kernels.
- NCCL / NVSHMEM / RDMA: collective communication, topology, and transport.
- vLLM / Megatron / DeepSpeed: real serving workloads and parallelism strategies.

Every side quest should eventually answer:

> How does this help explain or improve multi-GPU LLM serving communication?

## Repository Layout

```text
papers/        Paper notes grouped by area.
experiments/   Reproducible experiment slices, raw logs, tables, and figures.
ideas/         Research questions, possible paper angles, and design sketches.
templates/     Markdown templates for paper notes and experiment records.
```

## Current First Slice

- Understand tinygrad-notes and tinygrad tensor puzzles through small reproductions.
- Run NCCL baseline on 4x RTX 4090 PCIe dual-NUMA topology.
- Save commands, raw logs, topology, and first observations.

## Working Rule

One experiment should answer one question. Keep raw data, commands, environment, and observations together.
