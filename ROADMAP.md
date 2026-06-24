# Roadmap

## MLSys 主线：Exposed Data Movement Model

目标：不是做一个更大的 serving engine，而是解释并预测 multi-GPU LLM serving 中 data movement 什么时候会影响用户可见延迟。

核心问题：

> NCCL bandwidth、GPU FLOPS、KV cache hit rate 为什么不能直接预测 TTFT / ITL / P99？缺失变量是不是 exposed communication / exposed KV movement？

短期产出：

- 完成 [`paper/mlsys-2027/model.md`](paper/mlsys-2027/model.md) 的第一版变量定义。
- 完成 [`paper/mlsys-2027/claims-and-experiments.md`](paper/mlsys-2027/claims-and-experiments.md) 中 C1-C3 的最小实验证据。
- 将 NCCL roofline 和 vLLM trace 对齐，形成第一版 ECR（Exposed Communication Ratio）。
- 明确 KV cache 支线是否进入第一篇 paper，还是作为后续扩展。

## Phase 0：建立第一版研究工作台

目标：产出可复现的 baseline slice，并明确不同硬件平台能回答什么问题。

- 阅读 tinygrad-notes 和 tensor puzzles。
- 用小规模 Python 代码复现 primitive composition。
- 在 4x4090 上只做 NCCL / PyTorch distributed / PCIe topology 热身，不把它作为 DeepEP V2 结论平台。
- 在 H100/Hopper 环境可用后，再进入 DeepEP V2 主实验。
- 记录 topology、commands、raw logs 和第一版图表。

预期产出：

- tinygrad primitive 笔记。
- 4x4090 NCCL all-reduce、all-gather、reduce-scatter、all-to-all warmup baseline。
- H100 DeepEP V2 vs NCCL/PyTorch `all_to_all` 的实验计划。
- 每个 experiment slice 都有一份 README。

## Phase 1：Communication Roofline

目标：为当前硬件建立 empirical roofline，但只把它当作后续实验的测量工具，不把 4090 结果外推到 DeepEP V2。

- 对比 near pair 和 cross-NUMA pair。
- 对比 2-GPU 和 4-GPU collectives。
- 找出 latency-bound 和 bandwidth-bound 区间。
- 为关键 message size 增加 Nsight Systems trace。

## Phase 2：DeepEP / NCCL 代码阅读

目标：先不依赖机器，把 DeepEP 和 NCCL / PyTorch reference path 的差异讲清楚。

重点文件：

- `deep_ep/utils/refs.py`：NCCL / PyTorch reference dispatch / combine。
- `deep_ep/buffers/elastic.py`：DeepEP V2 Python API。
- `csrc/kernels/backend/nccl.cu`：NCCL Gin、device communicator、symmetric window。
- `csrc/elastic/buffer.hpp`：dispatch / combine host path。
- `tests/elastic/test_ep.py`：实验入口。

## Phase 3：H100 DeepEP V2 主实验

目标：把 MoE EP 的 communication primitive 和真实 serving / dispatch workload 连接起来。

- 对比 DeepEP V2、NCCL / PyTorch `all_to_all`、必要时加入 NVSHMEM legacy path。
- 分析 DeepEP 省掉了哪些通用 all-to-all 的额外步骤。
- 测 routing metadata / cached handle 对 decoding 的收益。
- 测 `num_sms` 减少后，GEMM overlap 是否更好。
- 测 QP 数、hybrid/direct path、RDMA/NVLink 域对 tail latency 的影响。
- 构造 token imbalance，判断瓶颈在网络、copy epilogue、CPU sync 还是 expert GEMM 等待。

## Phase 4：对齐 vLLM Trace

目标：把 communication primitive 和真实 serving workload 连接起来。

- 跑 TP / PP / EP serving workload。
- 识别 prefill 和 decode 阶段的 collectives。
- 将 vLLM 中观测到的 collective time 与 NCCL microbench roofline 对比。
- 判断 bottleneck 来自 communication volume、topology、launch overhead、scheduler、小算子密度、KV movement，还是 exposed critical-path waiting。
- 输出 trace decomposition：compute、communication、overlap、runtime、KV movement。

## Phase 5：小范围优化

目标：实现一个边界清晰的优化或决策策略。优化不是第一目标；第一目标是让模型能提前拒绝坏配置。

可能方向：

- topology-aware parallelism selection。
- message-size-aware collective diagnosis。
- MoE all-to-all profiling 和 backend selection。
- DeepEP-aware routing / layout / overlap policy。
- KV cache transfer 与 recomputation 的边界分析。
- runtime / graph small-op overhead 诊断。
