# Roadmap

## Phase 0：建立第一版研究工作台

目标：产出可复现的 baseline slice，而不是一开始就追最终优化。

- 阅读 tinygrad-notes 和 tensor puzzles。
- 用小规模 Python 代码复现 primitive composition。
- 在多机多卡或者单机多卡上跑 NCCL collectives。
- 记录 topology、commands、raw logs 和第一版图表。

预期产出：

- tinygrad primitive 笔记。
- NCCL all-reduce、all-gather、reduce-scatter、all-to-all baseline。
- 每个 experiment slice 都有一份 README。

## Phase 1：Communication Roofline

目标：为当前硬件建立 empirical roofline。

- 对比 near pair 和 cross-NUMA pair。
- 对比 2-GPU 和 4-GPU collectives。
- 找出 latency-bound 和 bandwidth-bound 区间。
- 为关键 message size 增加 Nsight Systems trace。

## Phase 2：对齐 vLLM Trace

目标：把 communication primitive 和真实 serving workload 连接起来。

- 跑 TP / PP / EP serving workload。
- 识别 prefill 和 decode 阶段的 collectives。
- 将 vLLM 中观测到的 collective time 与 NCCL microbench roofline 对比。
- 判断 bottleneck 来自 communication volume、topology、launch overhead，还是 scheduling。

## Phase 3：小范围优化

目标：实现一个边界清晰的优化或决策策略。

可能方向：

- topology-aware parallelism selection。
- message-size-aware collective diagnosis。
- MoE all-to-all profiling 和 backend selection。
- KV cache transfer 与 recomputation 的边界分析。
