# Paper：Tutel

## Citation

- Authors：Microsoft Research 等
- Venue / Year：MLSys 2023
- Title：Tutel: Adaptive Mixture-of-Experts at Scale
- Link：https://arxiv.org/abs/2206.03382

## 一句话总结

Tutel 是一个面向 large-scale MoE 的系统，关注 MoE routing、expert parallelism、all-to-all communication 和 adaptive execution。

## 问题

MoE 通过稀疏激活 experts 提升模型容量，但引入 token dispatch、expert load imbalance 和 all-to-all communication bottleneck。

## 关键想法

构建 adaptive MoE runtime，根据 workload 和系统状态优化 routing、parallelism 和 communication。

## System / Algorithm

需要重点理解：

- expert parallelism。
- token routing / dispatch。
- all-to-all communication。
- load balancing 与 adaptive parallelism。

## Experiments

- Hardware：large-scale GPU cluster。
- Workloads：MoE training / inference。
- Baselines：existing MoE systems。
- Metrics：throughput、scaling efficiency、communication overhead。

## 复现计划

短期只复现概念：

- 在 vLLM / Megatron / DeepSpeed 中观察 EP 相关配置。
- 用 NCCL / PyTorch `all_to_all` baseline 建立通信边界。
- 在 H100/Hopper 上加入 DeepEP V2 dispatch/combine 对照。
- 构造 skewed token distribution，观察 MoE dispatch 的尾部效应。

## 与我的研究主线的关系

EP / MoE 是 all-to-all 最自然的 LLM serving 场景之一。Tutel 帮助把 communication bottleneck 从抽象 collective 拉回到 token routing 和 load imbalance。

## Open Questions

- all-to-all 慢是 bandwidth 问题，还是 token imbalance 导致慢 rank 问题？
- expert placement 是否应该 topology-aware？
- PCIe-only 机器上 EP 的收益边界在哪里？这个问题只作为热身，不外推到 DeepEP V2。
