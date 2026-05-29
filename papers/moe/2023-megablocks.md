# Paper：MegaBlocks

## Citation

- Authors：MegaBlocks team
- Venue / Year：MLSys 2023
- Title：MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- Link：https://arxiv.org/abs/2211.15841

## 一句话总结

MegaBlocks 使用 block-sparse computation 改善 MoE 中由于 token imbalance 和 padding 带来的计算浪费。

## 问题

MoE 中不同 expert 收到的 token 数不均衡。传统实现常通过 padding 统一形状，导致大量无效计算和内存浪费。

## 关键想法

用 block-sparse representation 和 block-sparse kernels 表达 MoE computation，减少 padding waste，同时保持 GPU 上较高效率。

## System / Algorithm

需要重点理解：

- token-to-expert assignment。
- block-sparse matrix multiplication。
- grouped GEMM / sparse GEMM 与 MoE 的关系。
- load imbalance 如何影响 compute 和 communication。

## Experiments

- Hardware：GPU cluster。
- Workloads：MoE training。
- Baselines：padding-based MoE implementations。
- Metrics：throughput、FLOPs utilization、scaling efficiency。

## 复现计划

当前阶段作为 MoE compute-side baseline：

- 先读它如何刻画 token imbalance。
- 对照 DeepGEMM / grouped GEMM 理解 MoE expert computation。
- 后续结合 all-to-all trace，看 compute imbalance 是否放大 communication waiting。

## 与我的研究主线的关系

MegaBlocks 连接了 MoE 的 compute side 和 communication side。如果 expert GEMM 粒度或负载不均衡，communication 可能表现为等待慢 rank，而不是纯带宽瓶颈。

## Open Questions

- grouped GEMM 粒度能否隐藏 all-to-all communication？
- token imbalance 对 all-to-all p99 latency 的影响有多大？
- 在 inference 场景中，MegaBlocks 的训练侧结论能迁移多少？
