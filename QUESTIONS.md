# Research Questions

## 主问题

multi-GPU LLM inference 中的 communication bottleneck 在哪里？什么时候可以通过算法、scheduling 或 topology-aware decision 来改善？

## Communication

- NCCL collective time 什么时候会进入 critical path，而不是被 compute overlap 掉？
- 在 commodity PCIe 系统上，near-GPU 和 cross-NUMA 的 collective bandwidth 差异有多大？
- 在 4x4090 或者 H100，A100上，哪类 collective 最脆弱：all-reduce、all-gather、reduce-scatter，还是 all-to-all？
- 关闭 P2P（4090没有NVLink） 会造成多大影响？这能揭示真实 transfer path 的哪些信息？
- empirical NCCL roofline 能否预测 vLLM 中的 collective time？

## Serving Runtime

- 在 vLLM 中，哪个阶段触发最重要的 communication：prefill、decode、MoE dispatch，还是 KV movement？
- 在 PCIe-only 机器上，serving 什么时候应该选择 TP、PP、DP 或 EP？
- batch size、context length 和 concurrency 如何改变 communication / computation ratio？
- 哪些 workload shape 最容易暴露 all-to-all 或 all-gather bottleneck？

## KV Cache

- remote KV transfer 什么时候比 recomputation 更划算？
- KV movement 要产生收益，需要怎样的 cache hit rate？
- 对 tail latency 影响最大的部分是什么：lookup、transfer、decompression，还是 decode blocking？

## Kernel / Compute Side

- GEMM granularity 能否隐藏 communication？
- grouped GEMM 和 MoE token imbalance 会怎样影响 communication critical path？
- tinygrad / CuTe 的抽象能如何帮助理解 high-level tensor semantics 到 efficient kernels 的 lowering？
