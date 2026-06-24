# Research Questions

## 主问题

multi-GPU LLM inference 中的 communication bottleneck 在哪里？什么时候可以通过算法、scheduling 或 topology-aware decision 来改善？

更精确的 MLSys 问题：

- 什么时候 communication / KV movement 会暴露到 critical path，并转化成 TTFT / ITL / P99？
- total communication time、NCCL microbenchmark、GPU FLOPS、KV logical hit rate 哪些情况下会误导我们？
- 能否用少量 roofline + trace 建立一个可校准的 exposed data movement model？
- 这个模型能否在 held-out configs 上提前判断 bad TP degree、bad placement、bad KV transfer choice？

## Communication

- NCCL collective time 什么时候会进入 critical path，而不是被 compute overlap 掉？
- 在 commodity PCIe 系统上，near-GPU 和 cross-NUMA 的 collective bandwidth 差异有多大？
- 在 H100/Hopper 主实验上，哪类 communication pattern 最脆弱：all-reduce、all-gather、reduce-scatter、all-to-all，还是 MoE dispatch/combine？
- 4x4090 上关闭 P2P 会造成多大影响？这只能揭示 PCIe-only transfer path，不能外推到 DeepEP V2。
- empirical NCCL roofline 能否预测 vLLM 中的 collective time？
- exposed communication ratio 是否比 total NCCL time 更能解释 ITL / P99？

## DeepEP / MoE EP

- DeepEP V2 相比 NCCL / PyTorch `all_to_all`，究竟省掉了哪些通用通信路径的额外步骤？
- routing metadata、cached handle 和 ElasticBuffer 对 decode-heavy workload 有没有收益？
- `num_sms` 少用以后，expert GEMM 和 communication overlap 是否更好？
- QP 数、hybrid/direct path、RDMA/NVLink 域如何影响 P50/P99 latency？
- token imbalance 下，瓶颈是网络、copy epilogue、CPU sync，还是 expert GEMM 等待？
- A100/SM80 上的 NCCL / NVSHMEM legacy 对照，能解释哪些 V2 转向 Hopper/SM90 的动机？

## Serving Runtime

- 在 vLLM 中，哪个阶段触发最重要的 communication：prefill、decode、MoE dispatch，还是 KV movement？
- 在 PCIe-only 机器上，serving 什么时候应该选择 TP、PP、DP 或 EP？这个问题只作为热身和方法验证。
- 在 H100/Hopper 多节点环境中，DeepEP-aware EP 什么时候比通用 NCCL / PyTorch collective 更适合？
- batch size、context length 和 concurrency 如何改变 communication / computation ratio？
- 哪些 workload shape 最容易暴露 all-to-all 或 all-gather bottleneck？
- runtime scheduling、small-op density、dependency fan-in 是否会成为 communication 之外的 tail latency 来源？

## KV Cache

- remote KV transfer 什么时候比 recomputation 更划算？
- KV movement 要产生收益，需要怎样的 cache hit rate？
- 对 tail latency 影响最大的部分是什么：lookup、transfer、decompression，还是 decode blocking？
- logical hit rate 和 effective hit rate 在什么情况下分离？
- KV transfer、PD handoff、EP all-to-all、storage backup 是否会在同一张 fabric 上互相干扰？

## Kernel / Compute Side

- GEMM granularity 能否隐藏 communication？
- grouped GEMM 和 MoE token imbalance 会怎样影响 communication critical path？
- tinygrad / CuTe 的抽象能如何帮助理解 high-level tensor semantics 到 efficient kernels 的 lowering？
