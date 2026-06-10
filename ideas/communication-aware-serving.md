# Communication-Aware LLM Serving

## 核心判断

multi-GPU LLM serving 的性能往往不只受 GPU compute 限制，还受到 collective communication、topology、serving schedule 和 workload shape 之间相互作用的影响。

## 初始假设

一个小而清晰的 empirical communication roofline，可以帮助预测 TP / PP / DP / EP 什么时候会在给定 topology 上表现不佳。

但是平台边界必须拆开：

- 4x4090 / Ada / PCIe-only：只做 NCCL、PyTorch distributed、topology 和 profiling 流程热身。
- A100 / SM80：做 NCCL / PyTorch `all_to_all` / NVSHMEM legacy 对照。
- H100 / H800 / H20 / SM90：作为 DeepEP V2 主实验平台。

## 可能贡献

- 一套把 NCCL microbenchmark 和 vLLM trace 连接起来的 measurement methodology。
- 一个 topology-aware parallelism selection policy。
- 一个小工具，用于估计某个 serving configuration 是 latency-bound、bandwidth-bound，还是 compute-bound。
- 一个基于 4x4090 dual-NUMA PCIe topology 的 warmup case study。
- 一个 DeepEP V2 vs NCCL / PyTorch `all_to_all` 的 H100 MoE EP case study。

## 第一批需要收集的证据

- `nvidia-smi topo -m`。
- NCCL all-reduce、all-gather、reduce-scatter、all-to-all 曲线。
- pairwise near-GPU 与 cross-NUMA performance 对比。
- 一个 small-message collective 和一个 large-message collective 的 Nsight trace。
- NCCL warmup baseline 稳定后，再补 vLLM TP trace。

## DeepEP V2 支线判断

DeepEP V2 的价值不只是更快的 all-to-all，而是把 MoE EP 的 routing、layout、communication、combine reduction 和 compute-overlap 共同放进专用 runtime；NCCL 是底层通信能力，DeepEP 是 MoE-aware communication execution layer。

因此 DeepEP V2 的主结论必须来自 Hopper/SM90 平台。A100 更适合作为 legacy 对照。
