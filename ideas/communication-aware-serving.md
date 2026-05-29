# Communication-Aware LLM Serving

## 核心判断

multi-GPU LLM serving 的性能往往不只受 GPU compute 限制，还受到 collective communication、topology、serving schedule 和 workload shape 之间相互作用的影响。

## 初始假设

一个小而清晰的 empirical communication roofline，可以帮助预测 TP / PP / DP / EP 什么时候会在 commodity PCIe multi-GPU system 上表现不佳。

## 可能贡献

- 一套把 NCCL microbenchmark 和 vLLM trace 连接起来的 measurement methodology。
- 一个 topology-aware parallelism selection policy。
- 一个小工具，用于估计某个 serving configuration 是 latency-bound、bandwidth-bound，还是 compute-bound。
- 一个基于 4x4090 dual-NUMA PCIe topology 的 case study。

## 第一批需要收集的证据

- `nvidia-smi topo -m`。
- NCCL all-reduce、all-gather、reduce-scatter、all-to-all 曲线。
- pairwise near-GPU 与 cross-NUMA performance 对比。
- 一个 small-message collective 和一个 large-message collective 的 Nsight trace。
- NCCL baseline 稳定后，再补 vLLM TP trace。
