# Paper：MSCCL++

## Citation

- Authors：MSCCL++ team
- Venue / Year：arXiv 2025 / 会议版本待核对
- Title：MSCCL++: Rethinking GPU Communication Abstractions for AI Inference
- Link：https://arxiv.org/abs/2504.09014

## 一句话总结

MSCCL++ 重新设计 GPU communication abstraction，让 AI workload 能更灵活地表达和实现 communication / computation 交织的模式。

## 问题

传统 collective library 提供固定 API，如 all-reduce、all-gather、all-to-all，但 AI inference 里经常需要更细粒度、更可编程、更容易融合的 communication pattern。

## 关键想法

提供更低层、更可编程的 GPU communication primitive，使系统可以把 communication 与 kernel / scheduling 更紧密地结合。

## System / Algorithm

需要重点理解：

- GPU-side communication abstraction。
- 与 NCCL / MSCCL 的关系。
- AI inference 中为什么固定 collective API 不够。
- fused communication kernels 或 communication-compute overlap 的表达方式。

## Experiments

- Hardware：modern NVIDIA GPUs / multi-GPU systems。
- Workloads：AI inference communication patterns。
- Baselines：NCCL、MSCCL、framework-level collectives。
- Metrics：latency、bandwidth、end-to-end inference performance。

## 复现计划

短期先阅读，不急着复现：

- 对照 NCCL tests 理解 fixed collective abstraction。
- 找论文中最小 communication primitive 示例。
- 记录哪些场景与 vLLM / MoE / KV transfer 相关。

## 与我的研究主线的关系

MSCCL++ 很贴近“communication + inference runtime + 底层掌控”。它可以作为后续从 NCCL microbench 走向 programmable communication 的参考。

## Open Questions

- 哪些 vLLM collective pattern 需要超出固定 NCCL API？
- MoE all-to-all 是否适合用更可编程的 communication abstraction？
- 对 4x4090 PCIe topology，MSCCL++ 类方法是否能带来可观收益？
