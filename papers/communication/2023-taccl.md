# Paper：TACCL

## Citation

- Authors：Shah 等
- Venue / Year：NSDI 2023
- Title：TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches
- Link：https://www.usenix.org/conference/nsdi23/presentation/shah

## 一句话总结

TACCL 在 collective algorithm synthesis 中引入 communication sketches，让用户用较高层的 sketch 指导算法搜索，从而兼顾可控性和性能。

## 问题

完全自动 synthesis 搜索空间很大，且不一定体现系统设计者对 topology 或 workload 的先验理解。

## 关键想法

用 communication sketch 表达算法结构或约束，再由系统根据 topology 和 cost model 搜索具体 schedule。

## System / Algorithm

需要重点理解：

- communication sketch。
- topology-aware schedule generation。
- collective primitive 与算法结构的关系。
- sketch 如何缩小搜索空间。

## Experiments

- Hardware：多 GPU / 多节点不同拓扑。
- Workloads：collective microbenchmarks。
- Baselines：NCCL、SCCL、hand-tuned algorithms。
- Metrics：latency、bandwidth、scalability。

## 复现计划

当前阶段重点是阅读和抽象迁移：

- 把 4x4090 topology 写成 graph，作为 PCIe-only 热身。
- 手工 sketch 几种 all-reduce / all-gather 路径。
- 对比 NCCL logs 中实际 ring / graph 与手工直觉是否一致。
- 后续在 H100/Hopper 上再考虑 MoE EP dispatch/combine 是否需要 workload-aware sketch。

## 与我的研究主线的关系

TACCL 适合启发后续“topology-aware parallelism / collective selection”。它让我们不只测 NCCL，还能问：在这个 topology 和 workload 下，算法本身是否还有优化空间？

## Open Questions

- LLM inference 中的 collective pattern 是否可以用 workload-aware sketches 描述？
- 对 PCIe-only 机器，sketch 是否能避免跨 NUMA link 的过度使用？
- 这类方法如何与 vLLM runtime scheduler 结合？
- DeepEP V2 已经把 MoE routing/layout/communication 放进专用 runtime，TACCL 的 sketch 思想是否能用于解释或改进它的路径选择？
