# Paper：Splitwise

## Citation

- Authors：Microsoft Research 等
- Venue / Year：ISCA 2024
- Title：Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- Link：https://arxiv.org/abs/2311.18677

## 一句话总结

Splitwise 将 generative LLM inference 的不同 phase 拆开，让 prefill 和 decode 使用更合适的硬件资源和调度方式。

## 问题

Prefill 和 decode 的计算/内存特征不同，用同一种资源配置服务两者会造成资源浪费或尾延迟变差。

## 关键想法

Phase splitting：把 inference workload 拆成不同 phase，并针对每个 phase 使用不同硬件或不同调度策略。

## System / Algorithm

需要重点理解：

- phase-level workload characterization。
- prefill / decode 的 compute 与 memory profile。
- 异构硬件上的 placement。
- phase splitting 与 KV transfer 的关系。

## Experiments

- Hardware：异构 GPU serving resources。
- Workloads：生成式 LLM inference。
- Baselines：不区分 phase 的 serving。
- Metrics：throughput、latency、cost efficiency。

## 复现计划

- 在 4x4090 上先做 phase characterization。
- 分别测 prefill-heavy 和 decode-heavy workload。
- 记录 GPU utilization、memory bandwidth、collective time。
- 作为 DistServe 的对照阅读。

## 与我的研究主线的关系

Splitwise 帮助建立 workload phase 与 resource demand 的 mental model。communication optimization 不能脱离 phase，因为 prefill 和 decode 对 communication 的敏感性不同。

## Open Questions

- Phase splitting 是否会引入额外 KV movement bottleneck？
- PCIe-only 机器上能否通过 phase-aware scheduling 改善 TP collective 的 critical path？
- 异构硬件结论能否迁移到 commodity multi-GPU？
