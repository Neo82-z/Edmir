# Paper：DistServe

## Citation

- Authors：Yinmin Zhong 等
- Venue / Year：OSDI 2024
- Title：DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- Link：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin

## 一句话总结

DistServe 将 prefill 和 decode 分离到不同资源池中，并以 goodput 和 SLO 为目标进行 serving 调度。

## 问题

Prefill 和 decode 的资源需求不同：prefill 更偏 compute-intensive，decode 更偏 memory / latency sensitive。把它们放在同一 GPU pool 中容易互相干扰。

## 关键想法

Prefill / decode disaggregation：为两个阶段分别配置资源，并通过调度与 KV transfer 连接两阶段。

## System / Algorithm

需要重点理解：

- prefill / decode disaggregation。
- goodput-oriented serving。
- SLO-aware placement。
- 两阶段之间的 KV handoff / communication cost。

## Experiments

- Hardware：多 GPU serving cluster。
- Workloads：不同 request rate、prompt length、output length。
- Baselines：co-located prefill/decode serving。
- Metrics：goodput、TTFT、TPOT、SLO attainment。

## 复现计划

第一阶段不复现完整系统，先复现实验思想：

- 在 vLLM 中区分 prefill-heavy 与 decode-heavy workload。
- 用 trace 分解 prefill / decode 时间。
- 估算如果 P/D 分离，需要多大的 KV transfer bandwidth。
- 与 NCCL / PCIe baseline 对齐，判断 transfer 是否可能成为 bottleneck。

## 与我的研究主线的关系

DistServe 直接把 serving 和 communication 绑在一起。P/D disaggregation 的核心问题之一就是 KV movement 是否会进入 critical path。

## Open Questions

- 在没有 RDMA 的 commodity PCIe 单机上，能否模拟 P/D disaggregation 的通信边界？
- KV handoff 的成本什么时候超过 recomputation？
- P/D 分离后，network QoS 是否会主导 p99 latency？
