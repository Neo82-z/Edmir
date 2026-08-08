# Paper：Sarathi-Serve

## Citation

- Authors：Amey Agrawal 等
- Venue / Year：OSDI 2024
- Title：Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- Link：https://arxiv.org/abs/2403.02310

## 总结

Sarathi-Serve 通过 chunked prefill 和 stall-free scheduling 改善 LLM serving 中 throughput 与 latency 的 tradeoff。

## 问题

Prefill 通常 compute-heavy，decode 通常 memory-bandwidth-heavy。二者直接混排时容易互相阻塞，导致 latency 或 throughput 受损。

## 关键想法

把长 prefill 拆成多个 chunks，让 prefill 和 decode 更细粒度地交错执行，减少 decode stall，同时维持较高 GPU 利用率。

## System / Algorithm

需要重点理解：

- chunked prefill。
- stall-free scheduling。
- prefill / decode 的资源需求差异。
- chunk size 对 latency 和 throughput 的影响。

## Experiments

- Hardware：GPU serving environment。
- Workloads：不同 prompt length、decode length 和 arrival rate。
- Baselines：vLLM / Orca-style scheduler。
- Metrics：throughput、TTFT、TPOT、SLO attainment。

## 复现计划

- 在 vLLM 中构造长 prompt workload。
- 比较 chunked prefill 相关配置对 TTFT / TPOT 的影响。
- 用 Nsight Systems 看 prefill / decode 是否互相阻塞。
- 后续再观察 TP 场景下 collective 是否出现在 stall 区间。

## 与我的研究主线的关系

Sarathi-Serve 提供了 serving schedule 与 runtime trace 之间的连接。它帮助判断 communication 是真正 bottleneck，还是被调度造成的 compute / memory stall 放大。

## Open Questions

- chunked prefill 会增加还是减少 collective fragmentation？
- 更细粒度 scheduling 是否有助于 overlap communication？
- 在 PCIe-only 多卡上，chunk size 是否需要 topology-aware 调整？
