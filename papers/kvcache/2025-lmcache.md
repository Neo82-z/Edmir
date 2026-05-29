# Paper：LMCache

## Citation

- Authors：LMCache team
- Venue / Year：arXiv 2025
- Title：LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- Link：https://arxiv.org/abs/2510.09665

## 一句话总结

LMCache 试图把 KV cache 做成 serving engine 外部的 cache layer，让不同请求、不同 engine 或不同实例之间可以复用 KV。

## 问题

LLM serving 中大量请求可能共享 prefix 或历史上下文，但 KV cache 通常被绑定在单个 engine / GPU / process 内，跨请求和跨实例复用困难。

## 关键想法

抽象出 KV cache layer，支持 offload、reuse 和跨 serving engine transfer，使 KV cache 不再只是单个 inference engine 内部的临时状态。

## System / Algorithm

需要重点理解：

- KV cache connector。
- KV offload / reload。
- prefix reuse。
- 与 vLLM / SGLang 等 serving engine 的集成方式。

## Experiments

- Hardware：GPU serving machine + CPU memory / storage。
- Workloads：prefix sharing、long context、enterprise serving。
- Baselines：无 KV reuse / engine-local KV cache。
- Metrics：TTFT、throughput、cache hit rate、memory usage。

## 复现计划

- 优先复现开源系统的 connector / offload 路径。
- 构造 shared-prefix workload。
- 对比 cache hit / miss 下 TTFT 和 throughput。
- 暂时不把它作为底层 RDMA paper，先作为 KV movement baseline。

## 与我的研究主线的关系

LMCache 可以作为 vLLM 侧 KV movement 的工程入口。它帮助把 Mooncake 这类系统问题落到更容易动手的 serving stack。

## Open Questions

- cache hit rate 多高时才明显改善 TTFT？
- offload / reload 是否会与 GPU collective communication 竞争 PCIe bandwidth？
- KV cache layer 的 placement policy 是否需要 topology-aware？
