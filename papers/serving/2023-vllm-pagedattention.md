# Paper：vLLM / PagedAttention

## Citation

- Authors：Woosuk Kwon 等
- Venue / Year：SOSP 2023
- Title：Efficient Memory Management for Large Language Model Serving with PagedAttention
- Link：https://arxiv.org/abs/2309.06180

## 一句话总结

vLLM 通过 PagedAttention 把 KV cache 管理做成类似 virtual memory 的 block-based 管理，从而提升 LLM serving 的吞吐和显存利用率。

## 问题

LLM serving 中 KV cache 会占用大量 GPU memory，传统连续内存分配容易造成 fragmentation 和低利用率，限制 batch size 和 throughput。

## 关键想法

把 KV cache 切成固定大小 blocks，通过 block table 管理 token 到 KV block 的映射，让请求之间可以更灵活地分配、复用和释放 KV cache。

## System / Algorithm

需要重点理解：

- PagedAttention 的 KV block abstraction。
- continuous batching 与 KV cache allocation 的关系。
- prefill / decode 阶段对 KV cache 的不同压力。
- throughput、latency、memory utilization 之间的 tradeoff。

## Experiments

- Hardware：论文中使用 GPU serving cluster；本地 4x4090 只作为 commodity PCIe warmup baseline。
- Workloads：不同模型、batch size、sequence length、request rate。
- Baselines：HuggingFace Transformers、FasterTransformer、Orca-style serving。
- Metrics：throughput、latency、GPU memory usage。

## 复现计划

第一阶段只复现局部 baseline：

- 跑 vLLM benchmark。
- 改变 batch size、input length、output length。
- 记录 TTFT、TPOT、throughput、GPU memory。
- 与 NCCL baseline 对齐前，先建立单机 serving baseline。
- DeepEP V2 / MoE EP 相关结论必须迁移到 H100/Hopper 环境验证。

## 与我的研究主线的关系

vLLM 是后续 communication trace 的真实 workload 入口。PagedAttention 也直接连接 KV movement、KV cache offload 和 remote KV transfer 等问题。

## Open Questions

- PagedAttention 本身降低的是 memory fragmentation，那么它对 multi-GPU communication 的间接影响是什么？
- TP / EP 下 KV cache 的 placement 和 movement 如何改变 communication critical path？
- vLLM trace 中哪些 collective 与 PagedAttention 的调度策略有关？
