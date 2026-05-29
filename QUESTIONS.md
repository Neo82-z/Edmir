# Research Questions

## Main Question

Where are the communication bottlenecks in multi-GPU LLM inference, and when can they be improved by algorithm, scheduling, or topology-aware decisions?

## Communication

- When does NCCL collective time become part of the critical path rather than being hidden by compute?
- How different are near-GPU and cross-NUMA collective bandwidths on commodity PCIe systems?
- Which collectives are most fragile on 4x4090: all-reduce, all-gather, reduce-scatter, or all-to-all?
- How much does disabling P2P hurt, and what does that reveal about the actual transfer path?
- Can an empirical NCCL roofline predict vLLM collective time?

## Serving Runtime

- In vLLM, which phase triggers the most important communication: prefill, decode, MoE dispatch, or KV movement?
- When should serving prefer TP, PP, DP, or EP on PCIe-only machines?
- How do batch size, context length, and concurrency change the communication/computation ratio?
- What workload shapes expose all-to-all or all-gather bottlenecks most clearly?

## KV Cache

- When is remote KV transfer cheaper than recomputation?
- What cache hit rate is required for KV movement to pay off?
- Which part matters most for tail latency: lookup, transfer, decompression, or decode blocking?

## Kernel / Compute Side

- Can GEMM granularity hide communication?
- How do grouped GEMM and MoE token imbalance affect communication critical path?
- What can tinygrad/CuTe abstractions teach about lowering high-level tensor semantics into efficient kernels?
