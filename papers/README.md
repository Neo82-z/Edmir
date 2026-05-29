# Papers

论文笔记要优先服务“可复现的 idea”和“可落地的实验入口”。

建议按这些方向分类：

- `serving/`：vLLM、Orca、Sarathi-Serve、DistServe、Splitwise。
- `communication/`：NCCL、SCCL、TACCL、MSCCL、NVSHMEM。
- `kernels/`：DeepGEMM、CuTe、CUTLASS、Triton、tinygrad。
- `kvcache/`：Mooncake、LMCache、KV offload 和 transfer systems。

每篇论文尽量写一份简洁 note，不要大段摘抄。重点提取：

- problem statement
- key mechanism
- experimental setup
- what to reproduce
- open questions
