# Papers

当前 compiler v0.1 的入口：

- [`compiler/communication-compute-fusion.md`](compiler/communication-compute-fusion.md)：直接竞争工作、NVIDIA 技术栈、collective compiler 谱系和 Edmir claim boundary。

建议按这些方向分类：

- `serving/`：vLLM、Orca、Sarathi-Serve、DistServe、Splitwise。
- `communication/`：NCCL、SCCL、TACCL、MSCCL、NVSHMEM。
- `kernels/`：DeepGEMM、CuTe、CUTLASS、Triton、tinygrad。
- `kvcache/`：Mooncake、LMCache、KV offload 和 transfer systems。
- `compiler/`：communication-compute fusion、distributed kernel compiler 和 lowering/legality prior art。

每篇论文可以写一份简洁 note 重点提取：

- problem statement
- key mechanism
- experimental setup
- what to reproduce
- open questions
