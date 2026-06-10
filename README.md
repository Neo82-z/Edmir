# LLM Serving 研究笔记

这个仓库用于记录 multi-GPU LLM serving 方向的研究过程，重点关注 communication bottleneck 的诊断与优化。

## 研究主线

**面向 multi-GPU LLM inference 的 communication bottleneck 诊断与优化。**

研究的支线：

- tinygrad：理解 tensor IR、primitive composition 和 collective 语义。(个人非常喜欢的张量计算引擎)
- CUDA C++：理解 kernel、memory hierarchy、stream 和 profiling。（底层语义复杂，更推荐triton来理解算子）
- DeepGEMM / CuTe / CUTLASS：理解计算侧粒度，以及 GEMM / MoE kernel。（MoE是当前阶段的重要模型结构）
- NCCL / NVSHMEM / RDMA / DeepEP：理解 collective communication、MoE-aware communication runtime、topology 和 transport。（算子库与专用执行层）
- vLLM / Megatron / DeepSpeed：回到真实 serving workload 和 parallelism strategy。（推理引擎和训练架构）

但需最终回归到

> 这件事如何improve multi-GPU LLM serving 中的 communication

## 平台边界

当前先把硬件平台分成三层，避免把不同平台上的实验结论混在一起：

- **H100 / H800 / H20 等 Hopper / SM90 平台**：DeepEP V2 主实验平台。重点研究 MoE EP dispatch / combine、NCCL Gin、ElasticBuffer、SM/QP、hybrid/direct path、RDMA/NVLink 域和 compute-communication overlap。
- **A100 / SM80 平台**：NCCL、PyTorch `all_to_all`、NVSHMEM legacy path 和 DeepEP V1 思路的对照平台。可用于解释历史演进，但不作为 DeepEP V2 最终性能结论平台。
- **4x4090 / Ada PCIe-only 平台**：流程热身和本地 profiling 平台。适合做 NCCL/PyTorch distributed、PCIe/NUMA 拓扑、Nsight 流程和小型代码阅读验证；不进入 DeepEP V2 论文式性能结论。

## 仓库结构

```text
papers/        论文笔记，按研究方向归类。
experiments/   可复现实验 slice，包括 raw log、结果表和图。
ideas/         研究问题、潜在 paper 方向和设计草图。
templates/     paper note 和 experiment record 的 Markdown 模板。
```


## 工作规则

一个实验只回答一个问题。raw data、commands、environment 和 observations 要放在一起，避免之后无法复现或解释。
