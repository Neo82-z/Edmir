# 研究笔记

用于记录 multi-GPU LLM serving 方向的研究过程，重点关注 communication bottleneck 的optimize or trace。

## 研究主线

**面向 multi-GPU LLM inference 时候 communication bottleneck and overlap trade-off.**

方向：

> 多 GPU LLM serving 的性能不是由 FLOPS、NCCL bandwidth、KV cache hit rate 或 overlap 开关单独决定，而是由 data movement 是否暴露在 request critical path 上决定。目标是建立一个可校准的解释模型或者说解释路线，可以是一个由多个nodes组成的graph，判断 topology、parallelism、KV movement、runtime scheduling 和 overlap policy 什么时候会影响 TTFT / ITL / P99，后续可以做codegen，持续optimize

当前更具体的研究问题：

> When does overlap help or even harmful in multi-GPU LLM serving?

overlap 包括 DBO-like microbatch overlap、compute-communication overlap、prefill/decode multiplexing 等。核心判断是：

```text
overlap removed exposed communication
  > overlap introduced split / stream / metadata / contention overhead
```

如果这个不成立，trace 里的通信再大，也可能不是正确优化目标。

研究的支线：

- tinygrad：理解 tensor IR、primitive composition 和 collective 语义。(个人非常喜欢的张量计算引擎)
- CUDA C++：理解 kernel、memory hierarchy、stream 和 profiling。（底层语义复杂，更推荐triton来理解算子）
- DeepGEMM / CuTe / CUTLASS：理解计算侧粒度，以及 GEMM / MoE kernel。（MoE是当前阶段的重要模型结构）
- NCCL / NVSHMEM / RDMA / DeepEP：理解 collective communication、MoE-aware communication runtime、topology 和 transport。（算子库与专用执行层）
- vLLM / Megatron / DeepSpeed：回到真实 serving workload 和 parallelism strategy。（推理引擎和训练架构）

但最终的目的

> 如何improve multi-GPU LLM serving 中的 communication？

## 平台边界（not necessary）

当前先把硬件平台分成三层，避免把不同平台上的实验结论混在一起：

- **H100 / H800 / H20 等 Hopper / SM90 平台**：DeepEP V2 主实验平台。重点研究 MoE EP dispatch / combine、NCCL Gin、ElasticBuffer、SM/QP、hybrid/direct path、RDMA/NVLink 域和 compute-communication overlap。
- **A100 / SM80 平台**：NCCL、PyTorch `all_to_all`、NVSHMEM legacy path 和 DeepEP V1 思路的对照平台。不作为 DeepEP V2 最终性能结论，（ada架构有点太老了）
- **4x4090 / Ada PCIe-only 平台**：流程热身和本地 profiling 平台。适合做 NCCL/PyTorch distributed、PCIe/NUMA 拓扑、Nsight 流程和小型代码阅读验证；不进入 DeepEP V2 论文式性能结论。（没有nvlink，ib不好做消融）
## 仓库结构

```text
paper/         面向 MLSys 2027 的论文问题、模型、claims 和实验矩阵。
papers/        论文笔记，按研究方向归类。
experiments/   可复现实验 slice，包括 raw log、结果表和图。
ideas/         研究问题、方向和设计草图。
templates/     paper note 和 experiment record 的 Markdown 模板。
```

## 当前论文工作区

- [`paper/mlsys-2027/README.md`](paper/mlsys-2027/README.md)：MLSys 方向总览。
- [`paper/mlsys-2027/scope.md`](paper/mlsys-2027/scope.md)：问题边界和研究 taste。
- [`paper/mlsys-2027/model.md`](paper/mlsys-2027/model.md)：exposed data movement 模型草稿。需持续演化。
- [`paper/mlsys-2027/claims-and-experiments.md`](paper/mlsys-2027/claims-and-experiments.md)：claim-to-experiment 矩阵。


## 工作规则

一个实验只回答一个问题。raw data、commands、environment 和 observations 要放在一起，避免之后无法复现或解释。
