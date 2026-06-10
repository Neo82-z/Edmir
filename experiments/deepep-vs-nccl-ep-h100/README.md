# Experiment：DeepEP V2 vs NCCL/PyTorch EP on H100

## 定位

这是 DeepEP / MoE EP 支线的主实验 slice。当前先记录研究计划，等 H100/H800/H20 等 Hopper/SM90 环境可用后再补 commands、raw logs 和结果。

核心判断：

> DeepEP V2 的价值不只是更快的 all-to-all，而是把 MoE EP 的 routing、layout、communication、combine reduction 和 compute-overlap 共同放进专用 runtime。NCCL 是底层通信能力，DeepEP 是 MoE-aware communication execution layer。

## 平台边界

| 平台 | 角色 | 是否进入 DeepEP V2 主结论 |
|---|---|---|
| H100 / H800 / H20, SM90 | DeepEP V2 主实验平台 | 是 |
| A100, SM80 | NCCL / PyTorch `all_to_all` / NVSHMEM legacy 对照 | 否，只作历史和 baseline 对照 |
| 4x4090, Ada PCIe-only | NCCL/PyTorch distributed、PCIe topology、profiling 流程热身 | 否 |

## 第一阶段：本地读代码

目标：先不依赖机器，把 DeepEP 与 NCCL / PyTorch reference path 的差异讲清楚。

重点文件：

- `deep_ep/utils/refs.py`：NCCL / PyTorch reference dispatch / combine。
- `deep_ep/buffers/elastic.py`：DeepEP V2 Python API。
- `csrc/kernels/backend/nccl.cu`：NCCL Gin、device communicator、symmetric window。
- `csrc/elastic/buffer.hpp`：dispatch / combine host path。
- `tests/elastic/test_ep.py`：实验入口。

需要回答：

- DeepEP V2 的 dispatch / combine API 如何映射到 MoE EP workload？
- reference path 使用了哪些通用 collective 或 PyTorch/NCCL 操作？
- ElasticBuffer 在 host path 和 device path 中分别承担什么角色？
- cached handle / routing metadata 如何避免 decode 阶段重复开销？

## 第二阶段：H100 主实验

主问题不是“DeepEP 比 NCCL 快多少”，而是：

- DeepEP 省掉了哪些通用 all-to-all 的额外步骤？
- routing metadata / cached handle 对 decoding 有没有收益？
- `num_sms` 少用以后，GEMM overlap 是否更好？
- QP 数、hybrid/direct、RDMA/NVLink 域如何影响 tail latency？
- token imbalance 下，瓶颈是网络、copy epilogue、CPU sync，还是 expert GEMM 等待？

初始 metrics：

- dispatch latency。
- combine latency。
- end-to-end MoE layer latency。
- P50 / P95 / P99 latency。
- tokens/s。
- GPU SM utilization。
- NCCL / DeepEP communication time ratio。
- overlap 后仍暴露在 critical path 上的 communication time。

## 第三阶段：A100 对照

A100 可以回答：

- 标准 NCCL / PyTorch `all_to_all` 的 baseline。
- A100 NVLink/IB 下 MoE dispatch 的通信边界。
- DeepEP legacy V1 / NVSHMEM 与 V2 思路的历史对照。
- 为什么 V2 转向 Hopper/SM90、NCCL Gin 和更少 SM 占用。

但不要用 A100 去证明 DeepEP V2 的最终性能。

## 当前状态

- [ ] 本地 clone / 阅读 DeepEP V2 主线代码。
- [ ] 补第一版 code-reading note。
- [ ] 申请 H100/H800/H20 Hopper 环境。
- [ ] 确认多节点 RDMA/NVLink/NVSwitch topology。
- [ ] 跑 reference NCCL / PyTorch dispatch/combine baseline。
- [ ] 跑 DeepEP V2 dispatch/combine。
- [ ] 补 Nsight Systems trace。
