# Baseline Reading List

这份列表按 Edmir v0.1 的进入顺序排列，不代表论文价值排名。完整定位矩阵见 [`compiler/communication-compute-fusion.md`](compiler/communication-compute-fusion.md)。

当前成本建模优先阅读：**Analytical Performance Estimation during Code Generation on Modern GPUs** — [全文](https://arxiv.org/html/2204.14242v1)、[本地简记](compiler/analytical-performance-estimation-2022.md)。重点看指标与性能的分层验证、经验项的边界和失败情况；下面融合相关工作继续用于划定 prior art。

2026-10-28 针对误选的最短增补（[代码与公开数据核对记录](../experiments/2026-10-28-overlap-evidence/README.md)）：

- [Demystifying NCCL v3](https://arxiv.org/html/2507.04786v3)：启动、协议、channel/分块；版本为 NCCL 2.19.1，常数不直接移植。
- [DeepEP](https://github.com/deepseek-ai/DeepEP) + [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data)：资源需求与真实时间线；区分 kernel 并发和网络传输隐藏，固定版本。
- [T-CCL](https://arxiv.org/html/2610.07098v1) + [artifact](https://github.com/ooverlap/ooverlap/tree/9abd9cf93286fc8ca5a0a041db6234c4e86e1635)：受限资源的通信/计算曲线与 overlap 负收益；借测量设计，不迁移 H100/GH200 速度。
- [Alibaba InferSim](https://github.com/alibaba/InferSim)：公开 GEMM/grouped GEMM 数据及 MFU 查询；不是我们 A100 候选排序的训练标签。

2026-10-25 补充的最短清单：

- [Hong & Kim，ISCA 2009](https://faculty.cc.gatech.edu/~hyesoon/hong_isca09.pdf)：访存并行度、延迟隐藏与带宽限制。
- [NeuSight，ASPLOS 2025](https://arxiv.org/html/2407.13853v3)：tile 利用率与性能约束、未见形状验证；不直接迁移其学习模型。
- [Entwine，2026-09 预印本](https://arxiv.org/html/2609.11562v1)：通信并发与计算减速的受控消融，任务是 GEMM–ReduceScatter。
- [RRZE gpu-benches](https://github.com/RRZE-HPC/gpu-benches)：独立延迟、带宽和缓存微基准；原实现不能直接代表 peer 访问。

这些材料对应的主张边界、可复用范围与八卡验证项见[机制证据计划](../paper/P8_MECHANISM_EVIDENCE_PLAN.md)。

## P0：在设计 compiler IR 前必须读

1. **TileLink**
   - [paper](https://arxiv.org/abs/2503.20313)
   - 提取：tile-centric primitives、rank/shape/channel mapping、release/acquire lowering、dynamic mapping。

2. **Triton-distributed**
   - [paper](https://arxiv.org/abs/2504.19442) / [code](https://github.com/ByteDance-Seed/Triton-distributed)
   - 提取：symmetric memory、signal exchange、async task、AllGather/GEMM 与 All2All 的实现边界。

3. **AutoOverlap**
   - [paper](https://arxiv.org/abs/2601.20595)
   - 提取：local Triton kernel + chunk schedule 的输入契约和 source-to-source transformations。

4. **FLUX**
   - [paper](https://arxiv.org/abs/2406.06858) / [code](https://github.com/bytedance/flux)
   - 提取：GEMM prologue/epilogue fusion、All2All substep、signals、swizzle 和 autotuning。

5. **NCCL Device API**
   - [documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
   - 提取：window、LSA pointer、barrier session、memory order、version compatibility 和硬件要求。

6. **CoCoNet**
   - [paper page](https://www.microsoft.com/en-us/research/publication/breaking-the-computation-and-communication-abstraction-barrier-in-distributed-machine-learning-workloads/)
   - 提取：communication/compute first-class IR 和 transformation 边界。

## P1：在写 verifier/lowering 前读

7. **MSCCLang** — [paper](https://doi.org/10.1145/3575693.3575724)
   - 提取 collective correctness、deadlock/data-race avoidance 和 lowering contract。

8. **MSCCL++** — [paper](https://arxiv.org/abs/2504.09014)
   - 提取低层 primitive、同步/一致性抽象和 portability boundary。

9. **GC3** — [paper page](https://www.microsoft.com/en-us/research/publication/gc3-an-optimizing-compiler-for-gpu-collective-communication/)
   - 提取 data-oriented collective DSL 与 runtime 接口。

10. **SCCL / TACCL**
    - [SCCL](https://arxiv.org/abs/2008.08708)
    - [TACCL](https://arxiv.org/abs/2111.04867)
    - 提取 topology/schedule 表达；v0.1 不实现它们的 collective synthesis。

## P2：MoE 与后续机制选择

11. **DeepEP / DeepEP V2** — [repository](https://github.com/deepseek-ai/DeepEP) / [V2 announcement](https://github.com/deepseek-ai/DeepEP/pull/605)
12. **NCCL EP** — [paper](https://arxiv.org/abs/2603.13606)
13. **GPU-Initiated Networking for NCCL** — [paper](https://arxiv.org/abs/2511.15076)
14. **Comet** — [paper](https://openreview.net/pdf?id=fGgQS5VW09)
15. **Lancet** — [paper](https://arxiv.org/abs/2404.19429)

这些材料用于 v0.1 之后的动态 MoE、GIN/RDMA 和 mechanism selection，不应扩大当前 scope。

## 阅读规则

每篇只先记录八项：

1. 输入 IR/API；
2. schedule unit；
3. communication backend；
4. synchronization/memory model；
5. 自动决定了什么；
6. 用户仍需手写什么；
7. 生成或执行什么 artifact；
8. 对 Edmir candidate claim 的直接反例是什么。
