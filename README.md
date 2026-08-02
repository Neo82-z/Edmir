# Edmir

Edmir 是一个面向多 GPU 通信—计算流水的编译研究原型。

当前仓库只推进一个可验证问题：

> 给定单节点拓扑和一个静态 `All2All → GEMM` 数据流，能否生成正确的顺序执行与 tile-level 流水调度，并解释它们在什么条件下值得使用？

`All2All → GEMM` 是 v0.1 的研究载体，不是“首次融合该算子组合”的创新声明。FLUX、TileLink、Triton-distributed、AutoOverlap、DeepEP/NCCL EP 等工作已经覆盖了相邻或部分重叠的设计空间。Edmir 只有在明确区分这些工作的前提下，才继续讨论论文贡献。

## v0.1 范围

固定输入：

- 单节点、多 GPU；
- 静态 shape 和静态 rank 数；
- 等尺寸、对称布局的 All2All；
- 一个消费通信结果的 GEMM；
- 已知的 P2P/NVLink/PCIe 拓扑描述。

固定输出：

1. 顺序基线：host-launched NCCL All2All，随后执行 GEMM（NCCL 2.30+；旧版本以 grouped send/recv 等价实现）；
2. tile schedule：通信 tile、计算 tile、rank order 和依赖关系；
3. 合法性检查：producer/consumer、buffer lifetime 和 release/acquire 顺序；
4. 一个可执行或可检查的 lowering artifact；
5. 与基线一致的数值结果和可复现实验记录。

完整边界见 [`SCOPE.md`](SCOPE.md)。

## 当前状态

| 组件 | 状态 | 在 v0.1 中的角色 |
|---|---|---|
| `EDM-IR/` | 已有可运行原型 | trace-derived 观测与 critical-path 分析；保留为后续测量/校准层，不冒充 compiler IR |
| compiler schedule IR | 尚未实现 | 当前主线 |
| sequential lowering | 尚未实现 | 第一条正确性基线 |
| tiled pipeline lowering | 尚未实现 | v0.1 的主要编译产物 |
| H100/A100 实验 | 只有前期 smoke-test/计划 | 完成离线正确性后再租卡验证 |

## 明确不做

v0.1 不包含：

- 多节点、RDMA、GIN；
- 动态 MoE routing、token imbalance、Grouped GEMM；
- vLLM、DeepEP 或 serving runtime 集成；
- 同时实现 Triton、TVM、MLIR、CuTe 多套后端；
- PTX/SASS 或 tinygrad 后端；
- 自动合成任意 collective；
- B200/Blackwell 的未来 SM-initiated Copy Engine 路径；
- 完整 autotuner 或通用 cost model。

这些主题可以作为背景或后续工作，但不得阻塞 v0.1。

## 实现顺序

```text
static All2All -> GEMM graph
  -> schedule IR
  -> sequential schedule
  -> tiled producer/consumer schedule
  -> legality verification
  -> lowering artifact
  -> CPU/synthetic validation
  -> rented H100 validation
```

第一阶段优先建立语义和正确性；硬件性能只在实现闭环之后验证。

## 仓库结构

```text
EDM-IR/       已有 trace-analysis 原型；后续用于测量和校准。
paper/        论文问题与历史研究草稿；不是 v0.1 的任务列表。
papers/       相关论文笔记与 prior-art 矩阵。
experiments/  可复现实验 slice、命令、环境和结果。
ideas/        尚未进入 active scope 的设计草图。
templates/    paper note 与 experiment record 模板。
```

## 必读材料

- [`papers/compiler/communication-compute-fusion.md`](papers/compiler/communication-compute-fusion.md)：直接竞争工作、技术栈和 claim boundary。
- [`papers/baseline-reading-list.md`](papers/baseline-reading-list.md)：按当前主线排序的阅读入口。
- [`ROADMAP.md`](ROADMAP.md)：只包含 v0.1 的执行顺序和停止条件。

## 工作规则

- 一个 commit 只推进一个可验证 milestone。
- 一个实验只回答一个问题；raw data、commands、environment 和 observations 放在同一 experiment slice。
- 文档中的“已支持”必须能指向代码和测试；计划功能必须标为 planned。
- 新想法先进入 `ideas/`，只有替换当前 scope 时才进入 `ROADMAP.md`。
- 在读完并比较直接竞争工作之前，不写论文 novelty claim。
