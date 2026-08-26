# Edmir

Edmir 是一个面向多 GPU 通信—计算流水的编译研究原型。

当前仓库只推进一个可验证问题：

> 给定单节点拓扑和一个静态 `All2All -> GEMM` 数据流，能否生成正确的顺序执行与 tile-level 流水 Program，并解释它们在什么条件下值得使用？

仓库同时从 trace 依赖图计算通信/数据移动的**反事实关键路径暴露量**。核心问题不是“trace 里通信占了多久”，也不是“能否把通信和 GEMM 融合”，而是：

> 如果某类成本变成零，端到端关键路径实际缩短多少；在合法性、拓扑和额外开销约束下，编译器是否应该采用流水 Program？

`All2All -> GEMM` 是 v0.1 的研究载体，不是“首次融合该算子组合”的创新声明。FLUX、TileLink、Triton-distributed、AutoOverlap、DeepEP/NCCL EP 等工作已经覆盖相邻或部分重叠的设计空间。完整边界见 [`SCOPE.md`](SCOPE.md)。

## v0.1 范围

固定输入：

- 单节点、多 GPU；
- 静态 shape 和静态 rank 数；
- 等尺寸、对称布局的 All2All；
- 一个消费通信结果的 GEMM；
- 已知的 P2P/NVLink/PCIe 拓扑描述；
- 从真实测量获得的通信与计算校准参数。

固定输出：

1. 顺序基线：全部通信完成后执行 GEMM；
2. tile-level 流水 Program：只保存 Buffer 和按序 Op；
3. 自动派生的 queue order、data-ready、release/acquire 与 buffer-reuse 依赖；
4. producer/consumer、buffer lifetime、覆盖率和同步合法性检查；
5. 候选 Program 的成本预测、选择理由和可复现实验记录。

## 当前实现

仓库包含两个分离但共享图语义的部分：

1. **Trace analysis IR**：导入观测事件和可审计依赖，报告图覆盖率、诊断路径统计，以及通过重新计算最长路径得到的反事实通信暴露量和数据移动暴露量。
2. **Program compiler IR**：针对单节点、静态形状、等长分块的 `All2All -> GEMM`，构造 sequential/pipelined Program，派生依赖，验证合法性，并用校准成本模拟和选择候选。

| 组件 | 状态 | v0.1 角色 |
|---|---|---|
| trace analysis IR | 已实现并有测试 | 测量、诊断和校准层 |
| `Buffer + Op + Program` IR | 已实现并有测试 | 唯一 compiler IR |
| typed Dataflow IR / frontend / passes | 已实现并有测试 | tensor semantics 与 pattern lowering |
| sequential / pipelined Program 构造 | 已实现并有测试 | 正确性基线和主要候选 |
| dependency derivation / verifier / simulator | 已实现并有测试 | 共享语义与 CPU reference model |
| measured pairwise NCCL curves | 已实现解析与 simulator 接入 | 逐 `(src,dst,size)` 通信成本 |
| JSON / C++20 descriptor backend | 已实现并有编译测试 | runtime ABI 与可审计 artifact |
| C++ CUDA/NCCL/cuBLAS execution runtime | 已实现；host ABI 已测试，GPU 路径待上机验证 | 最小可执行 backend |
| A800/H100 实验 | 只有前期 smoke test 和预注册方案 | backend 闭环后再运行 |

当前实现输出并模拟可检查的 Program，并提供解释该 Program 的最小 C++ GPU runtime；**尚未**把未上机验证的 CUDA 路径或性能数据写成实验结论。

## 编译路径

```text
static All2All -> GEMM problem
  -> Program(Buffer, Op)
  -> derive queue + RAW/WAR/WAW dependencies
  -> legality verification
  -> calibrated simulation and candidate selection
  -> EventGraph / counterfactual exposure analysis
  -> C++ ProgramView ABI
  -> CUDA streams/events + NCCL send/recv + cuBLAS GEMM
```

Program 不序列化第二份 edge list。verifier 和 simulator 调用同一个依赖推导器，避免 schedule、验证器和执行模型分别维护相互漂移的语义。

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

## 快速验证

```bash
cd EDM-IR
python -m unittest discover -s tests -p '*_test.py' -v

python -m edmir compile examples/a2a_gemm_problem.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --out /tmp/edmir-program.json
python -m edmir verify /tmp/edmir-program.json
python -m edmir inspect /tmp/edmir-program.json --dependencies
python -m edmir simulate /tmp/edmir-program.json \
  --profile examples/synthetic_profile.json
```

示例 profile 和 topology 仅用于 CPU 测试，不是 A800 实测数据。任何性能结论都必须绑定实际 topology snapshot、校准数据、原始日志和代码版本。4x A100 SXM 的已归档冒烟实验位于 [`experiments/2026-08-25-runpod-4xa100-sxm`](experiments/2026-08-25-runpod-4xa100-sxm)，冻结的训练/留出方案位于 [`experiments/2026-08-cost-model-validation`](experiments/2026-08-cost-model-validation)，完成后的 50-candidate 原始数据和派生摘要位于 [`experiments/2026-08-26-runpod-4xa100-cost-model`](experiments/2026-08-26-runpod-4xa100-cost-model)。

## 仓库结构

```text
EDM-IR/       分析 IR、Program compiler、验证器和 CPU 模拟器
experiments/  预注册实验、命令、环境快照和原始数据
paper/        研究边界、定位和早期笔记；不是论文正文的第二来源
papers/       相关工作阅读笔记和 prior-art 矩阵
ideas/        尚未进入 active scope 的设计草图
templates/    paper note 与 experiment record 模板
```

论文正文位于相邻仓库 `../llm-serving-mlsys-latex`。

## 必读材料

- [`papers/compiler/communication-compute-fusion.md`](papers/compiler/communication-compute-fusion.md)：直接竞争工作、技术栈和 claim boundary；
- [`papers/baseline-reading-list.md`](papers/baseline-reading-list.md)：按当前主线排序的阅读入口；
- [`ROADMAP.md`](ROADMAP.md)：v0.1 的执行顺序和停止条件；
- [`experiments/2026-08-a800-a2a-gemm/README.md`](experiments/2026-08-a800-a2a-gemm/README.md)：预注册实验矩阵。
- [`experiments/2026-08-4x4090-nccl-baseline/README.md`](experiments/2026-08-4x4090-nccl-baseline/README.md)：空闲 4x4090 的 NCCL 排布和 compiler calibration。

## 下一阶段

- 已在 4x A100 SXM 上完成 6-pair NCCL、精确 cuBLAS/local-copy 曲线和冻结的 27-row train / 23-row heldout cost-model sweep；
- 当前 heldout MAPE 为 7.97%，policy direction accuracy 为 95.24%，selection regret 为 0；
- 保留 NCCL+cuBLAS 作为 reference backend，下一步加入一个受控的 CuTe/CUTLASS lowering，而不是扩大 IR；
- 只在预注册表格填入可复现数据，无法匹配的机制或 baseline 明确记为不适用。

## 工作规则

- 一个 commit 只推进一个可验证 milestone；
- 一个实验只回答一个问题，raw data、commands、environment 和 observations 放在同一 experiment slice；
- 文档中的“已支持”必须能指向代码和测试，计划功能必须标为 planned；
- 新想法先进入 `ideas/`，只有替换当前 scope 时才进入 `ROADMAP.md`；
- 在读完并比较直接竞争工作之前，不写论文 novelty claim。
