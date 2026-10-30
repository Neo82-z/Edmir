# Edmir

Edmir 是一个面向多 GPU 通信—计算流水的编译研究原型。

**最新交付（2026-10-30）**：[源码入口与状态](DELIVERY_README.txt)；
[精简参考论文 TXT](papers/REFERENCES_2026-10-30.txt)。保留开发评分器在 P4/FP16
回放中为 3/41 误选，后续依赖/复用模型仍属诊断，未替换默认策略。
三个真实 MoE 候选的权重和软件缓存已在无 GPU 环境准备完成，计算实例已停机；
BF16 子图接入、真实引擎 A/B 和冻结六形状独立评价尚未完成。
下列较早日期的记录保留其历史语境，不能当作新的验收结果。

**当前工作（2026-10-25）**：已在 4×A100 SXM4 80GB / NV12 完成 390 次独立进程采集、39,000 个计时样本，并将完整归档校验回本地。M256 与大 N 的主要错序重复出现；补充的合并 host 基线明显更强，模型仍未获得自动选择资格。见[本轮验收报告](experiments/2026-10-25-runpod-hour-01/README.md)、[本轮后的修改方案](paper/POST_GPU_REVISION_PLAN.md)、[论文与实验边界](paper/BOUNDED_SELECTION_PLAN.md)、[旧数据检查](experiments/2026-10-25-bounded-candidate-selection/README.md)及[冻结 GPU 包](experiments/2026-10-25-regime-gpu-validation/README.md)。本轮使用冻结的 10/21 执行身份，没有验证后来修复的 runtime；下文早期 v0.1 实现与历史结果不代表新模型已通过。

最新[本地修订结果](experiments/2026-10-25-post-gpu-revision/README.md)：已实现中间量审查和有限组合修订；新表达有改善但未获晋升。单独补入已有 host 标签后，M512 合并 CUTLASS 的留出误差从 36.28% 降到 6.63%。没有新增 GPU 采集，后续转向定向阶段诊断。

**八卡主验证更新**：已实现显式 P4/P8 计数与成本表达，保留论文原始 11 候选和 P8 四形状标定/十九形状评价划分。历史回放中，P8 标定后的平均选择损失从 15.02% 降至 0%，P4→P8 直接迁移仍有 24.13% 最大损失；不代表独立泛化或强基线加速通过。见[完整对照与代码](experiments/2026-10-25-initial-data-replay/README.md)及[论文推进计划](paper/P8_PUBLICATION_PLAN.md)。

**收尾范围**：八卡算子验证之后完成一个真实 MoE 模型的八卡推理对照。权重和引擎先在无卡环境准备；固定一个模型、一个引擎、四个负载点，完成后进入成稿，不再增加研究方向。[真实 MoE 收尾方案](paper/P8_MOE_CLOSEOUT_PLAN.md)记录资产、接入、指标和停止条件；当前尚未创建准备节点或完成真实 GPU MoE 接入。

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
for test in tests/*_test.py; do PYTHONPATH=. python "$test" || exit 1; done

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

- 以八卡 A100 为主验证规模，先核对修复 runtime 的八卡执行账本、数值正确性与基线身份；
- 补齐八卡合并 CUTLASS/cuBLASLt 强基线，四卡子集用于机制与迁移对照；
- 针对 grouped-N 迁移、大 N 和 M512 的残留错误做定向诊断，再执行冻结的新增形状与选择成本比较；
- 7.97% MAPE、95.24% 方向准确率和零选择损失属于 2026-08 的早期 host 调度实验，不能沿用为当前 fused 选择器的验收结论。

## 工作规则

- 一个 commit 只推进一个可验证 milestone；
- 一个实验只回答一个问题，raw data、commands、environment 和 observations 放在同一 experiment slice；
- 文档中的“已支持”必须能指向代码和测试，计划功能必须标为 planned；
- 新想法先进入 `ideas/`，只有替换当前 scope 时才进入 `ROADMAP.md`；
- 在读完并比较直接竞争工作之前，不写论文 novelty claim。
