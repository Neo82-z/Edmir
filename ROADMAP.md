# Edmir v0.1 Roadmap

本路线图只服务于 [`SCOPE.md`](SCOPE.md) 中冻结的单节点静态 `All2All → GEMM` 编译原型。

## 当前阶段：2026-10-25

研究主张与下一轮顺序以[有界候选选择方案](paper/BOUNDED_SELECTION_PLAN.md)为准。以下 Milestone 0–6 保留早期实现记录，其验收结果不覆盖当前候选和新执行身份。

- [x] 明确计数、中间机制、耗时排序、选择价值四层证据。
- [x] 实现 k=1/2/3 的只读候选评估及强 host、旧模型、固定顺序、随机集合对照。
- [x] 冻结 GPU 执行包；保持本轮 57 个主配置和 4 个预热诊断。
- [ ] 新环境先复测全部九对错误及关键 host，随后执行已安排的交叉点。
- [ ] 对可复现但仍无法归因的错误，安排独立于主计时的必要机制诊断。
- [ ] 补足强基线覆盖后，再冻结独立选择评估及完整一次性成本核算。

停止条件：若固定强 host 消除可用收益，或简单旧模型取得同等效果，收缩论文贡献；不通过删除基线或重命名开发数据宣称泛化。尚未租服务器。

## Milestone 0：范围与 prior art

- [x] 固定 v0.1 输入、输出和 non-goals。
- [x] 区分 overlap、wait-logic fusion 和 true in-kernel fusion。
- [x] 建立直接竞争工作矩阵。
- [ ] 为每个可能 claim 找到最近的 prior art 和反例。

停止条件：不能用一句可证伪的话解释 Edmir 与 TileLink、Triton-distributed、AutoOverlap、FLUX 的区别时，不进入论文 claim。

## Milestone 1：Schedule IR

- [x] 用 typed Dataflow IR 表达 rank-local tensor shape、distribution、`all_to_all` 和 `gemm`。
- [x] 用唯一低层 `Program(Buffer, Op)` 表达 rank、tile、queue 和 placement。
- [x] 从 buffer access 自动派生 producer/consumer、lifetime、barrier 和 acquire/release dependency。
- [x] 提供稳定 JSON schema 与 `edmir inspect --dependencies` dump。
- [x] 为合法与非法 Dataflow/Program 添加 unit tests。

停止条件：一个 2-rank、2-tile toy example 可以完全手算并与 dump 对齐。

## Milestone 2：两种 schedule

- [x] 生成 sequential `All2All → GEMM` Program。
- [x] 生成有限 chunk/buffer grid 的 pipelined Program。
- [x] 显式记录 cyclic rank order、tile order 和 staging slot。
- [x] 用可替换 calibration profile 做有限候选选择；不做在线 autotuning。

停止条件：两个 schedule 对同一逻辑图保持相同数据语义。

## Milestone 3：Legality verifier

- [x] 检查 dependency cycle。
- [x] 检查 consumer early read。
- [x] 检查 producer buffer early reuse。
- [x] 检查 sequential readiness barrier 覆盖。
- [x] 以统一 RAW/WAR/WAW dependency 表达 acquire/release 可见性约束。

停止条件：verifier 能拒绝 deliberately broken schedules，而不依赖 GPU 才发现错误。

## Milestone 4：Lowering artifact

- [x] 顺序基线 lowering：共享 ProgramView ABI、grouped NCCL send/recv 与 cuBLAS GEMM。
- [x] pipeline lowering：先以 host-launched CUDA/C++ + NCCL send/recv + cuBLAS 建立可测 baseline；不把它表述为 Device API fusion。
- [x] 建立 Frontend → Dataflow passes → Program → Backend 的完整 pipeline。
- [x] 生成含 Buffer/Op/Access/Dependency 的 C++20 静态 descriptor，并通过真实编译检查。
- [x] 实现 host validator、rank-device mapping、CUDA stream/event lowering 和 generated-program smoke runner。
- [ ] CuTe DSL 仅在不扩大 scope 的情况下用于 GEMM 部分。
- [x] 在实验归档中记录编译器、CUDA、NCCL 和目标架构版本。

停止条件：生成物能通过编译或明确的静态检查；不以伪代码冒充后端完成。

## Milestone 5：验证

- [x] CPU reference 与 GPU runtime 做确定性非零数值对照，并保留 zero-input smoke。
- [x] 在无 GPU 环境运行 frontend、IR、pass、verifier、simulator 和 C++ codegen tests。
- [x] 在单节点 4x A100 SXM 上先记录 topology/P2P，再运行 baseline。
- [x] 只在 baseline 正确后运行 pipeline variant。
- [x] 报告 latency、P2P/NCCL 带宽、GEMM probe 和数值误差；等待时间尚待下一轮模型实验分解。

停止条件：结果可复现，并能判断 pipeline 是 helpful、neutral 还是 harmful。

## Milestone 6：有界 cost model

- [x] pairwise NCCL 曲线进入 profile，而不是只保留日志。
- [x] 添加与 runtime 完全一致的 dtype/shape-specific cuBLAS GEMM benchmark。
- [x] 从 Program 派生 dispatch、event record/wait 和 buffer-reuse 特征。
- [x] 实现非负 runtime residual fit、train/heldout MAPE、方向准确率和 selection regret。
- [x] 冻结 27-row train / 23-row heldout 实验及断点续跑 runner。
- [x] 在新的 4x A100 SXM 实例执行冻结实验并通过 heldout 决策门槛：23 个 held-out 候选的 MAPE 为 7.97%，policy direction accuracy 为 95.24%，selection regret 为 0。

停止条件：heldout MAPE ≤ 15%，方向准确率 ≥ 85%，平均 selection regret ≤ 5%；否则不增加新 backend。

## Parked work

以下内容保留在 `ideas/`、`paper/` 或历史实验中，但不进入 v0.1：

- serving critical-path / ECR 论文主张；
- vLLM trace、DBO、KV movement；
- DeepEP V2、动态 MoE routing、Grouped GEMM；
- 多节点、GIN/RDMA、ElasticBuffer；
- collective synthesis、跨算子/跨拓扑的通用 cost model 和在线 autotuner；
- MLIR/TVM/Triton/CuTe 多后端；
- B200/Blackwell SM-initiated Copy Engine。

现有 `EDM-IR/` 同时包含 parked trace-analysis component 和 active compiler。两者只在低层 EventGraph/反事实分析处复用语义；typed Dataflow 与 `Program(Buffer, Op)` 才是当前 compiler IR。
