# Edmir v0.1 Roadmap

本路线图只服务于 [`SCOPE.md`](SCOPE.md) 中冻结的单节点静态 `All2All → GEMM` 编译原型。

## Milestone 0：范围与 prior art

- [x] 固定 v0.1 输入、输出和 non-goals。
- [x] 区分 overlap、wait-logic fusion 和 true in-kernel fusion。
- [x] 建立直接竞争工作矩阵。
- [ ] 为每个可能 claim 找到最近的 prior art 和反例。

停止条件：不能用一句可证伪的话解释 Edmir 与 TileLink、Triton-distributed、AutoOverlap、FLUX 的区别时，不进入论文 claim。

## Milestone 1：Schedule IR

- [ ] 表达 rank、tensor shard、tile 和 placement。
- [ ] 表达 `all_to_all`、`gemm` 和 producer/consumer dependency。
- [ ] 表达 buffer lifetime、signal/barrier 和 memory-order requirement。
- [ ] 提供稳定、可读的 textual dump。
- [ ] 为合法与非法 schedule 添加 unit tests。

停止条件：一个 2-rank、2-tile toy example 可以完全手算并与 dump 对齐。

## Milestone 2：两种 schedule

- [ ] 生成 sequential `All2All → GEMM` schedule。
- [ ] 生成一个固定 tile-size 的 pipelined schedule。
- [ ] 显式记录 rank order 和 tile order。
- [ ] 不做 autotuning；参数由 example 固定。

停止条件：两个 schedule 对同一逻辑图保持相同数据语义。

## Milestone 3：Legality verifier

- [ ] 检查 dependency cycle。
- [ ] 检查 consumer early read。
- [ ] 检查 producer buffer early reuse。
- [ ] 检查 signal/barrier 配对。
- [ ] 检查 acquire/release 或等价可见性约束。

停止条件：verifier 能拒绝 deliberately broken schedules，而不依赖 GPU 才发现错误。

## Milestone 4：Lowering artifact

- [ ] 顺序基线 lowering：NCCL All2All + GEMM（NCCL 2.30+；旧版本使用 grouped send/recv 兼容路径）。
- [ ] pipeline lowering：CUDA/C++ + NCCL Device API LSA，或在 API 不稳定时生成可检查的 code skeleton。
- [ ] CuTe DSL 仅在不扩大 scope 的情况下用于 GEMM 部分。
- [ ] 记录编译器、CUDA、NCCL 和目标架构版本。

停止条件：生成物能通过编译或明确的静态检查；不以伪代码冒充后端完成。

## Milestone 5：验证

- [ ] CPU/synthetic reference 验证数据重排和 GEMM 数值。
- [ ] 在无 GPU 环境运行 IR、verifier 和 codegen tests。
- [ ] 租用单节点 H100 后先记录 topology/P2P，再运行 baseline。
- [ ] 只在 baseline 正确后运行 pipeline variant。
- [ ] 报告 latency、有效带宽、GEMM 时间、等待时间和数值误差。

停止条件：结果可复现，并能判断 pipeline 是 helpful、neutral 还是 harmful。

## Parked work

以下内容保留在 `ideas/`、`paper/` 或历史实验中，但不进入 v0.1：

- serving critical-path / ECR 论文主张；
- vLLM trace、DBO、KV movement；
- DeepEP V2、动态 MoE routing、Grouped GEMM；
- 多节点、GIN/RDMA、ElasticBuffer；
- collective synthesis、完整 cost model 和 autotuner；
- MLIR/TVM/Triton/CuTe 多后端；
- B200/Blackwell SM-initiated Copy Engine。

现有 `EDM-IR/` 作为 parked-but-working trace-analysis component 保留。它可以在后续为 cost model 提供观测数据，但不是当前 compiler IR。
