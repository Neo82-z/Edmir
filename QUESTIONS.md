# Research Questions

## Active in v0.1

1. 静态、等尺寸 `All2All → GEMM` 的最小 schedule IR 需要哪些语义？
2. 如何在编译期验证 tile readiness、buffer lifetime 和跨 rank memory ordering？
3. sequential schedule 与 tile-level pipeline 的语义等价条件是什么？
4. NCCL Device API LSA 能否作为稳定的单节点 lowering target？
5. Edmir 与 TileLink、Triton-distributed、AutoOverlap、FLUX 的可证伪差异是什么？

## Required distinctions

- operator overlap 与 single-kernel fusion 有何不同？
- wait/signal 被放入 GEMM 是否等于数据传输也被融合？
- `GEMM → ReduceScatter` 中融合的 All2All 子步骤，是否能代表任意 `All2All → GEMM`？
- 静态 equal-size All2All 的结论能否外推到动态 MoE dispatch？默认答案是不能。

## Parked after v0.1

- 动态 MoE routing、token imbalance 和 Grouped GEMM；
- LSA、Multimem、GIN、Copy Engine 之间的 mechanism selection；
- topology-aware cost model 和 autotuning；
- 多节点 RDMA；
- vLLM serving、KV movement、DBO 和 request critical path；
- MLIR/TVM/Triton/CuTe 多后端 portability。

Parked 问题只有在替换 [`SCOPE.md`](SCOPE.md) 后才能进入 active roadmap。
