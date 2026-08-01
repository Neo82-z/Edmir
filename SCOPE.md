# Edmir v0.1 Scope Contract

Status: **frozen for v0.1**.

修改本文件中的核心问题、输入范围或停止条件，等同于重新立项；普通实现细节不应扩大范围。

## 核心问题

```text
Given:
  a fixed single-node GPU topology,
  a static equal-sized All2All,
  and a dependent GEMM,

can Edmir construct and verify a tile-level communication-computation
schedule, then lower it to an inspectable implementation artifact?
```

这里的“融合”统一指 communication 与 computation 在 device-side tile 粒度上协作。必须在结果中区分：

1. 多 Kernel、跨 stream 的 operator overlap；
2. GEMM 内只有 wait/signal、数据传输仍由 host/CE/另一 Kernel 发起；
3. 通信与计算真正存在于同一个 device Kernel。

不得把三者混写成同一种 fusion。

## 输入约束

- `world_size` 在编译时已知；
- 每个 rank 的输入 shape、dtype 和布局一致；
- 每个 destination 收到等尺寸分片；
- GEMM 直接消费 All2All 重排后的矩阵；
- 不包含 top-k routing、ragged tokens、capacity factor 或动态 expert placement；
- 第一目标拓扑是单节点 H100 SXM/NVSwitch；A100 只作为传统路径和 P2P 对照。

## v0.1 编译对象

最小逻辑图：

```text
input shards
  -> all_to_all
  -> assembled matrix B
  -> gemm(A, B)
  -> output C
```

最小 schedule 必须表达：

- communication tile 和 GEMM tile；
- tile 对应的 source/destination rank；
- producer/consumer dependency；
- buffer ownership、readiness 和 reuse；
- rank/tile execution order；
- acquire/release 或等价同步语义；
- baseline 与 pipelined variant。

## 后端边界

v0.1 只要求两条 lowering 路径：

1. **Sequential baseline**：NCCL All2All + 独立 GEMM（NCCL 2.30+；旧版本以 grouped send/recv 等价实现）；
2. **Experimental pipeline**：以 CUDA/C++ 和 NCCL Device API 的单节点 LSA 能力为首选实现目标。

CuTe DSL 可以用于 GEMM/代码生成实验，但不是完成 v0.1 的硬依赖。Triton-distributed、TileLink 和 AutoOverlap 是必须比较的 prior art，不在本仓库重新实现。

GIN、RDMA、Multimem、Copy Engine、跨节点和多后端 portability 全部延后。

## 正确性义务

生成 schedule 至少要验证：

- 每个 consumer tile 只在所有 producer tile ready 后读取；
- 不读取未到达或未发布的远端数据；
- buffer 不会在最后一次消费前被覆盖；
- rank 间 signal/barrier 次数匹配；
- schedule dependency graph 无环；
- sequential 与 pipelined 版本数值等价。

## 完成定义

只有满足以下条件，v0.1 才算完成：

- 一个固定 `All2All → GEMM` example 能构造两种 schedule；
- schedule dump 足以人工检查 rank、tile 和 dependency；
- verifier 能拒绝至少三类错误：early read、early reuse、dependency cycle；
- lowering artifact 可生成并通过静态/编译检查；
- CPU 或 synthetic reference 验证数值和依赖；
- 在可用的单节点多卡环境上完成一次 baseline/pipeline 对照；
- README 中所有“已实现”声明均有测试或实验记录支持。

## 论文 claim boundary

不能声称：

- 首次融合 All2All 和 GEMM；
- 首次使用 tile-level communication-computation overlap；
- 首次把通信原语加入 Triton/CuTe 类 DSL；
- 首次进行 topology-aware collective synthesis。

在完成 prior-art 对比和实现后，可能讨论但尚未成立的方向是：

> NCCL Device API mechanism-aware lowering and legality verification for fused multi-GPU schedules.

该句目前是研究假设，不是仓库结论。
