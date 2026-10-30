# fused 依赖、同步边界与 host 身份审查

日期：2026-10-24。仅本地审查；GPU 作业 0，训练轮次 0。未修改生产源码、默认选择或 10 月 21 日采集包。

结论：当前 P=4 的主要串行约束来自 IR 的资源声明。runtime 对这组程序没有增加 IR 之外的跨 GEMM 先后关系。可以设计受约束的精简依赖版本，但先要补齐输入就绪保证和执行身份封存，再考虑 GPU 对照。不能把链长从 11 减到 4 直接说成 2.75 倍加速，也不能把这些审查结果称为 M512 选择模型已经修复。

## 1. P1：资源声明将只读 peer GEMM 纳入两端通信队列

位置：`EDM-IR/edmir/compiler/model.py:412`；依赖生成在 `dependence.py:119` 附近。

`Op.resources()` 给 fused 操作同时分配源卡 comm、目标卡 comm、目标卡 compute 三个资源。`derive_dependencies()` 随后为同资源上的相邻操作加 queue 边。当前 16 个 GEMM 的输入和权重只读、输出互不重叠，没有 RAW/WAR/WAW 边；31 条边全部来自这一队列策略。

本轮编译了真实 `cuda_backend.cpp`，仅将 CUDA/NCCL 接口替换为已有 CPU 测试替身，记录实际调用。对事件使用 CUDA 12.8 的语义重放：record 保存此前 stream 工作的快照；wait 约束此后工作；之后重录事件不追溯改变旧 wait。依据 [CUDA Stream API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__STREAM.html) 和 [Event API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__EVENT.html)。替身不模拟数值运算、GPU 耗时或网络争用。

| CPU 对照 | IR 最长操作链 | runtime 最长操作链 | IR 以外的传递先后关系 | 每次 wait_event |
|---|---:|---:|---:|---:|
| legacy，正常设备映射 | 11 | 11 | 0 | 74 |
| prepared，正常设备映射 | 11 | 11 | 0 | 74 |
| legacy / prepared，反向设备映射 | 11 | 11 | 0 | 74 |
| 仅保留目标 compute 队列边的测试输入 | 4 | 4 | 0 | 36 |

最后一行只是隔离的 C++ 测试输入：未写回编译器，仍使用原 runtime 的完成事件，保留 28 次 record 和 12 次 stream synchronize。它证明“只更改 IR 输入，现有 runtime 就能表达较短的先后约束”；不证明 GPU 上正确、速度更快或适用于其他 Program。

原始图与 runtime 的传递闭包均有 107 对先后关系。三份既有 fused 时间线，每份 100 次稳态执行，共 32,100 次先后关系检查全部通过。采用同一份 trace 中每个核函数的实测持续时间给该图加权，其最长路径与 GPU GEMM 整体跨度的比值中位数为：M512/q2 91.53%，M512/q4 91.80%，大 N/q2 87.78%。这支持串行链是重要约束；剩余时间不能直接全归 CPU，上述值也不能外推到改图后的性能。

**本次排除的假设：** 在所审查的 unchunked P=4 程序里，没有发现 prepared metadata 引入新依赖，也没有发现 runtime 在原图之外另加跨 GEMM 先后关系。该结论不覆盖 chunked、混合操作或没有每轮全局完成边界的执行。

## 2. P1：首次输入就绪缺少显式保证

位置：`runtime/tools/benchmark_selector.cpp:111–118`、`runtime/src/cuda_backend.cpp:280–285`、`runtime/src/merged_host_backend.cpp:234–242`。

benchmark 从普通 `std::vector` 上传输入和权重，调用 `cudaMemcpy(...HostToDevice)` 后直接 execute。两个 backend 的工作流均使用 `cudaStreamNonBlocking`，没有上传流事件到消费流的连接。

[CUDA 12.8 同步语义](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)允许 pageable H2D 在数据进入 DMA staging 后返回，目的端 DMA 可能尚未完成；[默认流语义](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/stream-sync-behavior.html)也不替非阻塞流建立隐式等待。因此这是一个可定位的就绪合同缺口，不能把 API 名称中的“同步”当作完整证明。

已有三份 fused profile 中，所有通过同步 memcpy API 记录的 H2D 活动（包括初始化）均在首次 GEMM 前完成，最后一次完成提前约 365–557 μs；未观察到本次发生竞态。全部主测也通过计时前后数值检查。这里没有据此作废旧数据，也没有声称已复现错误输出。

修正目标：在计时之外显式完成所选四卡的上传，或建立各上传流到所有消费者的 ready 事件。只同步现有 comm/compute/control 流不足以证明默认上传流完成。未来流水输入必须保留真实 producer→consumer 和 buffer reuse 边。

## 3. P1：执行台账身份没有覆盖资源定义

位置：`edmir/compiler/execution_ledger.py:26–35` 及 `REVIEWED_SOURCE_HASHES`。

临时目录复制源码，唯一改动是将 fused 的 `Op.resources()` 改为目标 compute 资源。执行台账未拒绝，依赖从 31 条变为 12 条，但 `identity_sha256` 完全相同：

`1aba35bb81ce821f362f615204864cc2dbe25682d2d8ad3431af352840d02627`

原因是定义资源规则的 `model.py` 没有进入台账源码清单。依赖图变化后，旧 TimingEvidence 可能仍通过这一层的身份匹配。该问题局限于执行台账：GPU 采集包另有全量源码封存，本轮也核对了 10 个关键文件与被测版本一致。

修正目标：补入资源定义、程序类型和 lowering 所需来源；至少把序列化执行图 / 依赖摘要加入身份。相同源码修订标签不应允许不同执行图共用实测标签。已有记录保留，不覆盖旧摘要。

## 4. 强 host 可比，但尚未接入执行台账

`uniform_workload.cpp` 明确要求同一目标 d 的全部来源分块共享 B_d，且来源、输出布局完整。因此数学上：

`concat_s(A[s,d] × B[d]) = concat_s(A[s,d]) × B[d]`。

这为 merged host 提供语义基础。真实实现将来源块直接传入合并 staging 的对应行区间，保留 12 对远程 send/recv 和 4 次本地复制；把 16 个 M 行 GEMM 改为每卡一个 P×M 行 GEMM。搬运和 GEMM 都在 execute 的计时范围内，输入上传和 cuBLASLt heuristic 查询在计时之外。数值上通过同一 CPU 参考检查，不要求不同实现逐位相同。

该等价只适用于共享目标权重的当前 uniform workload；不能推广为任意专家/路由分组都能合并。它也不是纯粹替换库的消融：聚合粒度、启动数和事件结构同时变化。

四种 host 已有独立身份，但 `execution_ledger()` 当前只支持逐来源 CUTLASS；其他三种均明确拒绝。`assess_candidates()` 还要求 baseline_id 等比较上下文相同。这是尚未完成的适配边界，不应通过把 merged 标签改名为 host 或删除校验来绕过。

后续需要分开：共同工作负载/硬件/输入/计时合同，以及各候选自己的 partition、库算法、同步策略和源码身份。新 host 应有自己的执行台账，再接入同一测量上下文的比较。

## 收敛后的修改顺序和验收条件

1. **先修完整性。** 补上传就绪、资源定义的源码身份和执行图摘要。临时源码改动必须使旧证据失配；初次计算必须明确等待输入完成。
2. **再增加一个受限执行策略。** 仅面向输入/权重不可变、输出不重叠、每个 GEMM workspace 独立、每轮全部完成后才复用/释放的 uniform 工作负载。保留目标 compute 顺序、必要完成事件与真实数据依赖。保持旧策略可复现，给新策略单独身份，不能直接全局修改 `Op.resources()`。
3. **接入强 host 后再改成本表达。** 成本应说明执行图、每个 kernel 的服务成本和并发争用；M/N/K/q/g 的计数不是端到端延迟。依赖图对 q2/q4 都相同，不能单靠本次发现解释 q4 为何更慢，也不能宣布移除队列约束就会修复选择模型。

CPU 验收先覆盖 upload ready、跨来源 producer/consumer、输入复用、workspace 生命周期、反向设备映射和连续迭代。GPU 阶段以后只需小批验证新策略数值正确性及同机未插桩净收益，不需要现在租机或重新训练。

## 复现与交付

- `capture.cpp`：真实 runtime 的 CPU API 调用采集；编译命令及源码哈希见 `build-receipt.json`，构建输出在 `build.log`。
- 五个 `*-*-*.txt`：完整原始调用日志，各含 setup 与两次执行。
- `replay.py` / `replay_test.py`：事件先后关系重放与 6 个测试，`test.log` 保存通过结果。
- `analyze.py` / `results.json`：IR/runtime 对照、旧 GPU 时间线核对、隔离源码变异和 host 支持范围检查。
- `source-verification.json`：被审查的当前关键文件与 GPU 冻结源码一致。

在仓库根目录运行 `python3 experiments/2026-10-24-fused-dependency-audit/replay_test.py` 和 `python3 experiments/2026-10-24-fused-dependency-audit/analyze.py` 可重做本地分析。完整 GPU 数据仍位于原 `2026-10-21-runpod-mechanism-collection/remote-run-01/server-backup/work/`，本目录不复制或改写它。
