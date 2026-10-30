# 四卡 A100 首轮采集结果 — 2026-10-21

采集结束，完整备份已下载并逐文件校验。后续分析不依赖服务器继续运行。本轮没有训练模型、修改 GPU kernel、改变执行依赖或更新编译器默认选择；现有误选不能宣称已经修复。

## 数据入口

| 文件 / 目录 | 内容 |
|---|---|
| `backup-verification.json` | 完整备份 SHA-256、2,571 个文件校验和本地复算结果 |
| `completed-run-01.tar.gz` | 完整服务器备份，123,506,669 字节；包括原始结果、二进制、构建日志、NCCL 动态库、时间线及控制脚本 |
| `server-backup/work/` | 已解压的完整工作目录 |
| `results/{anchors,mechanisms,geometry}-01/` | 提前独立回传的三阶段主测原始结果 |
| `validated-analysis.json` | 主测结果的本地身份检查、复算、配对差分及波动标记 |
| `server-backup/work/short-host-diagnostic-01/` | 短 K 的 20/200 次预热对照，单独保存，不能替换主测 |
| `server-backup/work/trace-*/` | 四份 Nsight Systems 报告、SQLite、完整 stdout 和 profiler 身份声明 |
| `server-backup/work/trace-analysis.json` | 时间线的受限稳态区间分析；与本地复算一致 |
| `server-backup/work/compiled-resources-02/` | 实际二进制资源表和 CUTLASS 类型大小 |
| `queue-dependency-audit.json` | 冻结源码中 fused 队列依赖的本地审计 |

完整备份 SHA-256：`0c28679a0a0052131c46ef86718f7e0f91cb743b19f0abcd031aae813f8f4856`。

## 已完成范围

- `anchors`：30 个进程，3,000 个计时样本。
- `mechanisms`：177 个进程，17,700 个计时样本。
- `geometry`：75 个进程，7,500 个计时样本。
- 额外短 K 预热诊断：24 个进程，2,400 个计时样本。
- profiler 诊断：4 个配置，各一次，全部与主标签隔离。

主测每个阶段配置有 3 个新进程区组；每进程预热 20 次、计时 100 次，保留全部样本。进程内迭代不能当成独立重复。计时范围为 host `steady_clock` 包围 `execute()+synchronize()`。数据准备、输入拷贝和数值验证在计时之外。

所有测量通过计时前后的全输出检查。数值合同为 FP16 输入/输出、FP32 CPU 参考；检查最大绝对误差及 `abs(error)/max(abs(reference),1)`，无非有限输出，要求最大绝对误差或上述归一化最大误差不超过 0.02。这不等于逐位相同，也不涵盖任意输入分布。

硬件实际为 4×A100-SXM4-80GB、四卡两两 NV12、P2P 全通、每卡 108 SM。环境为 CUDA 12.8、驱动 595.91.07、cuBLAS 12.8.4.1、固定源码构建的 NCCL 2.28.9 和 CUTLASS。GPU UUID、动态库、输入、源码及二进制身份均保存在 context/build receipt 中。原始包不改写；实际执行的 r01 包摘要为 `54aff5ff71ff1dbab33a90348e84ec9d3bc1ea8b755e11d30c6ebd777bd6edc5`。

## 本轮直接支持的结论

形状为 `(P,M,K,N)`，M 是每个源—目标分块行数。下表取 mechanisms 阶段三个进程中位数的中位数，单位 μs；g1/g2 和 prepared/merged 身份不能混用。

| 形状 | 逐来源 CUTLASS host / prepared | q2 g1 / prepared | q4 g1 / prepared | q4 g2 / prepared | merged cuBLASLt |
|---|---:|---:|---:|---:|---:|
| M64 K4096 N2048 | 582.05 | 718.09 | 718.35 | 599.77 | 318.76 |
| M512 K4096 N512 | 1076.49 | 1006.05 | 1047.65 | 725.40 | 637.68 |

1. **M512 的 q4/g1 误选复现。** anchors 三个区组中，q4 比 q2 慢 42.76–46.73 μs；mechanisms 的 prepared 路径仍慢 38.02–44.49 μs。预计算 metadata 没有消除这一排序。不能把 q4/g1 与明显更快的 q4/g2 合并成同一个候选标签。
2. **大 N 首先是 host/fused 判断问题。** prepared q2 在三个区组均比旧 host 慢 131.78–140.44 μs。q2/g1 与 q4/g1 的区组差分却跨零，不能认定它们有可靠的固定顺序。新旧协议、源码身份并不完全相同；旧数据只能用于排序对照，不能声称严格复现实验的绝对耗时。
3. **强 host 改变了研究结论。** 五个重点形状中，主测的最快已测候选均为 merged cuBLASLt。merged 同时改变 GEMM 聚合规模、启动次数和执行结构，因此不能把全部收益归给库算子。这里只覆盖已测候选，不能宣称全局最优；短 K 的部分 host 数据还存在预热敏感性。
4. **grouped-N 不能只按 M 设阈值。** 新交叉形状 M192、K3072、N512 的 q2 从 g1 的 552.16 μs 变为 g2 的 633.03 μs；同样 M/K、N1536 时却从 925.27 μs 降到 671.09 μs。q、M、N 的交互必须保留。
5. **短 K 的 20 次预热不足以稳定所有 host 配置。** 额外诊断中，merged CUTLASS 的三个进程中位数最大/最小比从 1.537 降至 1.022，prepared host 从 1.113 降至 1.014。原数据全部保留；这是调整后续预热协议的证据，不是删除样本或宣称普遍稳态的依据。

主测存在 9 个“末 20 样本 / 首 20 样本中位数超出 [0.9,1.1]”标记：mechanisms 8 个、geometry 1 个。跨进程 max/min>1.10 的标记仅出现在 mechanisms 的短 K 三种 host 配置。具体记录在 `validated-analysis.json`；阈值仅用于复查，不是统计显著性检验。全部同配置的 GEMM 算法身份在三个区组内一致。

## 时间线与源码提供的线索

四份 trace 都核对到每卡 484 个目标 GEMM：1 次正确性执行、20 次预热、100 次计时，每次每卡 4 个 GEMM。分析去掉每卡前 84 个 GEMM，并检查 compute stream 的串行顺序。没有迭代 NVTX 标记，API 窗口只是 GEMM 起止范围，不能当成精确的 CPU 迭代边界。

- M512 q4 的单卡四个 GEMM 累计时长中位数约 304–306 μs，q2 约 294–298 μs；执行层也出现了差距。但这不能把未插桩端到端差值全部归因于 semaphore 或计算。
- 当前 fused IR 的 16 个 op 有 31 条依赖，全部来自队列约束，最长单位权重链包含 11 个 op。源输入只读，输出互不重叠；`Op.resources()` 仍把完整 fused GEMM 声明为占用两端 comm 资源，runtime 再把完成事件传到源卡通信流。时间线中各卡 GEMM 跨度远大于各自内核耗时之和，与受执行依赖约束的时序一致。**是否能安全放宽这些约束，是下一步优先审查项；尚未实现或实测放宽后的版本。** 单位权重路径不是实际延迟预测，不能直接换算加速比。
- g1 的静态资源为 128 线程、36 KiB 动态共享内存；g2 为 256 线程、60 KiB。g2 同时减少 tile 数并增加每 CTA 资源需求。这些是编译资源和启动参数，不是 achieved occupancy。
- profiler 明显扰动 host：大 N host 的 profiled wall 中位数约 951.99 μs，q2 为 754.65 μs，甚至反转未插桩主测顺序。**不得用这四次 profile 的 wall 时间做性能标签，不得将其 GPU 时间从未插桩 wall 时间中相减。** 本轮未采物理 HBM/NVLink 字节或 semaphore stall 硬件计数。

## 工程变更与边界

服务器驱动的拓扑文本带 ANSI 控制码，原预检查无法正确匹配 NV12。本轮只修正解析器，保留原始文本并增加回归测试；7 个相关包测试通过。r01 只替换采集工具及对应封装身份，GPU 实现、输入、计时策略不变。

静态资源辅助脚本首版因给命令包装器传入不支持的 `stdin` 参数而失败；修正后在独立 `compiled-resources-02` 目录完成。失败日志保留，未影响主测或改写结果。

本轮数据已经被用于检查机制，属于开发数据。没有新增训练轮次，没有独立模型泛化验收，也没有自动放行编译器选择。后续先在本地分析队列依赖、明确强 host 的执行身份和成本表达，再决定是否需要下一次 GPU 对照。
