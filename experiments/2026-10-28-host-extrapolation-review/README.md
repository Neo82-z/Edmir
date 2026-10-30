# Host 外推路径与计时口径复核

2026-10-28。本轮新增可复核的评分解释和独立主机阶段计时程序。没有训练、改系数、改选择策略或启动 GPU。原 41 形状/346 候选评分逐值不变，仍为 **3/41 误选、最坏选择损失 1.5801%**；M512 整组留出仍错 1/3。本轮不宣称预测可靠性验收通过。

## Host 为什么会低估

新增 [prediction_explanation.py](../../EDM-IR/edmir/compiler/prediction_explanation.py) 逐候选输出实际使用的底座、各项贡献、K 锚点调整、host 分段曲线是否启用，以及退回原底座的原因。解释函数不读取查询耗时，所有贡献必须精确重建当前预测。没有改变默认模型，也不把经验贡献命名为真实通信/计算阶段耗时。

当前四折回放中，41 个 host 判断有 **20 个使用 tile 曲线，21 个退回原底座**。不能把模型配置名 `host_tiles_fused_residual` 理解为所有形状都使用 tile 曲线。

在全部 M512 留出的 `(4,512,4096,512)` 上，没有同 M/N 的 K 锚点，保护规则禁用新的 host 曲线，实际使用：

```
T_host(K=4096,N=512) = 634.65767 + 67.11369 × log2(ceil(M/64))  [μs]
```

该底座每次 M tile 数翻倍只增加相同的 67.11369 μs。为隔离这一表达，下面比较同一个原底座，**没有把 M256 实际启用的 tile 曲线混入增量**：

| M | 原底座 μs | 实测 host μs | 该查询实际使用的路径 |
|---|---:|---:|---|
| 256 | 768.885 | 824.800 | tile 曲线，预测 814.769 μs |
| 512 | 835.999 | 1079.900 | 原底座，预测 835.999 μs |

原底座误差可精确拆成：

```
M512 误差 = M256 原底座误差 + M256→M512 增量误差
−243.901 = −55.915 + (67.114 − 255.100)  [μs]
```

所以问题包含原底座已经存在的误差、过小的跨 M 增长，以及支持不足时回到这条旧表达。它不是优化器没跑够，也不是将所有候选同时平移就能消除的误差。

本轮也计算了同一份已保存 tile 曲线在 M512 的**未启用值**：1053.439 μs。它明显接近实测，但仍比模型预测的 q4=1054.888 μs、q2=1056.826 μs 小，启用它仍会选 host。因此没有取消边界保护；此前 `global_K_range` 的失败也已保存在 [conditional-curves/run-03](../2026-10-28-conditional-curves/README.md)。未启用值只是诊断，未用于重新选择或评价一个新模型。

## 短 K：核对了原始记录，但仍不能归因到 CPU 或设备

只读核对五个已有 K512 开发形状的 direct/q2，共 10 条原始日志。CSV 来源哈希、原始日志统计和评分标签一致。每组两候选都有相同的外围计数：16 个操作、28 次 event record、62 次 event wait，相同预热和计时次数；CTA 数随 q 变化。**相同调用次数不等于相同调用耗时，也不能排除阻塞或执行依赖的影响。** 本轮没有发现这些行被脚本错贴为另一候选的证据。

同时检查了归档记录提交 `f584d7d` 的 benchmark 源码与当前源码：

| 项目 | 旧归档源码 | 当前封存采集包 |
|---|---|---|
| 外围计时 | `execute + synchronize` | `execute + synchronize` |
| 输入 | 外部 input/weight 全零 | 固定随机 FP16 输入及 CPU FP32 参考 |
| 300 次的中位数 | 排序后第 150 个值 | 第 150、151 个值的平均 |
| 留存样本 | 汇总值，输出保留四位小数 ms | 原始样本保留 |
| 主机提交与完成等待 | 未分别记录 | 原采集程序也未分别记录 |

以上旧口径来自记录提交的源码；归档没有可执行文件，不能把源码核对称为二进制同一性证明。它们必须作为不同测量身份，不能直接拼接绝对时间。

四位小数 ms 的显示舍入至多给两候选差值带来约 0.1 μs 误差，不足以单独翻转这些 1.8 μs 以上的旧差值。没有逐样本数据则无法重算统一中位数，也不能构造配对置信区间。没有用“可能有噪声”撤销任何模型误选。

此前索引的 profile 集合没有 K512，也没有 M512 host；M512 q2/q4 的 prepared 插桩时间不能补成旧 legacy host 的缺失阶段标签。

## 新增独立阶段计时工具

- [host_phase_timing.hpp](../../EDM-IR/runtime/include/edmir/runtime/host_phase_timing.hpp)：在独立诊断窗口记录 `execute` 前、`execute` 后、`synchronize` 后三个主机时间戳。每次严格满足 `submit + completion_wait = wall`。
- [diagnose_selector_phases.cpp](../../EDM-IR/runtime/tools/diagnose_selector_phases.cpp)：单独可执行程序，使用同类 runtime、输入及正确性判据，计时前后检查全部输出。最多 64 个诊断样本，失败不输出成功记录。只支持当前 FP16、逐源 CUTLASS、resource_queues_v1 路径。
- [summarize_host_phases.py](../../EDM-IR/tools/summarize_host_phases.py)：严格验证诊断身份、整数时间和样本上限；分别计算各阶段及 wall 的均值和中位数，不相加阶段中位数。拒绝将普通计时行当诊断行。

提交阶段可能包含阻塞 API 和同时发生的设备工作；完成等待只是剩余主机等待。**两者都不能直接当作纯 CPU 时间或 GPU 总耗时，更不能标成独立通信/计算服务。** 输出固定为 `host_phase_diagnostic`、`training_eligible=false`、`device_service_us=null`，与主实验标签分开。它不能替代设备时间线或主循环/epilogue 的独立标定。

主计时程序没有加时间戳，旧封存包没有改动。新增程序完成 CPU 测试及针对已生成程序头的 C++20 语法检查，**尚未通过 CUDA 链接或 GPU 执行验证，也尚未加入旧封存上机包**。真正采集前需要把新增源码单独封存并记录二进制、输入、库和设备身份；不能将本地准备说成 GPU 已验证。

本目录 CMake 默认仅编译 CPU 测试。GPU 构建需显式设置 `EDMIR_PHASE_CUDA=ON`、`EDMIR_PHASE_PROGRAM`（生成程序目录）、固定 CUTLASS/NCCL 路径，沿用 CUDA 12.8/CMake≥3.30 环境。完成构建并核对身份后，单次诊断形式为：

```sh
timeout --kill-after=5 90 ./edmir_diagnose_phases FAMILY INPUT_FP16 REFERENCE_FP32 30 16 prepared > phase.json
python3 EDM-IR/tools/summarize_host_phases.py --input phase.json --out phase-summary.json
```

只对稳定复现的失败点运行这一独立窗口；已有原始输入和参考可复用，但必须核对哈希。命令不是本轮已执行记录，也没有授权或触发租机。

## 本轮验收

评分路径、原始日志和增量分解见 [verified/results.json](verified/results.json)。[verification.json](verification.json) 记录测试及哈希。冻结六形状只核对保护哈希，未解析、评分或训练；本轮没有新模型成绩。

接下来需要的是同身份的阶段证据。它将决定 host 的 M 扩展是否需要通信/计算分开表达，以及短 K 的方向变化出现于设备执行还是主机可见等待。现阶段继续改系数或取消保护，不能补上这些缺失观测。
