# 小范围数据清理、成本分解与预测

本次仅在本地分析。原始数据未改写；没有 GPU 实测、神经网络训练或超参数搜索。少量非负系数标定仍属于从数据估参。
已看过的数据用于开发，按整个形状留出可以检查泄漏和插值失败，但不是独立泛化验证。

## 数据

共 306 个进程、30,600 次计时；主测 282 个进程，预热诊断 24 个进程分开保留。
69 个 prepared/merged 主配置中 63 个满足既定稳定性规则；6 个完整配置隔离标记，不删除任何慢样本。
192 个封存文件、136 个源文件、20 个输入/参考二进制哈希核对通过；全部原始 stdout、计时统计、实现/算法身份、输入身份和区组完整性重验通过。
另核验了 63 个编译后二进制及归档 NCCL 动态库。两个 NCCL 头文件只有原构建记录的哈希，本地归档未包含它们；不声称完成全部外部构建依赖的重新验证。
三个进程的中位数再取中位数为标签。100 次迭代不是 100 个独立重复。短 K 的 200 次预热诊断不替换主测标签。

## 模型

fused 共享 12 个固定成本项。基础项为固定项、K/q 分区深度、grouped-N × 分区深度、CTA 数、唯一 A 大小、A 的逻辑重读量、split-K 额外串行输出读写量。
本轮细化项依次加入动态共享内存 opt-in 调用、K 分片 A 工作集、B 逻辑重读、可用 SM 分布缺口和每输出 tile 的串行累加边界。寄存器/共享内存/warp 变化与 g 绑定，不分别硬拟合物理成本。
四种 host 各自标定固定项、远端 A 字节、逻辑计算量；不把 merged 的 PM 行 GEMM 当成 per-source 的 M 行 GEMM。第二轮仅对已知 CUTLASS 的两种 host 增加 K/32 循环深度，修复相同 MK/MKN 却不同 K 的表示冲突；没有猜测 cuBLAS 内部 tile。
上述字节项是逻辑计数。系数解释端到端时间关联，不能当作实测带宽、占用率或信号量等待时间。固定 1e-6 数值正则；本轮每个模型最多枚举 4096 个非负解面，检查 KKT。
所有候选共享同一形状留出边界。强 host 与 fused 分别评估，也单独检查 fused 内部排序。平局按候选 ID，遍历顺序不改变选择。

## 留出结果

| 固定模型 | 按形状平均绝对相对误差 | 最大候选误差 | 错误方向/候选对 | 错选损失 >10% 的候选对 |
|---|---:|---:|---:|---:|
| service_detail | 6.7% | 27.0% | 11/191 | 2 |
| row_service | 6.2% | 27.0% | 10/191 | 2 |

reduced_counts 是本轮固定的删项对照，不是旧模型的跨版本复现；不能用这张表宣称战胜此前不同数据身份上的预测器。
开发验收要求：按形状平均误差 ≤10%、最大误差 ≤25%、错选损失超过 10% 的候选对为 0。
本轮修正版 row_service 验收：**未通过**。即使通过也不自动进入编译器决策。第二轮修正看过第一轮结果，因此不是独立验证。

| 决策范围 | 错选形状 | 平均选择损失 | 最坏选择损失 |
|---|---:|---:|---:|
| all_measured_stable | 2/10 | 0.3% | 2.3% |
| compiler_host_vs_fused | 1/10 | 0.1% | 1.2% |
| fused_only | 2/10 | 0.2% | 1.2% |

geometry 阶段没有测齐强 host；表中最优只限该形状已测且稳定的候选。候选间极小差异也计入原始误选数，不能宣称统计显著。

## M512 与大 N：数值预测

单位 μs。预测完全排除该形状所有候选；实测列不是训练输入。

| M,K,N | 候选 | 留出预测 | 实测 | 误差 | 质量标记 |
|---|---|---:|---:|---:|---|
| 64,4096,2048 | host@prepared | 593.4 | 582.0 | +11.3 | 无 |
| 64,4096,2048 | host_compiler_cublas@prepared | 574.8 | 478.2 | +96.7 | 无 |
| 64,4096,2048 | host_merged_cublaslt@merged | 350.2 | 318.8 | +31.4 | 无 |
| 64,4096,2048 | host_merged_cutlass@merged | 389.0 | 350.5 | +38.6 | within_process_drift |
| 64,4096,2048 | splitk2@prepared | 780.3 | 718.1 | +62.2 | 无 |
| 64,4096,2048 | splitk2_groupedn2@prepared | 843.7 | 776.6 | +67.1 | 无 |
| 64,4096,2048 | splitk4@prepared | 705.5 | 718.4 | -12.8 | 无 |
| 64,4096,2048 | splitk4_groupedn2@prepared | 667.8 | 599.8 | +68.1 | 无 |
| 512,4096,512 | host@prepared | 1101.6 | 1076.5 | +25.1 | 无 |
| 512,4096,512 | host_compiler_cublas@prepared | 903.6 | 906.0 | -2.4 | 无 |
| 512,4096,512 | host_merged_cublaslt@merged | 637.0 | 637.7 | -0.7 | 无 |
| 512,4096,512 | host_merged_cutlass@merged | 633.0 | 652.3 | -19.3 | 无 |
| 512,4096,512 | splitk2@prepared | 995.5 | 1006.1 | -10.5 | 无 |
| 512,4096,512 | splitk2_groupedn2@prepared | 840.0 | 776.3 | +63.7 | 无 |
| 512,4096,512 | splitk4@prepared | 1001.7 | 1047.6 | -45.9 | 无 |
| 512,4096,512 | splitk4_groupedn2@prepared | 753.2 | 725.4 | +27.8 | 无 |

## 不能被清理掩盖的模型错误

以下按实际损失排序，所有候选对和成本项差异见 pair-errors.json。

| M,K,N | 候选 A / B | 预测 B−A μs | 实测 B−A μs | 错选损失 |
|---|---|---:|---:|---:|
| 192,3072,512 | splitk2@prepared / splitk2_groupedn2@prepared | -22.0 | +80.9 | 14.6% |
| 192,3072,512 | host@prepared / splitk2@prepared | +23.7 | -65.0 | 11.8% |
| 256,8192,512 | splitk2@prepared / splitk2_groupedn2@prepared | -104.8 | +97.2 | 7.7% |
| 256,8192,512 | host@prepared / splitk2@prepared | +12.1 | -54.8 | 4.4% |
| 256,8192,512 | host@prepared / splitk2_groupedn2@prepared | -92.7 | +42.4 | 3.2% |
| 512,4096,512 | host_merged_cublaslt@merged / host_merged_cutlass@merged | -4.0 | +14.6 | 2.3% |
| 512,3072,512 | host@prepared / splitk4@prepared | +8.7 | -14.0 | 1.6% |
| 192,3072,512 | splitk4@prepared / splitk4_groupedn2@prepared | -61.5 | +6.2 | 1.2% |
| 192,3072,1536 | splitk2_groupedn2@prepared / splitk4_groupedn2@prepared | -30.6 | +2.8 | 0.4% |
| 64,4096,2048 | splitk2@prepared / splitk4@prepared | -74.8 | +0.3 | 0.0% |

## 条件暴露与新策略预测

对三个旧 profiler 捕获，保留各 kernel 原时长，重算旧图和精简依赖图的最长路径。差值是**固定时长假设下的队列依赖暴露**，不等于可实现的加速，也不是物理通信暴露。

| 捕获 | 旧图路径 μs | 精简图路径 μs | 条件暴露 μs |
|---|---:|---:|---:|
| largenq2 | 621.4 | 209.3 | 412.0 |
| m512q2 | 915.1 | 299.1 | 616.5 |
| m512q4 | 940.9 | 306.2 | 634.8 |

各列独立取中位数；暴露列先逐轮相减，再取中位数，因此不必等于前两列中位数之差。
新策略使用显式情景 T′=T[a+(1−a)(4/11)b]，a 为不随路径缩短的耗时占比，b 为并发改变后的服务时间倍率；二者尚未测得。
policy-scenarios.json 分开保存以留出预测和旧实测为锚的数值。a=0、b=1 是理想化假设，不能据此宣称 2.75 倍加速。物理 ECR/EDMR 保持未知。
情景包含 b=3 的争用增加情形：并行度增加也可能抹去收益。枚举范围不是统计置信区间。
当前修复后的源码、输入就绪策略与新依赖图均没有 GPU 标签，旧数据不能验证新运行时。

## 复现与文件

在仓库根目录执行（输出目录必须不存在）：

```sh
python3 EDM-IR/tools/evaluate_mechanism_prediction.py --out /tmp/edmir-mechanism-review
```

- normalized-processes.jsonl：306 条原始记录及来源；保留全部 samples_ms。
- configurations.json / dataset-manifest.json：标签、三次进程中位数、质量标记及哈希身份。
- models.json / fold-models.json：全量标定及每个留出模型、系数、秩、KKT、训练 ID。
- heldout-predictions.json / calibration-predictions.json：数值、成本项、外推和误差。
- pair-errors.json / decisions.json：双向 host/fused、q、grouped-N 错误清单。
- structural-aliases.json：旧计数表示的相同输入冲突，以及任何优化器都无法消除的误差下界。
- exposure.json / policy-scenarios.json：条件暴露与新策略假设，不混入主测标签。
- quality-sensitivity.json：含被标记配置的标定对照，明确标为敏感性分析。

本次生成耗时 3.90 秒；无外部依赖安装。软件执行耗时只用于检查本地流程，不代表 GPU 性能。
