# 数据来源审计与预测器离线修改

2026-10-20。仅整理已有测量、核查公开来源并执行 CPU 拟合；没有启动 GPU 实验。
EDM-IR 仍是编译器，本次修改的是其中的代价预测与候选选择模块。
新模型保留为实验实现，未接入 `compile-fused` 默认准入。

随后完成的[代码审查与 debug](../2026-10-20-predictor-debug/README.md)修复了输入、
模型状态、数据角色和准入检查，并纠正未参与 host 回归的 Gamma 口径。
本目录结果保留生成时的代码哈希；修复后的重放与失败点追踪保存在该 debug 目录。

**本轮结论：数据口径已经统一，新模型还不能替换旧模型。**
相对耗时训练加四点偏置校准改善了 P8 的平均选择损失，但没有解决最坏点；
不校准迁移反而更差。固定候选基线也表明，预测器的复杂度尚未带来稳定收益。

## 1. 哪些公开数据确实存在

审计范围是下表中的具体版本和文件，并非“所有公开数据都不可用”。
[public_sources.json](public_sources.json) 保存 19 个文件的完整 commit、URL、
SHA-256 和字节数；[public_audit.json](public_audit.json) 是可复查的计数结果。
下载文件只作为文本/CSV 解析，没有执行第三方脚本。

| 来源 | 查到的内容 | 当前用途与限制 |
|---|---|---|
| [TileLang benchmark，b658f7e](https://github.com/tile-ai/tilelang-benchmark/tree/b658f7e9f326156d11a09dff1e9825fa6d9a8767) | 大 M FP16 表有 13 个形状，cuBLAS 与 Triton 各 13 个值均为 `-1`；BitBLAS 有 13 个硬编码正值。另一个 N=K=8192、小 M 表的五组实现共 30 个值全为 `-1` | `-1` 是缺失，不能拟合为耗时。小 M 表还混有 INT4/INT2，不等价于我们的 FP16。可参考形状和实现，现有文件不足以形成完整对照 |
| [GPPerf CUTLASS CSV，d9ff663](https://github.com/pavlyhalim/GPPerf/blob/d9ff663d4b12fd9586f9d817d1b6db911570f342/cutlass-analysis/data.csv) | 16,128 行，GPU 全为 RTX 4070，kernel 为 SIMT SGEMM；beta 含 0、0.5、1 | 是实际数据，但硬件、精度、指令路径和算子范围均不同。`runtime` 单位尚未独立还原；tile 字段与 kernel 名存在不一致，`TFlops` 字段需复核；M=128256 原样保留。暂不进入当前训练 |
| [公开 A100 NCCL CSV，4a1ce39](https://github.com/bandlayash/multi-gpu-perf-project/blob/4a1ce397ff895b15f692b00884645e21cd5dbcbc/results/a100_4gpu/nccl_sweep_results.csv) | 496 行，4 种 collective 各 124 行，1–4 GPU，附 NV12 拓扑和 A100-SXM4-80GB 设备信息 | CSV 只存四舍五入后的 bus bandwidth、返回码；原始 time 与逐行 wrong 未保留。可作通信组件参考，不能还原精确延迟，更不能替代 direct-peer load 的服务时间 |
| [Tilebench，00a9751](https://github.com/Deep-Learning-Profiling-Tools/Tilebench/tree/00a9751dfcd5bfa5ad15f12775c03e0b506ff213) | 检查了 B200 的 CSV；FP32/FP16/FP8 matmul 在 default、autotune 条件下各 60 行，其中各 20 个 FP16 形状，含 PyTorch/Triton/cuTile/TileLang 耗时 | 同版本 config 补齐 M=N=4096，K=1024…20480；不是小 M A100 数据。default/autotune 不是重复测量；配置中的 graph、L2 flush、正确性容差也必须随数据保留。可参考组件比较方法 |
| [CUTLASS profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)、[NCCL tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、[Triton GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 测量工具、字段定义、实现与示例结果 | 工具存在不等于已有匹配数据。教程表不能自动获得完整的硬件/协议出处。Triton `GROUP_SIZE_M` 是 CTA 排序分组，不是我们的 grouped-N |

本轮从这些公开来源新增的**完整融合算子训练标签为 0 条**。
这不妨碍用公开代码设计基线，也不意味着以后不能建立独立的组件模型；
它意味着不能把单 GPU GEMM 或 NCCL 曲线拼成未经测量的融合“真值”。

用户补充的 FP16 GEMM 图明确标注 A100；上述 TileLang 仓库的 README 也列有
A100 matmul 结果。此前表中的 B200 属于另一个 Tilebench 仓库，不能混称。
图中为 M0–M7，README 对应的八项形状中 M 为 4096/8192；
之前核查的 `data_float16_gemm.py` 则列出 13 项，不能仅按 M0 等编号直接拼接。
硬件采用 A100 与图表原始耗时能否完整导入，需要分别核实。

## 2. 已整理的数据与计时口径

| 数据 | 记录数 | 实际覆盖 | 用途 |
|---|---:|---|---|
| P4 完整算子 | 346 | 41 个形状；20 个形状有 11 候选，21 个形状只有 6 候选 | 按原 sweep 留组评估 |
| P8 完整算子 | 253 | 23 个形状×11 候选；其中 4 个原校准点、19 个原评估点 | 回顾性迁移分析 |
| 自有 cuBLAS/CUTLASS GEMM | 72 | 9 个形状×2 实现×4 个 device | 独立组件参考 |
| 自有 NCCL All-to-All | 84 | P4/P8×21 个消息尺寸×2 种缓冲模式 | 独立组件参考 |

599 条完整算子记录对应 **64 个形状**，不能把每个候选都当作独立工作负载。
P4/P8 结果已经参与过模型设计，因此这里的原 `heldout` 角色只是历史字段；
本轮不把它们重新称为未见测试集。

文件入口：

- [data/manifest.json](data/manifest.json)：来源哈希、环境分组、计时实现版本、数量和转换规则。
- [data/selector_measurements.jsonl](data/selector_measurements.jsonl)：原标签、形状、q/g、kernel、计时摘要、warmup/iterations、原文件行号，以及重复测量的使用方式。
- [data/component_measurements.jsonl](data/component_measurements.jsonl)：GEMM 与 NCCL，明确禁止作为完整融合标签导入。

**计时范围是这次审计的重要发现。** P4/P8 的 collector 在
`steady_clock` 计时区间内执行 `backend.execute()` 和 `backend.synchronize()`；
它测量整个多 GPU 程序的提交、运行及同步，初始化和输入拷贝在区间外。
两个历史代码版本的 collector 哈希相同，记录在 manifest 中。
单独 GEMM 则使用 CUDA events。因此：

- 完整算子毫秒数不能直接与单个 kernel 的毫秒数混合拟合。
- CPU 调度、事件依赖和同步开销可能影响 P4→P8 迁移；现有数据没有独立测出这些项，不能据此断言它们解释了多少误差。
- 原始 `median_ms` 使用排序后下标 `floor(0.5*(n-1))`，原样保留；没有改造成另一种中位数。
- µs 统一除以 1000 转成 ms；NCCL `N/A` 验证数保存为 null，不冒充零错误。
- 12 条重复测量中，10 条替换原评估标签，另 2 条属于校准点，仅存档，仍使用原校准标签。重复记录不增加训练样本数量。
- `remote_input_bytes` 原本是全程序各 rank 之和，特征函数按 P 换成每目的端量；原字段不覆盖。
- 本项目 M 是每个源–目的对的行数，单个目的端合并后为 PM。外部单 GEMM 的 M 必须先匹配对应基线，再作比较。

## 3. 预测器具体改了什么

实现：[relative_selection.py](../../EDM-IR/edmir/compiler/relative_selection.py)。

旧模型分别拟合 host 和 fused 的绝对 log(ms)，最后比较两者。
新增模型令 host 得分为 0，fused 得分估计 `log(T_fused/T_host)`；
同一工作负载内的每对候选提供一个约束：

```text
loss = Σ_workload [ Σ_(a<b) ((score_a-score_b)-log(T_a/T_b))² / pair_count ]
       + ridge × Σ_nonintercept coefficient²
```

每个工作负载的总配对权重为 1，避免 11 候选形状仅因行数更多而取得更大权重。
标准化参数也只由训练数据计算，并按工作负载等权。
它仍是平方误差代理目标，不等于直接最小化选择损失，也不保证更优。

保留前一轮的 `grouped × log(K/q)`、`underfill × log(K/q)` 两个交互项，
允许 grouped-N 的得分差随 K 改变；`underfill` 仍只是 CTA 网格不足的代理，
不是实测 occupancy。所有候选由一个共同得分函数排序。
固定 ridge=1，没有在这批结果上搜索超参数；新旧目标的尺度不同，
相同数字的 ridge 不代表相同的正则化强度。

推断接口 `ScheduleCandidate` 不含 `measured_ms`，拒绝直接传测量对象；
同次选择必须属于同一形状和环境。跨环境推断须显式标记。
四个原 P8 校准形状只用于估计一个共同偏置，不重拟合候选排序。
它改变 host/fused 准入，不能解决 q/g 选错。

同时修正旧测量校验中 NaN/Inf 可能通过的问题。原模型有效输入的预测保持不变。
这些修改不改变物理计数恒等式，也没有把预测字节数改称实测通信流量。

## 4. 实际结果与失败点

[relative_results.json](relative_results.json) 保存模型系数、数据/代码哈希、
每个点的决策、平均/p95/最坏损失、host 回退损失和分候选数量结果。
定义 `regret = T_selected/T_best_observed - 1`，越低越好；
最优值只覆盖该点已经测过的候选。

| 方法 | P4 留组平均损失，41 点 | P4→P8 不校准，19 点 | P8 校准后的平均损失 |
|---|---:|---:|---:|
| 训练集选定的固定候选 | 5.43% | 9.53% | 9.53% |
| 原绝对耗时模型 | 4.19% | 10.87% | 15.02%，仅四点重新拟合 |
| 前一轮交互项模型 | 3.27% | 9.41% | 14.88%，仅四点重新拟合 |
| 新相对耗时模型 | 4.41% | 13.04% | 8.65%，P4 模型加四点偏置 |

“四点重新拟合”和“P4 模型加四点偏置”使用训练信息的方式不同，
最后一列不是受控的损失函数消融。固定候选在 P4 训练后得到 `splitk4`；
它完全不读取 P8 评分标签，必须保留为后续选择策略基线。

新相对模型加偏置后，最坏点仍为：

```text
P=8, M=512, K=8192, N=512
最优已测候选：splitk4_groupedn2，3.4733 ms
新模型选择：  splitk2，           5.7010 ms
host：                         5.0696 ms
选择损失：64.14%；相对 host 慢 12.45%
```

P8 校偏后仍有 2/19 点慢于 host，其中最大退步约 25.00%。
平均损失降低不等于可以部署。19 点的最优方向全是 fused，
所以方向准确率 100% 也不足以验证 host/fused 分界。

失败说明的范围：

1. **训练覆盖缺口是事实。** P4 的 20 个完整 grouped-N 形状全部为 M=64。
   P4 虽然有 M=512，却没有该 M 的 grouped-N 标签。最坏点需要跨 P、
   跨 grouped-N/M 组合泛化，不能只说“已经见过 M=512”。
2. **准入偏移与排序错误需要分开。** 单偏置能修正一部分整体相对耗时偏移；
   它不改变 fused 排名，因此保留了 64% 的最坏损失。
3. **计数到耗时之间仍缺条件。** 同样的 rho、CTA 数，并不说明寄存器、
   驻留 CTA、远端加载停顿、调度开销相同。当前改动改善了训练接口，
   尚未建立或验证这些条件的服务时间模型。

## 5. 下一步收敛到哪里

先保留当前失败结果，不继续在这 19 点上试大量特征直到数字变好。
下一版应分别处理完整程序的调度/同步开销、GPU 计算与远端访存，
把组件数据用在它们能约束的项上；现有记录不足以辨识的系数明确留空。
本轮相对模型是可对照的候选实现，不是已经验证的最终模型。

后续实验优先填两个缺口：不同 M 下的 grouped-N 成对测量，
以及强 host 基线下的选择空间。旧的逐源 CUTLASS host、合并 PM 的优化 GEMM、
分块流水基线不能互相替代；目前后两者没有对应新测量，
不能把当前的 oracle 空间直接写成相对强基线的收益。
先前设计中的大网格仍是候选范围，不是已承诺执行的 GPU 任务。

下一次新数据到来前冻结特征、校准方式、固定候选基线和评价指标，
再保留新的独立评估点。继续分别报告计数关系、条件预测、实测现象。

## 本地复查

在仓库根目录执行，输出路径必须尚不存在：

```bash
python3 EDM-IR/tools/prepare_selector_dataset.py --out /tmp/edmir-offline-data
python3 EDM-IR/tools/evaluate_relative_selector.py \
  --dataset /tmp/edmir-offline-data --out /tmp/edmir-relative-results.json
python3 EDM-IR/tools/audit_public_selector_sources.py \
  --catalog experiments/2026-10-20-offline-predictor/public_sources.json \
  --cache /tmp/edmir-public-sources --out /tmp/edmir-public-audit.json
python3 -m unittest discover -s EDM-IR/tests -p '*test.py'
```

前两条无 GPU、无网络；第三条按固定 commit 下载并校验哈希，缓存只用于复查，
不执行下载代码，也不自动把公开数据送入训练。
完整 Python 测试通过；旧原模型和交互项模型的三种协议指标与原归档逐项一致。
