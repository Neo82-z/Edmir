# 上机前本地交付：固定候选、成对诊断

本轮不拟合、不更新评分模型、不使用六个冻结形状。交付的通过标准是采集包可以在服务器上进入环境/数值检查，不是宣称模型或 GPU 数值正确性已经通过。

## 固定范围

| 阶段 | 形状 | 候选配置 | 独立进程 | 采集上限 |
|---|---:|---:|---:|---:|
| P4 诊断 | 6 个已有开发形状 | 20 | 70 | 1200 秒，含本次 preflight |
| P8 grouped 复核 | 1 个已测过的八卡形状 | 5 | 25 | 600 秒，含本次 preflight |

P4 的五个短 K 形状分别使用 host/direct/q2，三轮循环轮换位置；grouped 形状使用 host 加 q4/q8 × g1/g2 四格，五轮轮换。每个候选在每个位置恰好出现一次。每次 30 次预热、300 次计时，所有原始时间保留。P4 优先运行 grouped 四格，再运行三个错误点与两个正反对照。

上一版诊断规格中的 230 次进程未执行。根据已有服务器记录中每进程十余秒初始化/清理的成本，本轮在新数据采集前缩为 70 次；原规格保留不改。这些重复用于报告方向一致性，不给出独立置信区间。

P8 只运行 `(8,64,8192,1024)` 的同一四格及 host，不展开六形状独立验证或 MoE 新矩阵。机器仍可按单机 8×A100 SXM4 80GB、全互联 NV12/P2P 准备；P4 阶段用其中四张，两个阶段不能同时运行。

## 身份与验收目标

- FP16 输入/输出，FP32 累加；serial split-K 中间写回仍为 FP16。没有通过放宽容差解决精度失败。
- 统一逐源 CUTLASS host、prepared metadata、`resource_queues_v1`，对齐本轮候选间的执行条件。该 host 是机制诊断基线，不是论文最终最强 host 的完整比较。
- 两个阶段均是新构建/新输入身份。旧 P4 微秒数不能当作新环境的绝对时间预测；本轮 P8 也不能混入以前 `uniform_immutable_peer_v1` 的结果。
- 主要指标固定为整程序 host wall-time 中位数；同时记录均值、p95、每个重复块的成对差值和 q×g 交互。选择方向反复变化就保留弱证据，不改统计口径，不追加训练。
- 输出覆盖全部元素，沿用原最大误差检查：`max_abs <= .02 OR max_normalized <= .02`，归一化分母为 `max(abs(reference),1)`。它不是逐元素 `atol+rtol` 判据。失败时停止采集并保留日志，不生成耗时标签。

## 已在本地准备

两份封存包均包含生成程序、源码快照、逐文件哈希、规范输入和 CPU FP32 参考，输入/参考合计约 334 MB。固定 CUTLASS 和 NCCL 源码也一并提供，不需要上 GPU 后再下载仓库或模型。

本地测试和文件完整性结果见 [verification.json](verification.json)。测试范围包含候选/顺序覆盖、输入和实现身份、原始样本一致性、失败保留、断点续跑、预算耗尽、回收包哈希和生成程序布局。CUDA 编译、NVLink/P2P 和实际 GPU 数值检查仍需上机完成。

脚本特性：

- 每个测量进程最多 90 秒；超时会终止进程组，保留 stdout/stderr。
- 续跑只执行未完成项，保留旧失败日志，不覆盖原截止时间。跨环境或二进制变化不能续接同一结果目录。
- 失败、不完整结果也生成 summary，并可封存回收。只有全部计划项都存在，`complete` 才为 true。
- profiler 单独输出，不能导入主计时标签。没有承诺从当前 wall-time 自动分解出设备关键路径或通信等待。
- 采集截止不等于 RunPod 停机。真正开机时仍需先配置平台停机保护，遵守此前授权的最多 $15、最多一小时，并给数据回收留出时间；不因某阶段剩余预算而延长租机。

## 服务器执行入口

建议沿用已跑通过的 Linux + CUDA 12.8 devel 环境，Python 3.11+、CMake 3.30+。输入已在本地生成，服务器无需生成参考或训练。以下命令由操作方在确认当前机器后逐步执行，不自动租用、恢复或关闭实例。

先解包并检查交付哈希。在包目录中，编译固定 NCCL（已有匹配构建可复用，但 runner 会核验实际加载库哈希）：

```sh
python3 verify-delivery.py
timeout --kill-after=5 600 make -C dependencies/nccl src.build -j4 NVCC_GENCODE='-gencode=arch=compute_80,code=sm_80'
bash run-on-server.sh p4 verify
bash run-on-server.sh p4 inventory
bash run-on-server.sh p4 build
bash run-on-server.sh p4 preflight
bash run-on-server.sh p4 run
```

每个阶段应检查结果和剩余租机时间。只有时间足够时才进入下面的 P8 阶段，不为完成清单延长预算：

```sh
bash run-on-server.sh p8-grouped verify
bash run-on-server.sh p8-grouped build
bash run-on-server.sh p8-grouped preflight
bash run-on-server.sh p8-grouped run
```

`run` 完成或采集失败后，会尝试将 `work/<profile>/results/` 连同构建凭据、编译日志封存到 `work/<profile>/recovery-*.tar.gz`，同时生成 SHA-256。应持续回收该目录，不能只在关机前第一次下载。退出码 75 表示预算不足，不能当成完成；异常退出需要先检查失败文件。人工确认可以续跑且原截止时间未到时，才使用 `resume`。

profile 如有必要再运行 `bash run-on-server.sh p4 trace <job-index>`，输出在独立目录。当前包不把 profiler 当作每个配置的必做项。

## 科学问题的通过标准

1. 短 K：检查 direct/q2 的正反方向及其随 tile 数变化是否可重复；不能把 K512 一律切换成 direct。
2. grouped-N：分别核验 q 变化、g 变化和交互差值，不能只凭 q8/g2 与 q4/g1 的对角比较宣布模型正确。
3. 只确认稳定且身份一致的事实；尚不能定位的硬件原因写成证据缺口，不据此拟合掉错误。

前一轮问题定位见 [固定模型诊断](../2026-10-28-fixed-model-diagnosis/README.md)。本轮不修改其模型、标签或结论。
