# 第一次研究的原始输出（节点 A，2026-11-26）

已登记设计 `path-replay-2026-11-24.v3` 的正式队列：234 个进程，1170 行计时，全部一次通过。设计、队列和判定规则在
[`../2026-11-23-replay/`](../2026-11-23-replay/)。三个模型都测：GLM-4.7-Flash、Qwen3.5-35B-A3B、DeepSeek-V2-Lite。

| 文件 | 内容 |
|---|---|
| `run.tar.xz` | 节点上的整个运行目录 `run/`，5305 个文件，解压后约 148 MB |
| `MANIFEST.sha256` | 归档内每个文件的 SHA-256 |
| `results/` | `analyze.py` 的输出：`paper_results.json`（论文的表和图由它生成）、`effects.csv`、`phases.csv`、`timing_diagnostics.csv` |
| `reproduce.sh` | 解压、按清单核对、重跑分析，并与 `results/` 逐字节比较 |

`run.tar.xz` 的 SHA-256：`4bca94da24f6b7c7da78215f6ff22e046a2d0b78a825980200ee98a01155554f`。

## 归档里有什么

- `metrics.jsonl`：每个进程每个臂一行（1566 行）。1170 行是计时结果，含墙钟、事件时间和入队时间的中位数，以及对应样本文件的
  哈希。另外 396 行状态是 `not_run`，属于 `D_Q1_G1` 和 `BOLD_Q1_G1` 两个臂：它们只在某个 staged 臂失败时才启用，这次没有触发。
- 每行计时对应四个文件，`<进程>-<臂>-samples.json`、`-events.json`、`-enqueue.json`、`-phases.json`：100 次计时迭代的
  每次迭代数值；阶段文件里是每张卡每次迭代的阶段时间。
- `admission.jsonl`、`ADMISSION_LOCK.json`：数值准入记录和锁。
- `RUN.json`：设计、硬件、测试程序、输入和构建的哈希。`hw_gate.json`：硬件 gate 的输出。
- `prediction/`、`prediction_calibrated/`：开机前封存的预测和节点标定后的预测。
- `microbench.json`、`slt.json`、`smoke.json`、`trials-*.json` 和三个原生 kernel 配置文件：标定、调优和冒烟测试的输出。
- `job-logs/`、`process-started/`、`process-exits.jsonl`：每个作业的日志和退出状态。

## 复算

```sh
./reproduce.sh
```

不需要 GPU。脚本结束时四个结果文件都应显示 `identical`。`analyze.py` 报告的 “156/189 complete comparisons” 里，不完整的 33 个
就是上面没有触发的那组条件比较。

第二次研究的输出在 [`../2026-11-27-confirm/results/`](../2026-11-27-confirm/results/)。它的测试程序只保存每个进程的中位数和
阶段时间，没有保存每次迭代的样本。
