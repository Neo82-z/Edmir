P1/P2 正式数据与离线复核入口 — 2026-11-18

实验已结束。先读 REPORT.txt，再按以下分类查阅。原目录与文件字节保留，
避免更改回收清单、预登记分析或冻结模型的身份。DATA_CATALOG.json 按用途
列出 Git 中每个文件；RELEASE_MANIFEST.json 保存发布文件的大小与 SHA-256。

1. 冻结设计与源码
   bundle/plan.json、balanced-p1-plan.json、preregistered-predictions.json、
   analysis-preregistration.json、frozen-primary-model.json、protection-before.json。
   bundle/source/ 为实际执行的源码快照；build/receipt.json 保存构建身份。
   六形状、预测器和系数不因整理而改变。

2. 正式测量
   recovered/results/：主体 51 组，包括正确性、候选选择及验证区组。
   recovered/results-p1-balanced/：主 P1 六组同种子、反转候选顺序配对。
   collect-*.json 和配套 stdout/stderr 保留原始计时样本及数值失败。
   recovered/recovery-manifest.json 是最终 279 文件的远端回收清单。
   原始 Nsight/SQLite 与编译二进制保留本地，不进入 Git。

3. 派生分析
   analysis.json：主要结果；block-regret-summary.json：逐进程区组统计。
   ranking-diagnostics.json、frozen-model-scope.json：误选与适用范围。
   trace-analysis.json：按完整内核符号解析的追踪统计。
   mechanism-source-account.json：源依赖计数；不是硬件带宽或占用率。

4. 图表
   figures/ 保存分析图。论文排版图及生成脚本在 Neo82-z/paper/latex/。
   图表只由已采集数据生成，不替代原始测量。

5. 历史失败与修正
   14 条数值失败完整保留在原始记录及 analysis.json。
   runner-correction.json、测前顺序修正回执保留执行过程。
   trace-analysis-initial-shortnames.json 和 analyze_traces.py 为历史版本；
   正式追踪重算入口为 analyze_traces_full_names.py。
   不删失败，不以更改分析版本掩盖错误。

6. 仅本地保留
   checkpoint/ 是运行中增量副本；payload/ 是上传包副本，均不是正式结果。
   inputs/*.bin、安装包、构建产物、原始 profiler 文件留本地。
   云端访问、计费、实例删除回执及旧 SSH 监控脚本不发布。
   INDEX.txt、SCOPE.txt、ACCEPTANCE.txt 保留原实验语境；其本地运行回执
   不保证存在于 Git clone。当前发布范围以本文件及分类索引为准。

从 Git clone 复核（Python 3.10+，标准库，无 GPU，无训练，无下载）：
   python3 experiments/2026-11-18-p1-p2/verify_release.py

该检查验证发布文件及已发布回收记录的哈希，从原始 samples_ms 重新计算
P1 配对加速比、P2 三进程聚合、独立选择及全部策略损失，并与分析结果核对。
它不模拟 GPU，也不绕过原 analyze.py 对完整回收包和本机保护路径的检查。
完整取证复核仍使用原 analyze.py，需要恢复本地二进制/追踪等全部回收文件。
Git 的部分数据发布不是完整回收包，不称为“279 文件全部随 Git 发布”。
