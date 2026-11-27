# EDM-IR：通信与 GEMM 的执行代价研究

本项目研究单节点多 GPU 上，通信结果的消费方式、分块粒度和归并精度如何影响 GEMM 整程序耗时，并比较强 host 基线、直接访问候选、固定策略与共享评分。

**当前状态：2026-11-27。** 论文主线已改为“expert 的输入行怎样到达 GEMM”：All-to-All staging、从对端内存拉取、在 GEMM 里
就地读。两次上机（11-26 节点 A 的已登记研究，11-27 节点 B 的封存预期研究）已完成，三个模型都测：GLM-4.7-Flash、
Qwen3.5-35B-A3B、DeepSeek-V2-Lite。下面“已有证据”一节是 11-18 之前的 CUTLASS 映射研究，现在是论文的一节和补充材料。

## 两次上机（当前主线）

- [实验报告：事实、证据等级、不能写的说法](experiments/REPORT_2026-11-27.md)（[HTML 版](experiments/REPORT_2026-11-27.html)）
- [第一次研究的设计包](experiments/2026-11-23-replay/)：已登记的设计、队列、判定规则和开机前预测。
- [第二次研究](experiments/2026-11-27-confirm/)：封存的预期 `arms.json`、234 个进程输出、`verify.py`（核对输出与封存文件）、
  `judge.py`（三对规则判定）、`RESULTS.md`、`RUNLOG.md`。
- [探针、一次启动消费者和拉取的代码](experiments/2026-11-26-peer-read-probe/)，以及候选清单 `CANDIDATES.md`。
- [论文源码](https://github.com/Neo82-z/paper/tree/main/latex)（私有仓库）。

## 11-18 之前的研究：先读这四处

- [当前结论、待办与重新评估](paper/PROJECT_STATE_2026-11-18.txt)：唯一的当前状态入口；旧日期计划保留历史语境。
- [核心代码阅读顺序](artifacts/2026-11-17-paper-closeout/CODE_INDEX.txt)。
- [分类数据入口与发布范围](artifacts/2026-11-18-data-release/README.txt)、[证据索引](artifacts/2026-11-18-data-release/DATA_INDEX.json)。
- [论文源码](https://github.com/Neo82-z/paper/tree/main/latex)：唯一正文来源；当前仍为研究稿排版，FGCS 材料尚待收尾。

## 已有证据

| 证据 | 当前可支持的结论 | 阅读入口 |
|---|---|---|
| 八卡 FP16 算子与 FP32 partial 归并 | 候选收益取决于执行身份、数值准入和 q/g 组合；强 host 与固定 q4/g2 均需保留 | [综合分析](experiments/2026-11-17-evidence-synthesis/REPORT.txt) |
| 最后一次大 N 平衡顺序复测 | 封存评分能选到快于 host 的候选，但 fused 内部仍误排；汇总选择损失为 3.48% | [复测报告](experiments/2026-11-17-final-operator/REPORT.txt) |
| P1：M512 受控依赖与分块比较 | 六组反转顺序配对均同向；pipeline/serial 和 q4/g2 对 q4/g1 的配对加速比中位数分别为 1.169×、1.686× | [P1/P2 报告](experiments/2026-11-18-p1-p2/REPORT.txt) |
| P2：六个冻结形状 | top-1 平均损失 2.31%，固定 q4/g2 为 1.08%，前三候选加独立测量为 0.65%；不改写为无需测量的预测胜出 | [分类与离线复核](experiments/2026-11-18-p1-p2/README.txt) |
| GLM-4.7-Flash BF16、vLLM 0.20.1、八卡 A100 | 原生性能基线、专家行数变化、padding 与路由分布 | [真实 MoE 分析](experiments/2026-11-17-real-moe-analysis/REPORT.txt) |

原生 MoE 案例没有接入 EDM-IR，不是端到端加速对照。它与算子实验的精度、通信语义及投影维度不同，不能直接搬用算子加速比。六形状评价支持有限范围内的测量辅助筛选，不建立普适泛化、双向 host/fused 分类或实测通信隐藏比例的结论。

## 代码与本地验证

`EDM-IR/edmir/compiler/` 包含 Program 构造、依赖验证和评分；`EDM-IR/runtime/` 为 C++/CUDA 执行；`EDM-IR/tools/` 为采集和分析入口。`experiments/` 按日期保存执行身份、结果和失败。`papers/` 保存参考资料；最短阅读清单见 [REFERENCES_2026-10-30.txt](papers/REFERENCES_2026-10-30.txt)。

```sh
cd EDM-IR
PYTHONPATH=. python3 tests/real_moe_collection_test.py
PYTHONPATH=. python3 tests/w1_collection_test.py
cmake -S runtime -B /tmp/edmir-host-check -DEDMIR_ENABLE_CUDA=OFF
cmake --build /tmp/edmir-host-check
ctest --test-dir /tmp/edmir-host-check --output-on-failure
# 在仓库根目录，只读 Git 中的 P1/P2 原始样本与冻结记录：
# python3 experiments/2026-11-18-p1-p2/verify_release.py
```

这些是 CPU 逻辑检查，不能替代 CUDA 构建与 GPU 性能验证。完整同步检查见[本轮记录](artifacts/2026-11-18-repository-sync/README.txt)。旧入口说明保留在[整理前快照](artifacts/2026-11-18-repository-sync/README.before.txt)。
