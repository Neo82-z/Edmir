# 实验证据索引

更新：2026-11-19。本文件只做索引：没有移动、重命名或删除任何目录，
论文脚本里的路径全部保持有效。

## 怎么用

- 写论文或回复审稿意见：只看 **A**；需要讲来历、动机或附录时看 **H**。
- **P** 是预测器冻结前的本地迭代，结论已收束进论文附录（`sections/12-19`），
  正文不再从这些目录取数。
- **O** 只有计划、执行包和计费回执，没有正式测量，不作为证据引用。
- "论文脚本引用"一列来自对 `llm-serving-mlsys-latex/latex/{scripts,data,sections}`
  的路径检索，共 24 个目录。论文另有一批 8 月数据已复制到
  `latex/data/a100_*`，对应 H 类中的 8 月目录。

## 证据链（按新论文叙事顺序）

| 论点 | 主证据 | 补充 |
|---|---|---|
| 直连消费可行且能赢简单 staged 路径 | `2026-11-17-final-operator` | `2026-08-26-a100-nccl-device-cutlass-fusion` |
| 直连内部的复用/并行/精度三向权衡 | `2026-11-18-p1-p2`（P1） | `2026-08-29-a100-grouped-n-plan`、`2026-08-29-a100-splitk-q-boundary` |
| replay 机制（M512 远端 GEMM −44%） | `2026-11-18-p1-p2`（trace） | `2026-11-19-offline-bottleneck-audit` |
| 精度属于映射空间 | `2026-11-16-shadeform-precision` | `2026-11-16-precision-analysis` |
| 冻结选择在未用形状上的损失 | `2026-11-18-p1-p2`（P2） | P 类（来历） |
| 真实 MoE 的 M 跨度（decode 1 行到 prefill 1.4 万行） | `2026-11-17-real-moe-gpu` | `2026-11-17-real-moe-analysis` |
| 原生 vLLM 端到端基线（GLM，S1/S32/L1/L32） | `2026-11-17-real-moe-gpu` | — |
| source-parallel 的动机与静态上界 | `2026-11-19-offline-bottleneck-audit` | `2026-11-19-source-parallel` |

## A. 新论文主证据（GPU 测量，论文正文引用）（8 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-11-16-precision-analysis` | 620K | 是 | 精度失效机制与准入边界分析 |
| `2026-11-16-shadeform-precision` | 24M | 是 | FP16/FP32 部分和精度准入的 GPU 测量 |
| `2026-11-17-evidence-synthesis` | 480K | 是 | 跨轮次证据汇总与论文插入数字 |
| `2026-11-17-final-operator` | 10M | 是 | 17 候选全空间算子测量（W1 收尾），直连 vs 合并 host 1.69–3.16× |
| `2026-11-17-real-moe-analysis` | 552K | 是 | 真实 MoE 路由/几何离线分析，论文 §5b 图表来源 |
| `2026-11-17-real-moe-gpu` | 1.1G | 是 | GLM-4.7-Flash 原生 vLLM 基线（S1/S32/L1/L32）+ 专家路由 M 分布 |
| `2026-11-18-p1-p2` | 7.3G | 是 | P1 M512 受控配对 + P2 六冻结形状选择；论文主表主图 |
| `2026-11-19-offline-bottleneck-audit` | 92K | 是 | 静态上界与 F 地板分析（source-parallel 动机） |

## H. 历史 GPU 证据（附录、动机与机制来源）（23 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-08-25-runpod-4xa100-sxm` | 628K | 是 | 4×A100 首轮后端验证（NCCL/cuBLAS 基线） |
| `2026-08-26-a100-cutlass-cost-model` | 536K |  | 4×A100 cost model 原始数据 |
| `2026-08-26-a100-cutlass-kernel-family` | 892K |  | CUTLASS provider/kernel family 替换 |
| `2026-08-26-a100-cutlass-provider` | 660K |  | CUTLASS provider/kernel family 替换 |
| `2026-08-26-a100-fused-heldout` | 2.5M |  | direct vs split-K 边界（K 扫描、held-out 网格） |
| `2026-08-26-a100-fused-k-boundary` | 328K |  | direct vs split-K 边界（K 扫描、held-out 网格） |
| `2026-08-26-a100-nccl-device-cutlass-fusion` | 64K |  | NCCL Device API LSA + CUTLASS 直连融合首个验证（direct-peer 起点） |
| `2026-08-26-a100-nccl-device-fused-reference` | 56K |  | NCCL Device API LSA + CUTLASS 直连融合首个验证（direct-peer 起点） |
| `2026-08-26-a100-nsys-trace-validation` | 4.9M |  | nsys 依赖语义验证 |
| `2026-08-26-a100-runtime-capabilities` | 16K |  | 运行时能力门槛（NCCL 2.25 不支持 Device API） |
| `2026-08-26-runpod-4xa100-cost-model` | 4.0M |  | 4×A100 cost model 原始数据 |
| `2026-08-29-a100-8gpu-transfer-plan` | 40K | 是 | P4→P8 迁移（8×A100，预登记） |
| `2026-08-29-a100-grouped-n-plan` | 3.7M | 是 | grouped-N 复用 vs 并行（4×A100） |
| `2026-08-29-a100-splitk-q-boundary` | 3.2M |  | split-K q 供给边界（4×A100） |
| `2026-08-30-a100-8gpu` | 3.3M | 是 | P4→P8 迁移（8×A100，预登记） |
| `2026-08-30-numerical-accounting` | 236K |  | 数值账本（8 卡前） |
| `2026-10-21-runpod-mechanism-collection` | 2.2G | 是 | 机制采集包（M512/大 N/host 基线） |
| `2026-10-25-p8-gpu-collection` | 2.2G | 是 | 10 月 8 卡有界采集 |
| `2026-10-25-runpod-hour-01` | 255M | 是 | 10 月一小时 A100 采集 |
| `2026-10-28-p8-session` | 119M | 是 | 8 卡采集与数值失败复现 |
| `2026-11-11-shadeform-w1` | 7.1M | 是 | W1 八卡四开发形状 GPU 原始数据 |
| `2026-11-11-w1-local-analysis` | 376K | 是 | W1 本地分析 |
| `2026-11-13-w1-local-convergence` | 18M |  | W1 本地收敛（精度准入、同条件评分） |

## W. 进行中（下一轮 GPU）（3 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-11-19-g1-preparation` | 160K |  | G1 准备与契约测试（下一轮 GPU） |
| `2026-11-19-source-parallel` | 9.0G |  | source-parallel 原型、静态边界、冻结计划草案（下一轮 GPU） |
| `2026-11-20-final-round` | 0.6M |  | 最后一轮执行包：设计表、预测/标定/分析脚本、开机前封存预测（未开机） |

## P. 预测器/选择器本地迭代（已收束，仅附录与复现）（47 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-08-cost-model-validation` | 16K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-19-selector-baseline-design` | 288K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-deepinfer-measurement-audit` | 36K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-host-metadata-optimization` | 245M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-local-replay-runtime-checks` | 247M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-offline-predictor` | 1.3M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-predictor-contracts` | 916K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-predictor-debug` | 508K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-selector-decision-audit` | 3.9M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-20-splitk-local-revision` | 245M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-additive-selector-repair` | 8.1M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-decision-loss-iterations` | 47M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-execution-cost-revision` | 8.9M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-execution-evidence-delivery` | 13M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-host-boundary-repair` | 7.3M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-joint-decision-review` | 68M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-round-two-error-audit` | 1.0M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-shared-candidate-review` | 38M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-21-structural-identifiability` | 5.6M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-24-execution-contract-repair` | 20M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-24-fused-dependency-audit` | 404K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-24-mechanism-prediction` | 9.8M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-bounded-candidate-selection` | 180K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-final-local-revision` | 6.2M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-fine-grained-prediction` | 16M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-initial-data-replay` | 13M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-latency-throughput-regimes` | 7.1M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-25-post-gpu-revision` | 86M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-26-neusight-geometry-audit` | 48K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-26-tile-model-revision` | 1.3M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-calibrated-structure` | 6.5M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-conditional-curves` | 17M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-dependency-structure-review` | 5.0M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-depth-cause-review` | 9.2M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-fixed-model-diagnosis` | 116K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-host-decision-audit` | 3.9M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-host-extrapolation-review` | 1.1M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-local-simulation-revision` | 20M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-muon-model-audit` | 180K |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-overlap-evidence` | 22M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-parallel-depth-review` | 34M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-precision-scoring` | 2.9M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-shape-evaluation` | 3.3M | 是 | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-shape-transfer-review` | 30M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-28-split-stage-review` | 11M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-29-operand-reuse-review` | 1.6M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |
| `2026-10-29-pipeline-dependency-review` | 1.4M |  | 预测器/选择器本地迭代（冻结前），只用于附录历史与可复现性 |

## O. 计划、执行包、准备与计费回执（无正式测量）（20 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-09-09-routed-offline-readiness` | 752K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-21-boundary-gpu-package` | 631M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-25-moe-model-selection` | 4.0K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-25-regime-gpu-validation` | 1.1G |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-28-paired-gpu-readiness` | 553M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-29-cpu-preparation` | 2.9M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-30-source-delivery` | 4.0K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-10-31-paper-experiment-plan` | 16K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-01-w1-readiness` | 40K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-11-w1-extension` | 4.9M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-11-w1-session` | 28K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-14-next-window` | 5.8G |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-16-next-window` | 124K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-16-offline-assets` | 268K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-real-moe-gpu-quote` | 12K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-real-moe-local` | 12M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-real-moe-plan` | 32K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-remote-cpu-budget` | 88K |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-remote-cpu-preparation` | 9.7M |  | 计划/执行包/准备/计费回执，无正式测量 |
| `2026-11-17-resource-check` | 68K |  | 计划/执行包/准备/计费回执，无正式测量 |

## X. 早期热身或无关支线（5 个）

| 目录 | 大小 | 论文脚本引用 | 作用 |
|---|---|---|---|
| `2026-05-30-nccl-4x4090-baseline` | 40K |  | 早期热身或与本文无关的支线 |
| `2026-05-30-tinygrad-notes-repro` | 4.0K |  | 早期热身或与本文无关的支线 |
| `2026-08-4x4090-nccl-baseline` | 12K |  | 早期热身或与本文无关的支线 |
| `2026-08-a800-a2a-gemm` | 8.0K |  | 早期热身或与本文无关的支线 |
| `deepep-vs-nccl-ep-h100` | 4.0K |  | 早期热身或与本文无关的支线 |

## 磁盘占用与可清理项（未执行，需确认）

`experiments/` 共约 33 GB，Git 跟踪 14,516 个文件。大头是输入张量和构建产物：

| 路径 | 大小 | 性质 | 建议 |
|---|---|---|---|
| `2026-11-19-source-parallel/inputs` | 8.1 GB | 可由种子重新生成的输入 | 下一轮仍要用，保留 |
| `2026-11-18-p1-p2/inputs` | 5.9 GB | P1/P2 实际输入，`verify_release.py` 不依赖 | 保留（完整取证需要），可移到外部盘 |
| `2026-11-14-next-window/cpu-prebuild` | 5.8 GB | 构建产物 | 可删或外移 |
| `2026-10-25-p8-gpu-collection.tar.gz` | 0.99 GB | 与同名目录重复的打包 | 核对 sha256 后可删其一 |
| `2026-10-25-regime-gpu-validation.tar.gz` | 0.54 GB | 同上 | 同上 |
| `2026-10-28-paired-gpu-readiness.tar.gz` | 0.24 GB | 同上 | 同上 |
| `2026-11-18-p1-p2/payload*` | 1.0 GB | README 注明"上传包副本，不是正式结果" | 可删或外移 |

如果要做物理重排（例如把 P/O 类移到 `experiments/archive/`），需要同时改论文脚本里的路径
并重跑 `check_revision_numbers.py`，建议放到 GPU 轮之后统一做。
