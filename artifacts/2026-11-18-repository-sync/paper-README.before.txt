# Communication Exposure and Execution Costs in Direct-Peer GEMM

2026-11-17 应用案例与叙述整理版。正文按执行机制、数值准入后的候选收益、固定q4/g2与共享评分比较、真实MoE工作负载组织。新增BF16 GLM-4.7-Flash原生vLLM性能表、完整阶段路由表和专家行数分布图。

- 最新PDF：[更新后的论文](../output/pdf/edmir-moe-narrative-2026-11-17.pdf)。
- [正文修改与验收](review/MOE_NARRATIVE_REVISION_ZH.txt)
- [终稿取舍原则](review/SUBMISSION_WRITING_POLICY_ZH.txt)
- 代码、数据与后续清单：相邻代码仓库 `../../llm-serving/artifacts/2026-11-17-paper-closeout/`。
- 本轮应用数据副本：[provenance.json](data/2026-11-17-real-moe/provenance.json)。包含来源和SHA256，表图重画不依赖相邻仓库。
- 生成应用表图：`make real-moe-results PYTHON=python3`（需要reportlab）；只读JSON，不训练或调用GPU。
- 算子图表：`make final-operator-revision PYTHON=python3`。
- 编译：`make review-pdf`。所有原始实验和失败记录保留在代码仓库。

当前PDF仍为MLSys排版研究稿。未执行的六形状协议保留原文件但不编入PDF，预测器及冻结系数不变。完整历史测量/方法仍在附录；本文将开发诊断与应用原生基线分别解释。FGCS模板、作者信息与提交附件在正文确认后统一处理。

以下为历史版本与复核入口，不代表当前实验状态：

首投目标已确定为 **FGCS**。投稿规则核验状态、模板迁移与材料准备见 [FGCS 投稿记录](review/FGCS_SUBMISSION_ZH.md)；当前 PDF 仍为 MLSys 样式研究初稿。

- 10 月 28 日 PDF：[output/pdf/edmir-shape-evaluation-revision.pdf](output/pdf/edmir-shape-evaluation-revision.pdf)。
- 对应源码包：[output/edmir-shape-evaluation-revision-source.zip](output/edmir-shape-evaluation-revision-source.zip)。
- 本轮代码、冻结与失败状态：[review/SHAPE_EVALUATION_ZH.md](review/SHAPE_EVALUATION_ZH.md)。
- NeuSight 阅读、适配与本地反例：[review/NEUSIGHT_ADAPTATION_PLAN.md](review/NEUSIGHT_ADAPTATION_PLAN.md)。
- 有界机制与流水化对照：[review/P8_MECHANISM_EVIDENCE_PLAN.md](review/P8_MECHANISM_EVIDENCE_PLAN.md)。
- 上一轮阅读记录：[review/LITERATURE_REVISION_ZH.md](review/LITERATURE_REVISION_ZH.md)。
- 源文件：`main.tex`、`sections/`、`references.bib`。
- 原始论文数据：`data/a100_grouped_n/`（P4）、`data/a100_p8_transfer/`（P8）、`data/selector_diagnosis/`。
- 本轮证据副本：`data/2026-10-25-revision/`，含来源路径和 SHA-256；重画本轮图表不依赖相邻代码仓库。
- 10 月 25 日数据修订与验收：[review/EXECUTION_REVISION_ZH.md](review/EXECUTION_REVISION_ZH.md)。

```sh
# 复核全部本轮图表数字，并生成两张矢量图及完整比较表
make execution-revision PYTHON=python3
# 重画开篇执行路径图；只使用构造的依赖示意，不读取测量数据
python3 scripts/generate_communication_story.py
# 编译当前稿件；需要 LaTeX、BibTeX、latexmk
make review-pdf
```

`execution-revision` 需要 `requirements-figures.txt` 的 reportlab。正文已包含生成结果，单纯编译无需 Python。
`make review-pdf` 写入通用文件 `output/pdf/edmir-draft.pdf`；本轮交付另存上述命名版本。

旧图和诊断保留原数据及脚本，可用 `make figures` 全部重画；`scripts/audit_advisor_numbers.py` 与 `scripts/analyze_selector_failures.py` 只核对早期论文对应数据，不能替代本轮回放验证。
`templates/` 和 routed 数据处理脚本供后续实验使用，不是已完成的 GPU 结果。
