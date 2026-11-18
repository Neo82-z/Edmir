EDM-IR 当前审阅入口（2026-11-17）

先看 NEXT_STEPS_ZH.txt：剩余证据及是否值得补，只有两个可选GPU实验，不自动执行。
再看 CODE_INDEX.txt 和 DATA_INDEX.json：定位当前实现与各执行身份的数据。

论文：../llm-serving-mlsys-latex/latex/main.tex
本轮修改以机制和完成的结果组织，增加真实BF16 MoE工作负载章节；固定q4/g2比较和large-N冻结复测保留。未执行的六形状协议仍留原文件，未编入本轮PDF。

代码审阅：EDM-IR/runtime → 封存w1_candidate_selection.py → benchmark/real_moe.py/vllm0201_capture.py。
数据审阅：11月16日主算子窗口 → 11月17日最终large-N → 同日真实MoE原生基线/独立路由诊断。
不要将历史神经网络尝试当成当前评分器，不要把原生vLLM性能当成EDM-IR端到端收益。

本地复核
1. 真实数据分析：在代码仓库运行 experiments/2026-11-17-real-moe-analysis/analyze.py。只用Python标准库，但须有DATA_INDEX指向的原始recovered目录。
2. 表图：在论文latex目录运行 scripts/generate_real_moe_results.py（reportlab），数据副本自包含，不要求相邻代码仓库。
3. 论文：在latex目录运行 latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex。
4. 导出包的SOURCE_MANIFEST.json用于逐文件SHA256校验，VERIFY_PACKAGE.py在解压后的根目录执行。

审阅包包括：核心Python/CUDA/C++源、相关工具与测试、封存评分文件、结果摘要/分析脚本、原始回收清单、论文源/图/表/数据与PDF。
审阅包不复制约1.14GB的真实MoE原始轨迹，也不复制旧窗口完整日志；它们保持在原仓库，由DATA_INDEX及回收清单定位。故该包可独立编译论文和重画表图，完整原始数据重算需额外取得相应recovered目录。
审阅包不包含模型权重、镜像、云账号/SSH配置、日志中心部署状态，不启动任何付费任务。

本轮没有修改runtime/采集器/预测器逻辑，没有重新训练、评分或GPU采集。新增的只是索引、数据副本、论文表图生成与叙述。
