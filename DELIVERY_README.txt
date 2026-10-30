EDM-IR 源码与实验交付入口（2026-10-30）

当前边界
- 保留的 P4/FP16 开发回放：3/41 误选，平均/最坏选择损失 0.0802%/1.5801%。
- 这些数据多次参与设计；不是独立泛化成绩，也不是 P8 或真实 BF16 MoE 收益。
- 新增流水线依赖、操作数复用和资源服务表达用于诊断，未替换编译器准入策略。
- 冻结六形状未用于本轮训练或评分。后续先固定 P8 policy，再独立评价。
- 真实模型资产已备好；BF16 GPU 子图、真实路由 hook 和引擎 A/B 仍待完成。

核心代码（均在 EDM-IR/edmir/compiler/）
1. 执行计数/身份：tile_execution_model.py、execution_ledger.py、execution_policy.py、host_baseline.py。
2. 开发评分器与对照：conditional_curve_selection.py、split_stage_selection.py、parallel_depth_selection.py。
   模型具体变体和参数以对应实验 JSON 为准；shape_transfer_selection.py 的新对照未获晋升。
3. 共享候选/学习分支：shared_candidate_selection.py、shape_decision_model.py、
   precision_candidate_selection.py、calibrated_candidate_selection.py。保留历史失败分支，不默认采用。
4. 条件机制表达：dependency_wave_model.py、operand_reuse_model.py、pipeline_dependency_model.py、
   resource_service_simulator.py。没有实测服务量时不输出伪造的 GPU 微秒标签。
5. 准入、证据与解释：fused_selection.py、prediction_support.py、prediction_explanation.py、overlap_evidence.py。
6. 原始编译路径：program/dependence/simulator/verifier 相关模块及 backends/。

执行与实验
- EDM-IR/runtime/：C++/CUDA、CUTLASS/NCCL 路径；host 基线和 fused 实现。
- EDM-IR/edmir/benchmark/：routed CPU 参考、计时记录合同与 MoE 形状统计。
- EDM-IR/tools/：采集包生成、候选测量、数据整理、离线评价和路由统计入口。
- EDM-IR/tests/、EDM-IR/runtime/tests/：数值合同、选择、顺序不变性、超时与 CPU 运行检查。
- experiments/2026-10-28-conditional-curves/、2026-10-28-split-stage-review/：评分表达与分片阶段。
- experiments/2026-10-28-shape-transfer-review/：当前保留版本与未采用的迁移对照。
- experiments/2026-10-29-operand-reuse-review/、2026-10-29-pipeline-dependency-review/：最新机制边界。
- experiments/2026-10-28-p8-session/：八卡已有采集、数值失败与恢复记录。
- experiments/2026-10-29-cpu-preparation/：模型/软件固定版本及离线准备代码。
- EDM-IR/docs/NEXT_P8_AND_SERVING_PLAN_2026-10-29.md：下一轮采集及实际 serving 的验收顺序。

最短参考清单：papers/REFERENCES_2026-10-30.txt。
论文源码在独立仓库：https://github.com/Neo82-z/paper 。

复核约定
- Python 基础要求 >=3.10；拟合/神经对照另见 EDM-IR/pyproject.toml 的可选依赖。
- 每个 Python 测试文件用独立进程运行，并把 EDM-IR 加入 PYTHONPATH。
  一次进程的 unittest discover 会受到历史 bundle 动态加载的 sys.path/模块名冲突影响。
- CPU CMake/CTest 不验证 CUDA kernel 的运行正确性；LaTeX 可编译也不证明研究结论通过。
- 本次测试回执随交付包保存；不会通过运行打包脚本启动 GPU、训练或评分冻结形状。

归档边界
- 源码、实验脚本、配置、小型原始计时/整理结果、设计与失败记录同步 Git。
- 大型压缩备份、权重、激活二进制、profiler 原始 trace、编译产物及云账户 UI/连接记录留在原位置。
- 历史源码快照保留用于解释执行身份；新修改以顶层 EDM-IR 为准。
- NCCL/CUTLASS 的嵌套依赖 checkout 不作为无配置 gitlink 提交；使用实验清单中的 URL/commit 获取。
- 因未附所有大文件，源码包不是完整的 GPU 原始数据镜像。已有复现脚本如需未附输入，按原清单重建/获取。
