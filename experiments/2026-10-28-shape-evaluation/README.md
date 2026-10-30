# 形状评价闭环与共享候选评分

2026-10-28。本轮完成本地代码、协议冻结和一次历史开发试验；**没有新 GPU 数据，没有通过预测可靠性或真实 MoE 验收**。默认编译器和旧结果不变。

## 先明确采用什么训练方式

[NeuSight](https://arxiv.org/html/2407.13853v3#S6.SS1)的 AdamW、epoch 和正则化属于优化设置，关键还包括它的 tile 表达、训练目标和独立评价。[Laya 的固定源码](https://github.com/NandhaKishorM/laya/tree/ffc142012af63d015fa19f05f29f9c5ff329a2cc)提供共享选项评分、结构化决策与选项重排的参考；[Jev 官方说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)介绍 RLCD，但没有提供足够信息让本项目复现其训练配方。Laya 当前微调文档还把概率校准数据在训练前分出。本轮没有概率校准结果。

我们实现数值候选评分：同一个函数评价 host、合并 host 和各 q/g 候选。31 个输入包括形状、P、SM 数、执行属性、tile 数和尾部、K/q、参考轮次、逻辑工作量及交互项。未知 cuBLASLt 几何使用缺失标志，不能继承 CUTLASS tile。P/SM 是输入并不代表跨硬件泛化已经成立；模型拒绝不同执行环境或 GPU 数直接混用。

三个预先指定版本使用相同输入与划分：32 参数线性耗时模型、32 参数线性决策模型、296 参数共享网络（8 个 tanh 隐单元和线性支路）。优化固定为 AdamW、100 轮、学习率 0.003、权重衰减 0.001；网络仅两个固定种子，由训练目标选取，保留两者日志。全批次按完整形状训练，单次拟合限 10 秒、回放限 120 秒。不进行超参数搜索或按评价结果追加轮次。

决策目标是 `L_logtime + L_expected_regret + L_best_gap`：

- 耗时项使用 log(us) 平方误差，每个形状等权，其中 host 类和 fused 类各占一半。
- 选择项对已测候选精确求和 `Σ p(c)·(T_c/min(T)−1)`；softmin 温度固定 0.05。
- 最优差距项使用 `mean_c [max(0, log(T_c/T_b)−(s_c−s_b))]^2`；若实测最优并列，对全部最优候选取平均。

未测候选不填标签。softmin 只是训练权重，不称为“成为最优的真实概率”。以上损失是本项目的设计，不是 Laya 原有损失，也不冒充 Jev RLCD。当前评分头是**经验模型**；独立服务时间尚缺，不能把它称为已经识别好的物理模型加残差。NeuSight 的结构表达和条件依赖汇总保留，不能用训练收益替代 overlap 机制证据。

## 三层形状评价

1. **原四点只做机制诊断。** 顺序 P/M/K/N 为 8/64/4096/2048、8/64/8192/1024、8/256/8192/512、8/512/8192/512。前两点同时改变 K 和 N；后两点改变 M 后，FLOPs、输入/输出体积、tile 数和参考并行度也一起变化。完整变化表已生成，不能把它们描述为只隔离一个原因。旧误选、失败配置和旧执行身份保留。
2. **六点用于最后边界检验。** 下表在没有新耗时标签的情况下冻结。模型、binary、环境身份仍须在打开标签前另行冻结，当前 `independent_evaluation_ready=false`。
3. **一个真实 MoE 检查应用代表性。** 延续 Qwen3-Next-80B-A3B-Instruct 的既定 revision 和 EP8/BF16 范围；读取真实 router 输出，不用均匀或随机路由代替。引擎 commit、请求语料和真实 hook 验证尚缺。

| 边界问题 | 冻结的 M/K/N（P=8） | 必须同时报告的变化 |
|---|---|---|
| M tile 尾部 | 56/3072/776；72/3072/776 | M 跨 64 行边界；固定 N 仍有列尾部；有效工作和 M tile 数一起改变 |
| 参考容量跨界 | 64/3072/3328；64/3072/3584 | q4/g1 的 CTA 数 104→112，跨 108 参考容量；N、工作量也改变；不是实测 occupancy |
| 同 MN、不同长宽比 | 320/6144/768；96/6144/2560 | MN 一致；M/N tile 排列、尾部和输入体积不同；不预设 host/fused 胜者 |

冻结前检索两仓库 6,879 个文本文件，约 509.5 MB，未找到上述 M/K/N 的显式数字组合；未跳过大文件。另核对原 P4/P8 有限轴取值。`exposure-inventory.json` 保留逐文件哈希和检索局限：正则不能证明不存在隐式生成、外部私有记录或人的先验知识。因此只能声明“在当前可查记录中未用于设计/测量”，不能声称随机抽样或广泛泛化。

候选固定为原 10 个 fused 和 3 个明确身份的 host。选择块和验证块分离。六点全候选各 3 个验证块，共 234；最终模型 k≤3 的选择最多 18 块，二 host 控制最多 12 块，等预算搜索最多 18 块。加四点诊断 156 块和已有两点流水化配对 12 块，上限 **450 个候选测量块**，不是 450 次整机启动；实际时长必须由 smoke 计时和租机截止时间控制，不能保证一小时容纳。此协议没有授权新租机。

验收同时看平均/最坏硬选择损失（≤1%/≤3%）、相对训练固定顺序及强 host 控制是否退步、host→fused 与 fused→host 两种误选、top-1/2/3 保留率、耗时误差、搜索加执行总成本和顺序稳定性。精确平局沿用原始 `host` 优先，其余按稳定候选 ID；所有候选先排序后评分，输入遍历顺序不进入网络。如果六点没有出现两种超过 2% 的实测胜负方向，就写“双向覆盖未建立”，不看完标签再挑有利形状替换。任何失败候选都保留；候选面不全不能通过全候选 oracle 验收。若看完结果后改模型，这批形状转为开发数据。

## 本地试验：有改善，但仍失败

使用原 P4 的 41 个形状、346 条实测候选，完整形状按固定哈希分四折。全体旧数据都已影响研究设计，所以只是开发试验；**该划分与旧论文/旧选择器报告不同，不能直接横比数值**。未使用新增六点标签，未使用新版 runtime 或 P8/BF16 标签。

| 同划分版本 | 平均选择损失 | 最坏选择损失 | host 更快却选 fused | 反向误选 |
|---|---:|---:|---:|---:|
| 线性耗时 | 3.41% | 19.62% | 3 | 0 |
| 线性决策 | 3.61% | 28.21% | 3 | 0 |
| 共享小网络决策 | 1.89% | 19.62% | 3 | 0 |

所有版本候选逆序评分一致。三个 host 优势点仍是 M64/K8192/N2048、M64/K4096/N2048、M512/K8192/N512。损失下降没有恢复双向边界；没有理由据此把网络设为默认，也不能据此宣称“再加参数就会解决”。训练目标均下降，只说明这 100 步优化没有恶化初值，不是优化已收敛或达到全局最优。这个试验保留为失败对照，下一步优先补同执行身份下的强 host、服务与依赖证据。

## 真实 MoE 的统计接口

`router_histogram()` 接收 post-top-k、pre-dispatch 的真实 expert IDs；只允许诊断窗口，Tensor 拷回 CPU 不进入稳态计时。每个工作负载固定取前 3 个完整 prefill 和前 3 个完整 decode step，覆盖全部 MoE 层和 EP 来源 rank，空 rank 也登记。每个 envelope 对应一个工作负载/重复，不能把独立 DP replica 当成 EP 来源；实际引擎须提供一致的全局 step/layer ID。

输入包含模型 revision、引擎 revision、语料哈希、采样规则、hook 来源、逻辑投影、各 rank 的 token 数和专家直方图。后处理检查 `Σ专家计数 = tokens×top_k`、所有来源与层齐全、重复 rank、专家范围；缺项就失败，不把局部统计当全局分布。`kernel_shapes` 只有从后端实际采集时才填；禁止用 ceil(M/tile) 自行伪造。

输出区分逐源专家 bucket M、合并后的专家逻辑 M、实际 padded kernel M；按 prefill/decode 与 gate/up、down 投影报告出现次数。分别计算形状次数、路由行数、逻辑 FLOPs 加权的精确 M/K/N 覆盖，并注明分母和采样范围。主比较覆盖原论文 **23 个已测 P8 形状**；四个诊断点和六个新边界点另列为计划集合。shape 相同仍不代表通信分布、dtype 或完整执行身份相同；时间覆盖率保持 `null`。shared expert、attention、dispatch、combine 不在“routed expert projection”分母里。

当前没有真实 trace，只有明确标记 `synthetic_test` 的单元测试。统计适配器已完成；生产引擎 hook、全模型资产准备和 GPU 运行未完成，不宣称真实代表性已经验证。

## 入口

- 冻结协议：[frozen/protocol.json](frozen/protocol.json)，几何与特征：[frozen/descriptions.json](frozen/descriptions.json)。130 个描述是 10 个形状×13 候选，全部新计划行都不是新实测。
- 训练实现：[shape_decision_model.py](../../EDM-IR/edmir/compiler/shape_decision_model.py)，历史试验：[historical-pilot/results.json](historical-pilot/results.json)。
- 真实路由统计：[moe_shape_profile.py](../../EDM-IR/edmir/benchmark/moe_shape_profile.py)，检查：[shape_evaluation_test.py](../../EDM-IR/tests/shape_evaluation_test.py)。
- 既有失败记录的哈希保存在协议中，不重写原始实验。

```bash
# 新输出目录，拒绝覆盖历史试验；只需要 CPU / NumPy。
python3 EDM-IR/tools/review_shape_decision.py --out /tmp/shape-decision-review-new
python3 EDM-IR/tests/shape_evaluation_test.py
python3 EDM-IR/tools/profile_moe_shapes.py --trace actual-router.json \
  --protocol experiments/2026-10-28-shape-evaluation/frozen/protocol.json \
  --out /tmp/actual-moe-shapes.json
```

`actual-router.json` 尚不存在，第三条只在实际诊断采集后执行。输入字段与验证规则由 `summarize()` 定义，测试夹具只解释数据格式，不能当作论文数据。端到端收益仍须实际集成后验证。
