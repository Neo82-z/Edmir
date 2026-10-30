# 精度契约、交叉熵与 Dion3 评分对照

2026-10-28。已修改评分模型并完成本地 CPU 训练；未启动云端服务器。六个冻结形状及既有标签不变。本轮没有新增 BF16 GPU 数据，也没有替换编译器默认选择器。

## 修改了什么

新增 `EDM-IR/edmir/compiler/precision_candidate_selection.py`，继续共享同一个小评分函数。31 个原有几何特征加 9 个精度描述，单隐藏层宽度 8，共 377 个参数。旧模型文件保留。

精度现在分为 input、weight、accumulator、partial、output；同时区分源码确定的累加方式与 cuBLASLt API 请求的 FP32（库内中间过程未知）。FP16 与 BF16 虽然都是两字节，但有效位数、指数范围不同；FP32 累加也不意味着 split-K 每段写回的是 FP32。加入这些字段用于表达与防止混用；现有数据输入类型恒为 FP16，**不能声称学到了 BF16 的性能差异**。未出现过的精度组合直接拒绝评分。

正确性证据绑定形状、候选、执行环境、精度、输入与实现身份，换输入或 dtype 不能继承原来的通过状态。失败候选不参与 softmax、交叉熵、排名或耗时回归，标签保持缺失；不会被编码成“很慢”。证据未知时不自动选择一个默认候选。这个绑定保证记录完整性，不是数学正确性证明。

评分网络权重与前反向使用 FP32，便携推理使用 NumPy 在 FP32 权重上做 float64 算术。Dion3 官方默认 Polar Express 正交化内部使用 BF16，已单独记入元数据；这与被预测算子的 dtype 是不同层次。

## 优化器和损失

更正：用户随后明确指的是 **Muon**；本轮将名称猜成 Dion 是助手的判断错误。以下实际接入和结果仍是 [Dion3](https://github.com/microsoft/dion/tree/7692479288f928a3d61d1cc58137b4dbe4f71000)（NorDion2 的导出别名），对应 [Dion3 论文](https://arxiv.org/abs/2608.11612)，不能当作 Muon 实验。没有自行近似实现。安装文件逐一与该 Git 提交核对。隐藏层矩阵使用 Dion3，标量评分头、仿射分支及 bias 使用 AdamW；无分布式训练收益的声明。后续归因见[Muon 阅读与模型审查](../2026-10-28-muon-model-audit/README.md)。

训练保留原 log-time MSE、实测期望选择损失和 best-gap 项，增加可消融的交叉熵：

$$L=L_{time}+L_{cost}+L_{gap}+\lambda_{CE}\,[-\sum_{c\in C_{valid}}q_c\log p_c],\qquad q_c=\frac{\exp(-\log T_c/\tau)}{\sum_j\exp(-\log T_j/\tau)}.$$

`tau=0.05` 在训练前固定，`lambda_CE` 只比较 0 和 1。近似平局得到相近的软目标，未测/失败候选不补标签。`p` 是选择权重，`q` 是由实测代价构造的目标，均不是“真实最优概率”。同时报告 CE 与 KL=CE−目标熵，避免把不同候选数造成的熵差当作进步。这是本项目的损失组合，不是 Dion3 自带的损失。

冻结的 2×2 对照为 AdamW/Dion3 × CE0/CE1；相同模型、1000 步、两个初始 seed、基础学习率 0.003、梯度裁剪与标量 AdamW 参数。Dion3 矩阵部分使用官方更新和 RMS 学习率归一化，fraction=0.5；因此是固定预算的优化器对照，不是各自充分调参后的最优比较。只按训练目标选择 seed。每 seed 最长 20 秒，每子进程 50 秒，总轮次预算 600 秒，禁止测试结果驱动的反复重跑。完整 32 次拟合任务约 85.5 秒，均成功。

## 留出结果

P4：沿用原来的 41 个整形状四折、346 个已测候选；每折使用此前仅在该训练折拟合的代价底座，残差在训练 log(M,K,N) 凸包内生效。P8：四个历史开发形状分别留一，60 次有效进程合并为 20 个配置标签，6 次失败明确排除并保留；三次重复不是三个形状。P4 和 P8 不混用参数或执行身份。

| 数据 | 优化器 / CE权重 | 平均选择损失 | 最大选择损失 | 平均 CE | host→fused / fused→host 误选 |
|---|---|---:|---:|---:|---:|
| P4 | adamw-ce0 | 0.5311% | 7.7978% | 1.2829 | 0 / 0 |
| P4 | adamw-ce1 | 0.5440% | 7.3566% | 0.9227 | 0 / 1 |
| P4 | dion3-ce0 | 0.5019% | 7.3566% | 1.3273 | 0 / 1 |
| P4 | dion3-ce1 | 0.4751% | 7.3566% | 0.9060 | 0 / 1 |
| P8 | adamw-ce0 | 0.0000% | 0.0000% | 1.6862 | 0 / 0 |
| P8 | adamw-ce1 | 0.0000% | 0.0000% | 1.6275 | 0 / 0 |
| P8 | dion3-ce0 | 0.0000% | 0.0000% | 1.7611 | 0 / 0 |
| P8 | dion3-ce1 | 0.0047% | 0.0189% | 1.6988 | 0 / 0 |

**判断：不因优化器较新就自动切换。** P4 上 Dion3+CE 的平均损失约 0.475%，与之前受约束残差约 0.475% 相当，仍逊于先前纯代价底座约 0.361%；7.36% 最坏损失未解决。交叉熵确实下降，未转化为全面更好的 top-1 选择。P8 上 AdamW 两种损失都选中该子集中的实测最佳，Dion3+CE 在一次近似平局中损失 0.0189%。四点均为有效 fused 占优，不能验收双向边界，更不能替代冻结六形状。

M256 留出进一步暴露分数问题：q2/q8 实测约 995.6/988.6 μs，Dion3+CE 预测约 1864.4/1092.9 μs。选中 q8 没错，但把差距夸大了；该形状 CE=4.96，说明不能只用 top-1 为零损失宣布模型可靠。没有为压低这个数字继续调温度、增轮次或修改样本。

## FP16 / BF16 的具体反例

在上一轮 q16 失败的单个元素上，用原始输入做本地数值重算。该输入是 integer/32，在两种格式中均精确可表示。FP32 参考为 0.8623046875：

| 输入/输出 | 分段中间存储 | 本地模拟结果 | 绝对误差 |
|---|---|---:|---:|
| FP16 | FP16 | 0.841796875 | 0.0205078125 |
| FP16 | FP32 | 0.8623046875 | 0 |
| BF16 | BF16 | 0.8515625 | 0.0107421875 |
| BF16 | FP32 | 0.86328125 | 0.0009765625 |

BF16 在这个元素上的误差较小，不表示整个 BF16 内核通过。这个例子支持继续检查中间存储精度，而不是只切换输入 dtype。FP32 partial/reduction 需要新的内核、内存和执行代价核算；表中模拟不能作为新 GPU 标签或提速结果。

## 文件与复现

- [固定设计及全部失败/成功结果](run-01/results.json)、[训练前设计](run-01/design-before-fit.json)。原运行源码已按哈希保存在 `run-01/source-snapshot/`。
- [当前评分实现](../../EDM-IR/edmir/compiler/precision_candidate_selection.py)、[检查入口](../../EDM-IR/tools/score_precision_candidates.py)、[固定对照脚本](../../EDM-IR/tools/review_precision_scoring.py)。
- [P4 Dion3+CE 开发模型](development-p4-dion3-ce1.json)、[P8 AdamW+CE 开发模型](development-p8-adamw-ce1.json)、[P8 Dion3+CE 开发模型](development-p8-dion3-ce1.json)。这些全量拟合文件仅用于检查，不用训练内分数证明泛化。
- [精度与证据绑定数据](datasets-with-evidence-contracts.json)、[BF16/FP16 数值模拟](precision-witness.json)、[依赖和官方源码核对](dependency-provenance.json)、[精确依赖版本](requirements.txt)。
- 训练后加固了证据绑定与显式 FP32 构造，未改损失或特征；[180 次形状级对照](binding-regression.json)的保存分数及选择逐项保持一致，没有因此重做超参数搜索。
- 26 项相关测试通过：新分支 9 项、上一版边界模型 8 项、冻结评分模型 9 项；覆盖梯度、屏蔽、dtype 拒绝、证据不可挪用、候选顺序、官方优化器分组及 NumPy/PyTorch 推理一致性。
- [遍历性能](inference-timing.json)：41 个形状/346 候选，冷运行约 10.0 ms，20 次暖运行中位数 8.8 ms，最大 14.6 ms。仅是本地 CPU 指标。

安装可选依赖 `EDM-IR[selector-neural]` 或使用本目录 `requirements.txt`；本次验证环境为 Python 3.12.14、PyTorch 2.14.1、NumPy 2.3.5、SciPy 1.18.1。

```bash
python EDM-IR/tools/score_precision_candidates.py \
  --model experiments/2026-10-28-precision-scoring/development-p8-dion3-ce1.json \
  --data experiments/2026-10-28-precision-scoring/datasets-with-evidence-contracts.json \
  --shape 8 512 8192 512
```

该例显式列出 q16/g2 数值失败并不给分，q8/g2 为已测有效集合中的选择，同时标记 `training_shape=true`、`compiler_admission=false`。

下一步继续收敛到：修复/定义精度合格的候选执行契约，解决 M256 的差值夸大及 M512 的外推失败，再做未参与设计的形状验证。当前证据不足以把 Dion3+CE 升为默认策略；原经验代价基线继续保留。
