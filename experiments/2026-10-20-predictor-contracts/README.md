# 预测器：特征耦合、grouped-N 适用范围与 host 身份

这轮完成了本地代码修改、历史数据诊断和新测量的冻结协议，**尚未获得独立 GPU 数据，也没有证明预测器可靠**。旧的 599 条完整算子记录仍是开发数据；不改动原始时间，不把旧 P8 “heldout” 重新称为独立测试。

当前结论：保留原 relative 模型作为比较项，增加明确的 grouped-N 适用范围；新拆分模型只用于诊断。拆分模型回放明显退步，不能替换现有模型，更不能接入编译器自动选择。

## 代码入口与产物

| 文件 | 内容 |
|---|---|
| [host_baseline.py](../../EDM-IR/edmir/compiler/host_baseline.py) | host 的库、kernel、M/PM 分块、通信和计时边界 |
| [factorized_selection.py](../../EDM-IR/edmir/compiler/factorized_selection.py) | 三个时间比值分别拟合；grouped-N 的训练支持范围 |
| [measurement_dataset.py](../../EDM-IR/edmir/compiler/measurement_dataset.py) | 读取原归档并显式绑定 per-source CUTLASS；原数据不改名 |
| [fused_selection.py](../../EDM-IR/edmir/compiler/fused_selection.py) | 测量 host 与实际回退 host 不一致时拒绝性能准入 |
| [evaluate_factorized_selector.py](../../EDM-IR/tools/evaluate_factorized_selector.py) | 耦合、成对 grouped-N 现象、旧/新模型的同协议回放 |
| [independent_selector_validation.py](../../EDM-IR/tools/independent_selector_validation.py) | 先冻结预测，再检查完整原始记录并评分；不执行 GPU kernel |
| [development_results.json](development_results.json) | 本轮全部历史回放、模型参数、逐点选择、来源哈希 |
| [independent_p4_plan.json](independent_p4_plan.json) / [independent_p8_plan.json](independent_p8_plan.json) | 新网格、固定选择、随机测量顺序、候选与来源哈希 |

## 1. 特征耦合：先分清恒等式与拟合假设

旧模型有以下计数关系：

```text
log2(Gamma) = log2(ceil(M/64)) + log2(rho_N) + log2(q)
log2(K/q) = log2(K) - log2(q)
local_bytes 已含 reduction_bytes；另外又有 log(reduction_bytes+1)、q-1。
```

这些量本身没有算错，但不能把它们的回归系数逐一解释成独立物理作用。P4 的候选特征矩阵含截距有 14 列，所有 M 合并时秩为 14；限制到 M=64 后秩为 12。`grouped` 与 `grouped × log2(K/q)` 的全 P4 相关系数为 0.965。不能据此声称整个旧模型全局奇异；ridge 仍可给出唯一解，但不保证系数的物理可解释性或外推能力。

新诊断模型使用：

```text
log[T(q,g)/T(host)]
  = log[T(1,1)/T(host)]
  + log[T(q,1)/T(1,1)]
  + log[T(q,g)/T(q,1)]
```

对同一工作负载的实测时间，这是一条恒等式；用三个回归器预测右侧，是新的统计假设。

| 分量 | 训练标签 | 输入与约束 |
|---|---|---|
| direct/host | 每个 workload 的 direct 与同一 host 成对 | log M、log K、log N |
| split/direct | 同一 workload 的 q 与 q=1 成对 | log q、log q×log(K/4096)、log q×log(base_grid/64)、(q−1)×4096/K；q=1 时严格为 0 |
| grouped/plain | 同一 P/M/K/N/q 的 g=2 与 g=1 成对 | log[(K/q)/4096]、log(base_grid/64)，含截距 |

`base_grid=ceil(M/64)ceil(N/128)`，不含 q、g。上述特征是候选参数化，不能等同于完整硬件成本方程。三项分别标准化，固定 ridge=1；每项每个 workload 总权重为 1，没有根据 P8 结果搜索超参数。grouped 标签改变不会改变前两项拟合；整体时间缩放会在所有比值中抵消。

P8 四点校准对拆分模型只修正 direct/host 的常数偏置，不重拟合 q/g，也不扩大 grouped-N 范围。旧 relative 仍用全部 fused/host 残差拟合偏置，因此两者校准对比不是只改损失函数的消融。

## 2. grouped-N 跨 M：有方向反转，但样本不足以给通用边界

P4 的 grouped-N 训练证据只有 M=64：20 个 workload、每个 q 有 20 对测量，共 100 对。仅 4 对的 g=2 时间低于 g=1。

P8 历史数据能看到反转。例如固定 P=8、K=8192、N=512、q=4：

| M | T(grouped-N)/T(plain) | grouped-N 的耗时变化 |
|---:|---:|---:|
| 32 | 1.1527 | 增加 15.27% |
| 64 | 1.1503 | 增加 15.03% |
| 256 | 0.8630 | 减少 13.70% |
| 512 | 0.6085 | 减少 39.15% |

所有 P8 匹配的跨 M 对照中，45 个固定 P/K/N/q 的集合有 7 个出现方向反转。M=256 和 M=512 各只有一个 K/N 工作负载，不能据此断言“大 M 都应该 grouped”，也不能给出可靠的临界 M。这些仍是历史点估计，没有独立重复测量置信区间。

本轮代码把**选择资格**限制为：该 q 在训练中有至少两对 grouped/plain 测量；M 必须精确出现过；K、N 落在该 M/q 的观测范围内。当前就是 M=64、K∈[512,8192]、N∈[128,1024]，q∈{1,2,4,8,16}。M=32 和 64 虽都只有一个 64 行 tile，也不互相替代。

这只是拒绝明显无依据外推的规则，**不是域内精度保证，也不是 kernel 合法性约束**。新机器/P 的迁移仍是显式实验；校准偏置不增加 grouped 训练样本。范围外只移除 grouped 候选，保留 host 和未分组的 fused 候选。

评价始终保留完整已测候选的 oracle，同时报告范围内 oracle、候选覆盖率和限制本身损失的性能，避免靠缩小分母美化结果。

## 3. host 身份统一：共同接口，分别保留实现

| 身份 | GEMM 划分 | 当前证据/实现状态 |
|---|---|---|
| `host_nccl_cutlass_sm80_per_source_v1` | 每 rank 顺序执行 P 个 M×K GEMM | 旧数据真正的 host |
| `host_nccl_cublas_per_source_v1` | 同样 P 个 M×K GEMM | 编译器实际回退；不能套用 CUTLASS 的时间 |
| `host_nccl_cutlass_sm80_merged_v1` | 接收后一个 PM×K GEMM | 新验证需要；本轮没有实现其采集器 |
| `host_nccl_cublaslt_merged_v1` | 接收后一个 PM×K GEMM | 新验证需要；本轮没有实现其采集器，须记录算法 ID |

所有身份都明确 FP16、row-major、host NCCL send/recv 与本地复制、先物化后计算，以及 `execute()+synchronize()` 外的 CPU wall timer。必要复制、重排、split-K reduction 必须在计时范围内；初始化与算法选择在计时外，搜索预算与配置另记。驱动、库版本、提交、拓扑另属于 environment 指纹，不能靠相同 baseline ID 视为相同机器。

编译器现在要求 `reference_host_baseline_id` 与实际 cuBLAS 回退一致。缺失或不一致时，即使 `--allow-experimental-fused` 也回退。保留的 A100 profile 已如实标成 CUTLASS，不能继续批准相对 cuBLAS 的性能选择；没有把旧时间改成 cuBLAS。选择报告同时列出参考 host 和实际回退身份。用于开发的显式 fused lowering 仍是另外的入口。

## 4. 回放结果：这轮没有获得更准确的新模型

以下是 mean regret = 平均[T(selected)/T(最快已测候选)−1]。缺测的候选不填充。所有数据均已被查看过，表格是开发诊断。

| 方法 | P4 留组（41 点） | P4→P8 无校准（19 点） | P8 四点校准后（19 点） |
|---|---:|---:|---:|
| 固定 split-K4，训练集选出 | 5.43% | 9.53% | 9.53% |
| 旧 relative | 4.41% | 13.04% | 8.65% |
| 旧 relative + 适用范围 | 3.54% | 13.04% | 8.65% |
| 拆分模型 + 适用范围 | 25.29% | 35.56% | 34.76% |

P4 的改善来自完整适用范围限制，包括留组后 N 边界收缩；不能归因为“已经学会跨 M 泛化”。P4 留组共有 20 个工作负载提供 grouped 标签，其中 10 个满足各折训练范围。

P8 的 grouped 候选仅在 7/19 点获得选择资格；单是限制候选就使最优可达平均 regret 为 4.37%。但拆分模型在允许候选内仍有 28.78% 的平均 regret，问题明显不只在保守限制。校准后旧 relative 有 2/19 点比旧 host 慢，拆分模型有 6/19 点，后者最坏慢 60.40%。

因此：减少特征耦合改善了可诊断性，却没有解决模型欠拟合、环境迁移、host 调度开销和 grouped 反转。没有部署拆分模型；旧 relative 也不能因 8.65% 这个平均值被称为可靠。

### 追加诊断：退步主要来自 split-K 分量

应用户追问进行了独立数值复核，结果保存在
[error_attribution.json](error_attribution.json)。用 NumPy 2.3.5 的 SVD
最小二乘重新构建设计矩阵、标准化及带权 ridge 增广系统，未调用生产代码的
正规方程求解器。三项系数最大差异分别为 2.22e-16、2.03e-15、2.78e-16，
支持“当前线性求解与标准化没有造成这次退步”，不等于排除了全部程序问题。

在同一 P8 历史集、相同选择资格下，逐项用实测比值替换预测分量：

| 诊断条件 | 平均 regret |
|---|---:|
| 原拆分预测 | 34.76% |
| 只把 direct/host 换成真实比值 | 29.50% |
| 只把 split/direct 换成真实比值 | 6.79% |
| 只把 grouped/plain 换成真实比值 | 33.78% |
| 三项全部换成真实比值 | 4.37% |

**这些替换使用答案，只是误差定位，不是可部署的预测器，也不是新的验证成绩。**
三项全真仍有 4.37% regret，是禁止无训练支持的 grouped 候选造成的可达性能损失。

例如 P=8、M=64、K=4096、N=2048，拆分模型预测 q=2/4/8/16 相对
direct 的时间比依次为 0.906/0.827/0.766/0.731，认为 q 越大越好；
实测却是 0.667/0.674/0.771/1.068。它因此选择 q=16，漏掉 q=2 的最优
未分组方案。split/direct 在 P4 训练集上的加权 log-MAE 已达 0.222，
并不是只有换到 P8 才出现误差。

目前更具体的诊断是：这一轮简化的 split-K 特征与单独拟合目标没有表达好
已有数据中的非单调最优点。继续增加 GPU 样本之前，应先本地修正和验证这项。
由于这轮同时改变了特征、拟合目标和校准方式，尚不能把所有退步归因于
“分解公式”一个因素。模型和已冻结的独立计划均未因这次诊断而改动。

## 5. 独立验证已经冻结，采集仍待完成

优先执行 P4，P8 为预算允许时的配对迁移检验，平台为 A100 SXM4 80GB、SM80、108 SM、完整 NV12 peer access。无需把 H100 作为前提。

每个 P 的冻结网格有 45 点：

```text
M = 48, 64, 96, 192, 384
K = 1536, 3072, 6144
N = 384, 768, 1536
```

所有 M/K/N 组合均不在已查看的 P4/P8 工作负载中，且满足现有 FP16/split-K 对齐要求。M=64 检查已有 M 上的新形状；其余 M 检查跨 M 行为。N=384 有奇数 N tile，N=1536 超出已有 grouped-N 的 N 范围。守范围策略只有 6/45 点允许 grouped，采集仍必须覆盖全部 grouped 候选。

每点 10 个 fused 候选 + 上表 4 种 host，共 14 个。比较策略固定为 host、训练集最优固定家族、旧 relative、旧 relative+范围、拆分模型+范围。预测在测量前写入 JSON；评分只读冻结选择，不能看到新时间后重选、重拟合。

每个 P：45×14×5=3150 条独立进程重复的中位数记录；每条至少 20 次 warmup、100 次正式迭代。这里的记录数不是 GPU 小时估计。每个重复内候选顺序已固定随机化；初始化、参考结果计算和基线算法搜索不进正式算子计时。相同输入的参考结果可预先计算并缓存，采集器仍须检查各候选的完整输出。

新环境不复用旧 P8 校准偏置。P4/P8 两份计划是同一批预先冻结的配对网格，不应被叙述成两次独立设计。如果看过 P4 后改模型，本轮 P8 不能再作为这个新模型的独立工作负载检验。

评分会拒绝：缺候选、缺重复、重复行、修改过的预测或源码、身份/计时单位/精度不符、错误的测量顺序、数值检查失败、输入不一致、原始记录哈希不符。报告包含：

- 相对所有 14 个已测候选的 regret，以及相对四种 host 各自的时间比。
- 相对“最快已测 host”的回退代价；该最快 host 是分析用 oracle，不宣称部署时可免费获知。
- 按 M、N 分层结果，grouped 选择资格覆盖率与范围内 regret。
- 对工作负载和成对重复进行分层重采样的 95% 区间。区间只描述这个网格/计时过程，不代表所有场景。

本轮评分针对完整算子 wall time，未包含 Python 选择器的执行成本，也未证明选择成本能够摊销。部署或论文端到端收益主张还需要单独记录选择/校准成本及其复用次数。

还缺的真实工作：实现并审查 merged host 采集器与统一的完整计时采集器，在新 A100 上完成正确性检查与测量。现有公开 CUTLASS、TileLang、NCCL 单组件数据不能替代本算子的这些标签。没有因此租机器或产生 GPU 费用，也没有生成伪造的独立实验结果。

## 复现与采集格式

在仓库根目录运行；工具拒绝覆盖已有结果：

```bash
python3 EDM-IR/tools/evaluate_factorized_selector.py \
  --dataset experiments/2026-10-20-offline-predictor/data \
  --out /tmp/factorized-replay.json

python3 EDM-IR/tools/independent_selector_validation.py freeze \
  --dataset experiments/2026-10-20-offline-predictor/data \
  --world-size 4 --out /tmp/independent-p4-plan.json

python3 EDM-IR/tools/independent_selector_validation.py score \
  --plan experiments/2026-10-20-predictor-contracts/independent_p4_plan.json \
  --run /path/to/new-run.json --out /tmp/independent-p4-score.json

python3 -m unittest discover -s EDM-IR/tests -p '*_test.py'
```

新 run 的 schema 是 `edmir.independent-selector-run.v1`，需引用冻结 `plan_sha256`。顶层复制计划中的 environment_id、baseline_contracts、timing_scope、latency_unit、dtype；environment 记录真实硬件与 driver/CUDA/NCCL/CUTLASS/cuBLAS 版本、提交、collector 源码 SHA256。`fresh_process_per_repetition=true`。

每条 `rows` 记录包含 shape=[P,M,K,N]、family、repetition（0–4）、trial_order（对应冻结顺序）、median_ms、warmup、iterations，以及：

- `implementation_id`：host 使用注册 ID；fused 使用 `direct_peer_cutlass_sm80_q{q}_g{g}`。cuBLASLt 另有 `gemm_algorithm_id`。
- `correctness_passed=true`、`checked_elements=P*P*M*N`、`nonfinite_outputs=0`、max_abs_error、max_rel_error、atol=0.02、rtol=0.02。
- `correctness_reference=CPU_FP32_on_identical_FP16_inputs`、input_seed=20261020、input_sha256。相对误差分母为 max(abs(reference),1)，沿用归档 validator 的最大绝对误差或最大相对误差通过规则。
- `source={path,sha256,line}` 指向原始 JSONL。被指向的行必须与该记录去掉 source 后完全一致。

这些元数据与哈希只能防止意外混淆、遗漏和事后更换，不能单凭声明证明采集器实现或此前无人看过数据。CPU 参考、容差与 merged 实现须在新测试标签产生前审查；若需要调整协议，另存新版本，不能改旧计划后仍称其预先冻结。

## 本地检查

新增测试覆盖三分量隔离、整体时间缩放不变性、同 tile 不同 M 的拒绝、校准不扩域、不同 host 拒绝、模型/数据非法状态、完整独立候选与原始来源检查，以及冻结选择不会重新拟合。合成时间仅在临时单元测试中使用，不作为实验数据。

本轮全套 191 项测试通过；CLI 检查确认旧 CUTLASS profile 在显式 opt-in 下回退到 cuBLAS，并写出身份不一致的具体原因。原 599 条记录与上一轮实验产物均保留。
