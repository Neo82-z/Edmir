# 候选执行账本与选择证据

2026-10-26 新增 [tile 表示与开发划分](../../experiments/2026-10-26-tile-model-revision/README.md)：
在本账本之上保留执行几何与尾部，并提供条件节点耗时的 DAG 汇总、整形状候选分区检查。
该分支未训练或计时，不改变现有模型准入；服务证据缺项保留为未知。

2026-10-24 更新：执行台账和选择报告均升级为 v2。旧 JSON 和实测记录保留，不自动换成新身份。

这版把“能从源码算出的量”“模型的条件估计”“实际测量”和“尚未知的成本”分开。
执行合同修复本身没有训练新模型、没有新增 GPU 结果。M512 和大 N 的设备耗时成因尚未定量解决。

后续小范围分析已新增独立的[数据清理与机制预测入口](../../experiments/2026-10-24-mechanism-prediction/README.md)：
保留所有主测和诊断原始样本，按完整形状留出，标定固定形式的非负成本项并输出微秒预测、
成本分解、误判清单和条件队列暴露。这是对旧冻结执行版本的数据估参，未获编译器准入，
也不替代下述当前 v2 执行台账。不能把旧测量挂到修复后的源码或精简依赖策略上。

10-25 的[细粒度表达修订](../../experiments/2026-10-25-fine-grained-prediction/README.md)进一步核验了
grouped-N 的动态共享内存 opt-in 调用、寄存器与 block 资源上界，并检验分片工作集、
SM 分布、串行边界和 M 行块耦合。源码计数、经验耗时表达与实际占用率仍分别记录；
完整形状留出改善，但整条 M64 轴留出有新的排序退步，未获得编译器准入。

随后新增[延迟下限与吞吐切换检验](../../experiments/2026-10-25-latency-throughput-regimes/README.md)：
以相同 8 个系数对照逐项相加和 `common + max(aL,bW)`，有限枚举训练比值区间并核验求解。
完整形状留出平均误差降至 4.18%，M192 严重误判与 M64 轴留出的 fused 大 N 误选改善；
仍有 M256 排序退步、host 误差和质量敏感性。切换点是统计模型预测，未识别物理瓶颈，
也没有新增 GPU 标签或变更自动选择。

## 1. 当前入口

在仓库根目录，仅需 Python 标准库：

```sh
PYTHONPATH=EDM-IR python3 -m edmir.cli explain-execution \
  --m 512 --k 4096 --n 512 --environment planned-a100-p4 \
  --out /tmp/m512-execution.json
```

也可运行 `python3 EDM-IR/tools/explain_execution.py`，参数相同。
输出路径必须不存在，已有文件不会被覆盖。`--families host splitk2 splitk4`
可缩小显式比较集合；单候选只出账本，不宣称找到最优方案。

当前支持 P=2..8、FP16 row-major、每源 SM80 CUTLASS 64×128×32 系列、
q=1/2/4/8/16、g=1/2。精确分片账本要求 K 是 32q 的倍数，N 是 8 的倍数。
这是实现范围，不是已测性能范围。`--environment` 是使用者声明的环境标识，
不会触发硬件检测，也不证明已满足 peer access 条件。

`--metadata-mode legacy|prepared` 属于候选身份。四种 host 使用既有采集器的名称：

| family | 实现 |
|---|---|
| `host` | NCCL 搬运，每目标 P 个 CUTLASS GEMM |
| `host_compiler_cublas` | 相同分区，每目标 P 个 cuBLAS GEMM |
| `host_merged_cutlass` | NCCL 搬运，每目标一个 PM 行 CUTLASS GEMM |
| `host_merged_cublaslt` | 相同合并分区，一个 cuBLASLt GEMM |

merged 固定采用自身的 `merged` metadata 模式。共享目标权重时，沿 M 轴拼接输入再计算
与逐来源计算在数学上等价；仍要求相同输入、全输出数值检查和容差。
库内部 CTA、tile、访存量保持未知，不套用 CUTLASS 公式。
cuBLASLt 在未提供 `--cublaslt-algorithm-id` 时可以生成静态台账，但不能接收 TimingEvidence。
该参数必须来自实际 benchmark 的完整每 rank 算法记录，不能用猜测值解除检查。

精简策略通过 `--dependency-policy uniform_immutable_peer_v1` 显式选择，仅作用于 fused。
默认仍为 `resource_queues_v1`。它只接受标准的未分块 FP16 uniform 程序，P=2..8；
每个目标共享权重，输入/权重只读，输出独立。动态 producer、缓冲区复用、chunked 和混合程序
均被拒绝，不会删除它们的 RAW/WAR/WAW 依赖。它不进入自动编译器选择。

生成可执行 C++ manifest 时使用：

```python
from edmir.compiler.execution_policy import with_dependency_policy, UNIFORM_PEER_POLICY
program = with_dependency_policy(program, UNIFORM_PEER_POLICY)
artifact = CppManifestBackend().emit(program, target)
```

策略会写入 `ProgramView`。runtime 再核对范围和目标计算队列边；必须先完整上传所有外部矩阵，
并在每轮后 synchronize 全部 rank 才能再次 execute、修改输入或读取输出。
新策略禁用可写原始指针入口；workspace 仍按操作独立分配，释放前等待设备完成。

两种 backend 的 `copy_from_host()` 都保证默认上传流完成后返回；split-K 工作区初始化
也在 prepare 返回前完成。等待在主计时之外，不在每轮稳态执行中插入 device-wide 同步。

## 2. 每份账本包含什么

| 字段 | 含义 | 不能据此声称 |
|---|---|---|
| `source_facts` | tile/warp/thread、K 区间、实际 Program、推导依赖、运行时顺序 | 已知关键路径的时间 |
| `counting_identities` | 有单位、作用范围、表达式及源码引用的计数 | 等于 HBM/NVLink 物理事务量 |
| `assumptions` | 源码/依赖版本、指针对齐、peer access、输入生命周期等前提 | 硬件已经验证这些前提 |
| `unknowns` | 实际流量、驻留资源、等待/重叠、CPU 服务时间、设备数值正确性 | 可用零值或拟合系数代填 |
| `latency_us` | 当前为 null | 计数公式已经预测了微秒数 |
| `execution_graph` | 实际后端执行描述、输入 manifest、依赖、分区和算法身份 | 已知节点耗时；merged 输入 manifest 本身不是其执行图 |
| `provenance` | 被引用源码的路径与 SHA-256、CUTLASS 语义所依赖的提交 | 当前加载二进制的身份或全构建依赖证明 |

M 是一个源—目标分块的行数；每个目标 rank 总共处理 PM 行。
账本保留独立 M/N tile 数、A/B 唯一工作集和分区工作集。
全 P-rank 计数与单 GEMM 计数分别标明，不能把全机字节除以单卡带宽。

计数公式绑定已审阅的资源定义、策略、lowering、codegen、ProgramView 类型、内核、运行时和计时源码快照。
身份另外绑定执行图摘要，比较时重算图和源码摘要；变更图不能继续使用旧 TimingEvidence。
这些文件缺失或哈希变化时停止生成账本，要求先复核公式再更新绑定；
不能仅记录一个新哈希就假定旧公式仍然适用。这仍不等于认证实际加载的 GPU 二进制。

设 tM=ceil(M/64)、tN=ceil(N/(128g))，单 GEMM 有 tM×tN×q 个 CTA。
在当前整除条件下，逻辑 A 读取为 2MK×tN，B 读取为 2KN×tM；q 个 K 分片
覆盖互不重叠的 K 区间，因此总读取不能再除以 q。缓存复用、事务放大不在这两个等式中。

q>1 采用 serial split-K：每个输出 tile 的 epilogue 按分片序号等待 semaphore，
后续分片读取上次 FP16 输出并累加，最后分片重置 semaphore。逻辑输出写入为
2MNq 字节，读取为 2MN(q−1)，较 q=1 多 4MN(q−1) 字节。
它也改变舍入过程；不能要求与 q=1 逐位相同，也不能免除全输出数值检验。

workspace 的分配和初始化在 `prepare()`，不计入每次稳态执行。
主计时仍包含 `execute()`、事件操作、CPU metadata 工作及每 rank 的 stream 同步：
普通 backend 为三条，merged 为两条。
只改成设备 kernel 计时会改变问题，必须建立新的测量身份。

这些成本通过依赖及共享资源影响最终完成时间。既不能把重叠的阶段时间直接相加，
也不能用理想计算/带宽时间的最大值冒充预测结果。

## 3. M512：历史模型归因记录，尚未分解 43.9 μs

第 3–4 节保留此前归因材料；10 月 21 日采集结果见对应实验目录，本次不改写旧数据。

P4/M512/K4096/N512，以下计数都是**单个源—目标 GEMM**：

| 项目 | q2 | q4 |
|---|---:|---:|
| CTA 数 | 64 | 128 |
| 每 CTA 的 K 主循环次数 | 64 | 32 |
| 每输出 tile 的有序 epilogue 数 | 2 | 4 |
| 相对 q1 的额外输出读写 | 1 MiB | 3 MiB |
| semaphore 槽数 | 32 | 32 |
| 历史完整程序中位数 | 1005.9 μs | 1049.8 μs |

前五行来自执行语义；末行是历史完整程序测量，**不能拿它直接减出单 GEMM 的阶段成本**。
已有错误审查确认 q4−q2 的表达在固定 M/N 下为 K 的仿射函数，而公共非线性项抵消；
留 K4096 时模型过早预测交叉。这是已定位的模型缺陷。

“增加的串行 epilogue、资源竞争或 CPU 开销导致多少差距”仍是假设。
沿用已冻结的 M512/K4096 对照与 K3072 诊断点，先测同身份 q2/q4 完整程序，
再单独采集 CPU 提交和 kernel 时间线。若差距主要出现在 CPU 提交而非设备执行，
把全部损失归因于 semaphore 的解释就不充分。只有时间线仍无法解释时，再查编译资源与停顿。
本轮不执行这些设备测量。

## 4. 大 N：总量、复用与基函数切换分别处理

P4/M64/K4096/N2048，历史 host=645.3 μs、q2=724.8 μs，差 79.5 μs。
更早模型预测 q2 快约 0.48 μs，说明小预测差不能作为真实接近平局的证据。

g1 的 tN=16，单 GEMM 的 A 唯一工作集为 0.5 MiB，逻辑 A 读取为 8 MiB。
g2 将 tN 降为 8，逻辑 A 读取降为 4 MiB，同时线程数从 128 增至 256。
真实缓存命中、共享内存/寄存器驻留和链路流量未知；这个形状没有 grouped-N 历史标签。
不补造标签，不把 M64 的其他点或 M256 的结果移过来。

模型层已发现两类问题：不同 M/N 只剩相同 grid 的信息冲突；host 基函数在轻微
数据干预后离散切换。这些结论不需要 GPU 才成立。但它们没有单独证明设备上的瓶颈。

沿用大 N 旧形状的 host/direct/q2 对照，以及现有执行包 K3072 下
M64/N1536 与 M192/N512 的等 tile 对照。先固定 legacy 身份复现；prepared
metadata 仅作为不同身份的控制。若流量近似而耗时仍不同，单一总流量解释不充分；
若 metadata 改变了差值，CPU 执行成本也必须纳入解释。

grouped-N 跨 M 的问题沿用四个优先形状中的 M64/M256，固定 q/K 再比较 g1/g2。
这是一项待完成的控制，不是新增扩大搜索，也不是宣称已可跨 M 外推。

## 5. 选择接口如何表达证据

`evidence_selection.assess_candidates()` 不训练，也不把公式转成耗时。
仅有账本或拟合分数时返回：

```json
{"status":"insufficient_evidence","choice":null,
 "retained_candidates":["host","splitk2","splitk4"],
 "eligible_for_compiler_admission":false}
```

这是部分字段的示意；实际输出还包含缺失证据、比较范围、理由和来源。
它没有把全部候选保留解释为“零误选”或“模型通过验收”。

程序接口可接收明确的 `TimingEvidence`：各自匹配本候选的源码、图、策略和算法身份；
候选间必须共享形状、环境、数学工作负载、dtype、上传完成约定和计时源码/范围。
metadata、分区、依赖策略和库允许不同，不能把不同候选改名为同一实现。
实测还须共享输入/数值检查等测量上下文及区组，每个候选至少三次独立进程中位数，有来源、正确性通过，
且不启用 profiler。一次进程内的多个迭代不等于多个独立进程。
输入声明和来源真实性须由采集器建立；此模块只验证一致性，不认证第三方数据。

仅当某候选的最大进程中位数加声明的比较余量，小于每个对手的最小中位数时，
返回 `separated_under_supplied_evidence`。这些范围是样本范围，**不是置信区间或未来耗时边界**。
host/fused 使用完全相同的规则。重叠、相等、缺少对手数据时保留待定候选；
字典排序仅规范展示，不用于打破性能平局。任何结果都不自动进入编译器。

已有 `predict_decision_schedule.py`、`predict_traffic_schedule.py` 现在采用这份报告：
旧模型偏好移到 `fitted_model_diagnostic.model_preference`，真正的决定在
`decision.choice`。旧模型内部研究 API 与冻结参数保留，便于复查历史错误；
它们不属于本次认可的选择依据。

## 6. 验收范围

- 独立枚举有效 tile/K 区间核对计数，保留同 grid 的 M/N 区别。
- 检查 grouped-N 的读取与资源变化、split-K 输出代价、准备/计时边界和真实 Program 依赖。
- 检查 host/fused 双向比较、候选重排、平局、缺失证据、身份/区组不匹配和不合法计时。
- 标准库环境运行命令行，输出文件禁止覆盖；输入 P/候选数/尺寸有上限。
- 41 个旧形状各生成 11 个候选账本；不训练、不计算虚假的新准确率。
- 旧模型、第二轮审查与已冻结 GPU 包保留；新源码和执行图使用新身份。

Ascend 记录用于提出“事务几何、资源生命周期、关键路径”的检查方法，不用于给
A100 提供数值参数。本交付将源码计数修正为可审阅材料，设备时间与独立验证仍待实测。

## 7. 2026-10-25 剩余误判与下一次采集

[本轮审查](../../experiments/2026-10-25-final-local-revision/README.md) 将九对误判分为七对三进程方向一致、两对方向不一致；全部保留在指标中。新增残余深度表达降低平均误差，却使错误对从 9 增至 12、全部候选最坏选择损失从 2.29% 增至 9.02%，因此没有替换原模型，也没有获得编译器准入。

[GPU 执行包](../../experiments/2026-10-25-regime-gpu-validation/README.md) 冻结两版模型及 57 个主配置、4 个预热诊断，等待新的四卡 A100 环境。它保留 10 月 21 日实测执行源码和策略，只更改采集编排；不能用于宣称后来修复的 runtime 已经验证。新机器的结果先保持独立 context，核对复测点、算法和质量，再讨论模型修改。
