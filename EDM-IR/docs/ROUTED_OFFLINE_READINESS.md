# Routed benchmark：离线准备与上机验收

本页是 D7 的执行准备，不是新实验协议，也不扩展论文 claim。
真实 trace 不可得时，先把语义、记录、统计和导出链路做通；不借合成
计时制造性能结果，不因为工具链通过而宣布 GPU 实验已经就绪。

## 1. 一条命令演练

在 `EDM-IR` 下运行，输出路径必须不存在：

```bash
python3 tools/prepare_routed_benchmark.py \
  --out-dir build/routed-rehearsal \
  --jar /absolute/path/tla2tools.jar
```

无需 CUDA、模型权重或网络。TLC 使用本地官方 JAR 和 Java；不传 `--jar`
时只审计已有模型/config 哈希，不冒充重新运行。不同演练用不同输出目录。

输出：

| 文件 | 用途 | 可作为性能结果？ |
|---|---|---|
| `synthetic-traces.json` | P2/P4/P8 的 balanced/hotset/empty 小输入 | 否 |
| `synthetic-ledgers.json` | 验证字节、CTA 和路由账本连接 | 否 |
| `cpu-numerical-checks.json` | 216 个 seed/split/group/layout 组合 | 否，仅 CPU 语义检查 |
| `synthetic-timings.json` | 18 个明确标记的人工计时 envelope | 否 |
| `synthetic-statistics.json` | 验证配对区间和 near-tie 分支 | 否 |
| `formal/*/{summary.json,*.txt}` | 本次 TLC 有界安全检查及变异反例 | 仅支持对应形式化 claim |
| `unit-tests.log` | 编译器、契约、参考和统计单测 | 否 |
| `campaign.template.json` | 待补资产的执行清单，默认未封存 | 否 |
| `source-manifest.json` / `artifact-manifest.json` | 源码和输出 SHA-256 | 来源审计 |
| `summary.json` | 明确区分离线通过与上机就绪 | 否 |

`q=7`、`K=7`、`N=9` 和 seed 0/1/2 是故意覆盖非整齐边界的测试参数，
不是给正式 schedule space 或三颗正式种子偷偷增加约定。`balanced` toy
允许 token 数的整数尾差；它不是实际 trace 的 balanced control 生成器。

## 2. 这一轮实际验证什么

### 路由与数值

`edmir/benchmark/routed_reference.py` 在 BF16 舍入后的输入上执行软件 FP32
累加。一个 global token 只有一个 source，多条 top-k 路由共享 activation，
不同 expert 有独立权重。输出必须覆盖每条 route 一次，恢复 canonical expert
顺序及双向 route index；空 expert 保留空 span。gate weight 只作为 metadata，
不在这一边界偷偷乘入 combine。重复、缺失、NaN/Inf 和 metadata 损坏被拒绝。

split-K 模拟改变累加顺序，group-N 模拟访问分组；它们不是 GPU kernel 的
执行模拟器。不能用 CPU 通过替代真实 BF16 kernel 的三种子误差检查。
当前可运行的历史 SM80 FP16 backend 也不能因此被称为 routed BF16 backend。

离线误差定义是 `||y-y_ref||2 / max(||y_ref||2, 1e-30)` 和
`max|y-y_ref| / max(max|y_ref|, 1e-30)`，阈值沿用 D7 的 1e-2/2e-2。
分母、零参考处理、输出 dtype 和正式种子必须在真实 campaign 中明确记录。

### 并发

保留两个小模型，不把整个推理系统塞进 TLA+：

- SlotLifecycle：先发布再读取、使用结束才能复用、split partial 消费完成再释放。
- RemotePublication：publish/acquire/consume/retire/overwrite 的 epoch 顺序。

每个模型有两个安全配置；合计七个故意破坏的配置必须出现指定 invariant
反例。运行器现在匹配完整 invariant 名与反例轨迹，并给原始日志加 SHA-256，
避免“命令失败”被误当作“成功找到预期反例”。

形式化模型/config 不变，编译器桥接测试继续执行。mutable remote publication
仍不能因为抽象模型安全就被放行执行；真实同步事件尚未进入 Program 的路径
继续拒绝。NCCL/CUTLASS 内部、GPU weak memory、warp 调度、浮点误差、性能均
不在这两个有限状态模型的保证范围。

### 计时与统计

`edmir/benchmark/measurement.py` 校验正常点的 10 warmups、10 blocks、
20 samples/block，保留每个 sample 的全部 rank 时间，先取 max rank 再统计。
不允许删单个 outlier，invalid block 必须在原始事件记录中保留并整块重跑。
真实记录需要实际 randomized candidate order；不同 pairing ID、trace、环境、
代码包或协议不能伪装成配对样本。

bootstrap 按配对 block 重采样；估计量是所有样本 pooled median 的比值，
不是 block 比值的中位数。near-tie 触发只返回 `needs_repeat`，**不自动给出
deployment decision**。本轮处理 normal envelope；2000 次 repeat 的采集格式
还必须随真实 harness 固定，不能把 repeat 数据塞成 normal point。

2000 bootstrap draws、95% CI、seed 1729、nearest-rank p95 是工程演练默认值；
正式 campaign 仍留空待确认。D7 的 2000 次 GPU repeat 和 bootstrap draws
不是一回事。请求哈希拆分维持父请求隔离，但 60/20/20 哈希比例不自动满足
各 phase 的固定抽样配额，还需要核对真实父请求清单。

## 3. 数据回来前必须完成的清单

以下不增加 D7 的 shape 数量或 GPU 预算；只是把执行所需的具体值绑定到
已接受的协议。未填完不能改 `status` 为 `sealed`。

- [ ] 模型 config/revision、trace 提取器版本、数据来源、输入请求和 route 文件哈希。
- [ ] 每个请求的 prefill/decode、layer/step、expert placement 与 top-k metadata。
- [ ] request-level split salt、父请求分区；所有衍生 control 与 P replay 同分区。
- [ ] 正式三种子、误差分母和输出 dtype；参考实现不能与被测 bug 共用关键索引逻辑。
- [ ] 每个 baseline/family 的实际实现、构建哈希、支持的 BF16/shape/layout，不能静默换方案。
- [ ] 正常点与 near-tie repeat 的原始 envelope、实际执行顺序及配对标识。
- [ ] bootstrap/quantile/aggregation 的固定定义；没有测量就保持 `null` / `--`。
- [ ] real/balanced/hotset 的构造约束和 D7 fragmentation sentinel 守恒检查。
- [ ] frozen candidate 与 policy 的清单、校准/调参隔离、tuning cost/cache 记录字段。
- [ ] 一键打包、恢复、校验哈希和导出路径预演成功。

## 4. 真正上机的准入与停止条件

只在 CPU 准备和资产齐备后开付费节点，沿用 D7 pilot 和总预算上限：

1. **环境留档**：同一个 A100 P8 节点，P2/P4/P8 固定实际 UUID 子集；
   driver/CUDA/NCCL/CUTLASS、topo/P2P、功率/频率、进程占用与构建日志留档。
2. **先 correctness**：真实 BF16 路径跑三种子、empty/small/tail/hotset 和
   canonical conversion；任何错误都不能进入计时/selector 数据集。
3. **再计时 pilot**：确认没有遗漏 route-dependent preprocessing、metadata、
   sync 或 canonical conversion；reference/setup 不混进被测区间。
4. **估预算后启动**：pilot 估算正常点、repeat 和数据回收时间；保持预留，
   不因出现漂亮结果追加 evaluation shape 或重训 frozen policy。
5. **停止/整块重跑**：CUDA/NCCL/Xid/ECC 错误、foreign process、协议列出的
   throttle/clock 异常等必须带原因记录。性能不符合期待不是删点理由。
6. **导出前再验收**：完整候选覆盖、缺失原因、数值结果、统计配对、封存 split、
   原始日志和哈希齐全。counter 不可访问写 unavailable，不能填 0 或称仿真值。

这套离线参考和统计接口仍需接入真实 routed GPU harness。不能把现有均匀
proxy 的正常运行当作这一接入已经完成；CUDA/强 baseline 适配是独立验收项。

## 5. 论文导出与复现

正常点的真实记录通过以下命令导出 JSON observations；未封存、缺失数据、
合成来源、数值未通过、trace/environment/build 不匹配都会拒绝：

```bash
python3 tools/export_routed_measurements.py \
  --input real-timings.json --campaign campaign.json \
  --out-dir build/routed-observations
```

这是数据完整性/绑定检查，不是来源真实性的密码学证明，也不会凭一个 boolean
替代人工审核原始数值报告。导出只包含 normal-point observations，不生成
oracle、可部署 selector、serving tokens/s 或自动挑选胜出论文结论。

论文仓库有 `latex/templates/routed_results.tex`：18 个 phase/control/P 格子及
policy 指标保持 `--`，暂不 include 到 main。沿用现有的 measured/concept 图，
不改历史 FP16 数据。图的缺口由真实 routed evidence 补，不靠更多 toy 点补。

独立 QA PDF：

```bash
python3 latex/scripts/render_routed_rehearsal.py \
  --input-dir /absolute/path/EDM-IR/build/routed-rehearsal \
  --output latex/output/pdf/routed-offline-preview.pdf
```

渲染器需要 ReportLab；只收合成演练，拒绝写入 `figures/`，每页标明离线工程
用途。它仅用于检查统计到制图的接口，不进入论文主结果。
