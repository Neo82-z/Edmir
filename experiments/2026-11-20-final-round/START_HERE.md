# 最后一轮 8×A100：直读与物化的选择边界（执行手册）

方案依据：`llm-serving-mlsys-latex/latex/review/FINAL_12H_PLAN_ZH_2026-11-20.txt`。两份文件冲突时，以本手册为准。

本包不写时间表。按 `design/queue.csv` 从上到下执行，队列顺序就是优先级。租期不够时从队尾放弃，放弃的作业记为 `not_run`。唯一和时间有关的是 `scripts/lease_guard.py` 的两个余量参数：`--no-start-min` 和 `--no-compute-min`，含义见脚本说明。

## 1. 问题与可以写的结论

主问题：给定每源行数 M、投影宽度 N（replay = ceil(N/tileN)）、目的卡负载倾斜和本机微基准参数，能不能在看到对决数据之前，判断该走远端直读还是先物化到本地？

两条比较边界分开报告：

- **B-a 物化代价**：同一个 J1 GEMM 消费者，只换输入来源，即远端直读与先 A2A 到本地。对应比较 `Ba_q1`、`Ba_q4`。如果 BNEW 没通过准入，用弱传输的桥接比较 `Ba_bridge_g1`。
- **B-b 实际选择**：固定 direct 对 A2A + 调优 cuBLASLt（`Bb_q1`、`Bb_q4`）。真实快照上的对手是原生 vLLM（`R_Bb_*`）。J8 对 SLT 的比较 `Bb_j8` 只作为消融另报，不进入 R2 的选择集合。

事前固定的结论规则：

- 只有相邻 M 档出现可重复的反向胜负，才报告交叉区间。
- 33 格全部由 staged 获胜时，结论只能是"这台机器、这些实现、已测范围内固定 staged 更好"。
- 预测器必须同时比"永远 staged"和"固定 direct"选错的代价更小，才写"有选择价值"。
- 论文不提前宣称模型能预测交叉点。

## 2. 证据口径

- **33 格是均衡代理**：3 个模型投影宽度 × M = 1…1024，每源到每目的的行数相同，FP16，合成输入。它们不是模型实测。
- **6 个真实快照是三模型证据**：GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V2-Lite-Chat 各取 prefill 和 decode，用真实激活、权重和路由，BF16。
- **历史格**：`QWEN36_M0032` 和 `QWEN36_M0064` 的几何以前测过，M64 当过校准点。`cells.csv` 的 `cohort` 列标了出来，R2 另报一份扣除这两格的结果。
- **J8 的范围**：J8 只有 q1，测不了 J1/J8 × q1/q4 的交互，只报 q1/g2 下 J1 与 J8 的差。
- **FLUX 的命名**：FLUX 写作"COMET 第一投影（FLUX V2 AGScatter）"，不称完整 COMET。

## 3. 开机前由用户交付的部分

本包只有设计、预测、分析和限时工具。下表的执行入口由用户在开机前完成，并在无 GPU 环境通过各自的 CPU 测试。

| arm | 内容 | 状态 |
|---|---|---|
| D_Q1_G2 / D_Q4_G2 / D_Q1_G1 | 逐源 direct CUTLASS SM80（J1） | 跑过 GPU；须重新构建并登记哈希 |
| BNEW_Q1_G2 / BNEW_Q4_G2 | ncclAlltoAll 到本地，再接与 D_* 完全相同的消费者（A 指针表换成本地段） | 新 |
| SLT | 单次 ncclAlltoAll + cuBLASLt：最多 8 个官方 heuristic，32 MiB 上限，剔除非法项，每格独立调优，冻结中位时延最低的配置 | 新 |
| HM_CUTLASS / HM_LT | host merged CUTLASS / cuBLASLt，作为历史基线 | 跑过 GPU |
| BOLD_Q1_G1 | pipeline：逐对 NCCL → staging → CUTLASS 64×128×32 | 跑过 GPU；与 direct 同 tile 配置但不是同一个 kernel，传输弱 |
| J8_Q1_G2 | 源并行 grouped GEMM | 仅 CPU 编译 |
| FLUX_V2 | FLUX `GemmGroupedV2AGScatterOp`，固定 commit 19831ca2；映射见 `experiments/2026-11-19-g1-preparation/contract.py` | 尝试构建 |
| NATIVE | vLLM 0.20.1：all-gather + Triton fused_moe 的 gate/up | 跑过 GPU |
| *_R | 真实快照的 ragged 版本：BF16、每卡多个专家、各专家权重不同、零行、不等长 | 新 |

### 3.1 执行入口的接口契约

所有入口都写 JSON Lines，每行一个对象，不覆盖旧文件，只追加。

**计时块**：给定 `--process-id`，按 `design/blocks.csv` 中该进程的 `arm_position` 顺序跑完全部 arm，每个 arm 写一行到 `metrics.jsonl`：

```
{"process_id": "B0001", "arm_id": "SLT", "unit_id": "GLM_M1024", "rep": "20261120", "orientation": "forward",
 "status": "ok", "median_wall_ms": 0.81, "harness": "<harness id>", "build_sha256": "...", "input_sha256": "...",
 "warmup": 20, "timed": 100, "samples_file": "raw/B0001/SLT.json"}
```

计时规则：

- 均衡块每个 arm 先预热 20 次，再计时 100 次；真实快照块同样。
- 每次迭代前八卡统一屏障，计"host 提交 → 本 rank 所有 stream 完成"，取八个 rank 的最大值，再取块内中位数。
- 原始样本全部保存。正式计时一律关闭 profiler。
- `status` 只能用：`ok`、`numerical_failure`、`fallback_detected`、`timeout`、`oom`、`runtime_error`、`unsupported`、`not_run`、`infrastructure_failure`、`lease_cutoff`。
- 一个 arm 没通过准入，就在块内写 `unsupported`，块里其余 arm 照常跑。

**harness 一致**：一个进程里的所有 arm 在同一个 harness 里跑，`harness` 字段如实填写。分析脚本拒绝跨 harness 的比值。FLUX 如果只能在每卡一进程的 harness 里跑，那么 `flux` 块里的 SLT 也必须在那个 harness 里实现。真实快照块同理：NATIVE 与 ours 必须同 harness。

**准入**：写 `admission.jsonl`：

```
{"unit_id": "GLM_M0001", "arm_id": "D_Q1_G2", "input_id": "20261120", "status": "ok",
 "scaled_max_err": 0.003, "nonfinite": 0, "fingerprint_ok": true}
```

- 均衡格的 `input_id` 是种子；真实快照写 `"real"`。
- 门槛：逐元素 max|y−ref|/max(1,|ref|) ≤ 0.02，参考为独立 FP32；没有非有限值；源指纹正确（检查源错排、尾块和残留旧值）。

**微基准**：写 `microbench.jsonl`，字段见 `scripts/calibrate.py` 开头。

- 每种 harness 各测一套，各自生成一份 `params_<harness>.json`。
- F、G、B_in 分别测量，不能用一个空 kernel 把三者混在一起：
  - F：保留原同步结构的空工作；
  - G：逐次 launch 在关键路径上的间隙；
  - B_in：1–7 个 peer 并读时的有效入口吞吐，需覆盖多种 CTA 数。

**捕获**：每个模型写一个目录，内容如下：
- 每个 rank 一份 `<block>-rank<r>.jsonl.gz`，格式与 2026-11-17 采集器相同：`kind=step` 记录带 `category`，`kind=route` 记录带 `layer` 与 `topk_ids`；
- 所选两个 step 在捕获层的输入激活、路由 ID/权重；
- 该层全部专家的权重；
- 以上全部文件的哈希。

捕获设置与 2026-11-17 相同：TP1/DP8/EP8，关闭 prefix cache、DBO、EPLB 与 speculative。请求集为 S32：32 条 512 token 的 prompt，各模型用自己的 tokenizer 精确截到 512，输出 128 token，ignore_eos，temperature 0。

**原生归因**：写 `native_attr.jsonl`，字段见 `scripts/amdahl.py` 开头。

- 每个模型只加载一次引擎，用运行时开关交替跑插桩和不插桩，各 3 遍。
- 事件放在实际的 stream 上；`replaceable_ms` 是 dispatch + gate/up 在关键路径上不重叠的时长。
- 不用 nsys：2026-11-17 在 nsys 下出现过 Xid 69。

## 4. 执行顺序

先在机器上跑一遍 `python3 -I tests/test_package.py`。它只用 CPU，在 Linux 上也会测 lease_guard。全部通过后，按 `design/queue.csv` 执行：

1. **`hw_gate`**：运行 `python3 -I scripts/hw_gate.py --out results/gate.json --load-test`（有 nccl-tests 时再加 `--nccl-alltoall`）。
   - `PASS`：继续。
   - `FAIL`：停止全部正式实验，保存回执，报告用户；停机或换机由用户在提供方操作。
   - `UNCHECKED`：先解决，再继续。
   - 脚本只读，绝不修改功率上限或时钟。瞬时功耗低不算失败，有效功率上限被调低才算。
2. **`build_identity`**：登记全部源码、二进制、输入和权重的哈希。
3. **`admission_smoke`**：每个均衡 arm 在 `GLM_M0001` 和 `GLM_M1024` 上过数值契约。
   - 失败的 arm 只允许一次针对性修复。
   - 再失败就记为未准入，后续块里写 `unsupported`。
   - BNEW 未准入时，B-a 改用 `Ba_bridge_g1`；SLT 未准入时，"永远 staged" 按 `policies.json` 的优先级顺延。这两条都按准入状态决定，不按快慢决定。
4. **`freeze`**：运行 `python3 -I scripts/manifest.py create`，对源码与二进制建清单。此后不再修改实现。确有必要的修改要另起实验 ID，其数据不与本轮混用。
5. **`admission_full`**：全部均衡 arm × 33 格 × 3 个种子，写入 `admission.jsonl`。
6. **`microbench` → `calibrate_predict_seal`**：
   - 跑 `scripts/calibrate.py` 和 `scripts/predict.py --design design`，输出到 `results/pred_cells/`；
   - 生成的 `PREDICTION_SEAL.json` 连同目录一起复制到机外并验证；
   - 复制完成之前，不启动任何计时块。
7. **均衡 core 块，第 1–3 轮**：每个进程用 `lease_guard.py` 包裹运行。每完成一对正反序就同步到机外，并用 `manifest.py verify` 核对。
8. **`capture`**：三个模型各一次。
9. **`route_stats_predict_seal`**：
   - 对每个模型运行 `route_stats.py`，参数为 `--snapshot <ID>:<category>:<ordinal>:<layer>`，取值见 `design/snapshots.csv`；
   - 用 `--merge` 合并三个模型的结果；
   - 运行 `predict.py --routes`，输出到 `results/pred_snapshots/`，封存后复制到机外。
10. **`admission_snapshot`**，然后跑**真实快照 core 块第 1–3 轮**。真实激活、权重与 ragged 路径三者齐全才算真实回放；只有路由时，交付负载分布，性能格留空。
11. **`native_attribution`**：三个模型各一次。
12. **FLUX 块、J8 块**：只在对应 arm 通过准入时运行。
13. **`udec_smoke`**：只在开机前已写好整模 decode 接入时运行。必须同时满足：真实输入、跨进程数据可见、全部 routed 层替换、无回退、输出核对。通过后才运行 `udec_ab`。租期内不从零开发。
14. **`final_analysis`**：

```
python3 -I scripts/analyze.py --design design --metrics results/metrics.jsonl --admission results/admission.jsonl \
    --pred-cells results/pred_cells --pred-snapshots results/pred_snapshots --out results/analysis
python3 -I scripts/amdahl.py --attr results/native_attr.jsonl --units results/analysis/unit_effects.csv \
    --out results/analysis/amdahl.csv
python3 -I scripts/manifest.py create --root results --out results/MANIFEST.json
```

最后复制到机外，运行 `manifest.py verify`。

## 5. 不允许的做法

- 看到结果后增删或更换 arm、格子、快照、比较或策略。
- 按快慢删除候选。
- 把失败、回退或缺失填成慢或持平。
- 用半对数据凑配对。
- 计算跨 harness 的比值。
- 用均衡 FP16 的速度代替真实 BF16。
- 封存预测之前打开对应的计时结果。
- 冻结之后修改实现却沿用原实验 ID。
- 正式计时开 profiler。
- 修改功率上限、时钟或驱动设置。

## 6. 判读规则（分析脚本已实现）

- **单格**：三对正反序的配对比值 T_baseline/T_candidate。
  - 三对都 > 1.03：candidate 明确胜；
  - 三对都 < 1/1.03：baseline 明确胜；
  - 三对都在 [1/1.03, 1.03] 内：工程持平；
  - 其余：方向未定；
  - 不足三对：incomplete。
- **交叉区间**：同一宽度、同一比较上，相邻 M 档出现明确的反向胜负（strict）。中间只隔持平或未定档的，单列为 loose，不作为主结论。
- **R1**：预测的交叉档与实测同档或相邻算命中。实测无交叉时，预测也无交叉才算命中。另报逐格胜负符号一致率。
- **R2**：P0 选择、永远 staged、固定 direct q4/g2 三种策略，分别报告相对逐格 oracle 的 regret。
  - R2a：P0 平均 ≤ 2.31% 且最大 ≤ 5.38%；
  - R2b：P0 平均胜过两种固定策略中较好的一种；
  - 33 格（含与扣除历史两格）和 6 个真实点分开报告。
- **整模**：只有 `udec_ab` 能写实测收益。`amdahl.py` 由 f 推出的上限单独标注，不与实测 TPOT 混写。
- **工程失败**：数值不过、隐式回退、没覆盖全部层、OOM 都记为未裁决，与性能落败分开。

## 7. 三种结局

- **A**：预测命中，且 direct 在小 M 或 decode 一侧胜出。写成完整的选择边界；只有 R2b 通过，才写"有选择价值"。
- **B**：direct 全输。结论只能是"固定 staged 更好"，用 B-a 的同消费者对照和微基准解释 replay、入口带宽、固定开销各占多少。
- **C**：预测不准。用微基准拆出误差来自哪个参数，如实报告；33 格本身仍是第一份同消费者的 direct/staged 对照。

## 8. 文件

| 路径 | 用途 |
|---|---|
| `design/` | 由 `scripts/make_design.py` 生成，不手改；重新生成的结果须与包内文件逐字节相同（测试会检查） |
| `params/scenario.json` | 开机前用的情景参数，不是测量值。开机前用它出一版预测并封存 |
| `scripts/predict.py`、`calibrate.py` | 两通道 max 模型及其标定 |
| `scripts/route_stats.py` | 真实路由的目的卡负载、倾斜和去重；已用 2026-11-17 GLM 数据复现 1.453 与 2.957 |
| `scripts/analyze.py`、`amdahl.py` | 判读与整模上限 |
| `scripts/hw_gate.py`、`manifest.py`、`lease_guard.py` | 硬件门槛、哈希清单、租期限时。`lease_guard.py` 改自 Codex 包 |
| `tests/test_package.py` | 离线测试 |
