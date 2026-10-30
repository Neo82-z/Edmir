# DeepInfer：基线、计时边界与可验证的优化方向

2026-10-20。结论来自源码阅读、纯 Python 计数函数回放和本仓库静态计数。**没有运行 GPU 实验，没有获得新的加速比，也没有据此改动预测器参数。**

最值得迁移的是明确区分完整执行成本与诊断用时，并加强 host 基线。DeepInfer 的算子和生产服务经验值得研究，但当前公开 benchmark 的硬件、精度、路由和计数条件均不同，不能直接作为我们 A100 上 uniform A2A–GEMM 的训练标签。

## 版本和复查入口

| 项目 | 本地目录 | 本次固定版本 |
|---|---|---|
| DeepInfer | `/Users/neo/code/DeepInfer` | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| deepseek-recipe | `/Users/neo/code/deepseek-recipe` | `935817c66f5d6e64838f419692545f06787e485f` |

这是本次拉取的版本，不是之前讨论引用的 `8b161497…`。未安装依赖、初始化子模块、下载模型或启动服务。DeepInfer 当前声明支持 SM100 / Ascend 950，不能把它的完整引擎当作现成 A100 基线。recipe 的示例服务使用 mock inference，主要负责协议和输入输出处理，本轮不纳入 GPU 算子基线。来源：[DeepInfer 支持范围](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md)、[recipe 示例服务](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/server-py/README.md)。

[audit.py](audit.py) 从固定版本 AST 中提取少量纯函数，不 import 推理框架；[audit_results.json](audit_results.json) 记录源码哈希、输入构造、检查结果，以及旧 GPU 小批包中的源码一致性检查。标准库即可运行：

```bash
python3 experiments/2026-10-20-deepinfer-measurement-audit/audit.py \
  --deepinfer /Users/neo/code/DeepInfer \
  --recipe /Users/neo/code/deepseek-recipe
```

## 1. 哪些测量方法可以借鉴

DeepInfer 把 warmup、日志、正式计时和 profiler 窗口分开；prefill 使用实际处理的 token 数除以累计时间，没有假设每步恰好装满 batch。这些做法值得保留。

但“计时窗口没有活动 profiler”不等于“进程未受采集影响”：

- `benchmark.py` 的 `setup_llm()` 无条件传入 `profile_init=True`；`_config.py` 会在模型构造时进入一次 `torch.profiler.profile()`。因此 `PROFILE=0` 只关闭后面的采集窗口，仍会做初始化采集。
- `PROFILE=1` 时，prefill 采集发生在 decode warmup 和测量之前；decode 自己的采集才是在 decode 测量之后。不能把整个进程概括成“所有计时均早于 profiler”。
- 关闭周期日志不等于关闭内部计时。CPU 区间仍可执行 `perf_counter` 和 `record_function`；部分层的 CUDA event 计时仍被启用。

这并不证明初始化采集一定污染了最后结果，影响大小需要同配置的新进程对照。我们的主成绩应由完全不启动 profiler 的进程产生；trace 用另一次运行解释执行组成，不直接提供最终时延。

源码：[初始化设置](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L151)、[初始化 profiler](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/_config.py#L86)、[测量与采集顺序](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L457)、[CPU 计时](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/timer.py#L48)。

## 2. “干净窗口”不是完整运行吞吐

`measure_prefill()` 和 `measure_decode()` 都剔除发生 LB 统计 flush 的区间。benchmark 将统计周期设为 10 步；flush 路径会分配临时张量、执行 EP all-gather、归约并把统计结果拷回 CPU。`decode_max_step_ms` 也只来自剩余区间，不是完整运行的最大延迟，更不是 p99。

这是有明确条件的诊断指标。若部署保留这项统计，它就是实际运行成本；若决定关闭统计，应执行“关闭统计”的实现重新测量，而不能仅从旧数据中删除发生统计的时间段。异步工作可能跨区间，标记一个区间也不能证明已消除所有后续影响。

本地直接调用上游计数函数得到以下**人工例子**：

| 人工输入：60 步，通常 1 ms，每第 10 步额外 5 ms | 包含所有区间 | 上游剔除规则 |
|---|---:|---:|
| 纳入步数 | 60 | 54 |
| 平均 steps/s | 666.67 | 1,000 |
| 最大单步 | 6 ms | 1 ms |
| 满接受计数吞吐，K=64、nextn=5、CP=1 | 256,000 tok/s | 384,000 tok/s |

这里删除了 10% 的区间，但删除了 40% 的时间；仅改变统计口径，吞吐就相差 1.5 倍。**1.5 倍是人工输入的代数结果，不是 DeepInfer 实测偏差，也不是优化收益。** 无 flush 标记和全部带标记两个控制输入也已检查。

建议每个测量记录保留原始区间、区间工作量、事件标记和剔除理由。主表使用完整算子边界内的原始耗时；诊断表另给条件窗口结果、排除区间数和排除时间比例。不能因为某个样本慢就删除它。

源码：[过滤与计算](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L383)、[统计 flush](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/eplb/__init__.py#L167)。

## 3. 基线身份必须包含输入与执行语义

| 源码事实 | 对比较的含义 |
|---|---|
| benchmark 设置 `DEEPINFER_FORCE_LOAD_BALANCE=1`；该值进入 `gate_force_random`，随后作为 gate 的 `force_random` 参数 | 这是改变路由的实验设置，不能当作自然请求下的路由分布。也不能仅凭开关名称声称每个专家严格等量；本次核查到 gate 调用边界，没有执行外部 gate kernel |
| 先计算共享前缀，再 fork 成多个请求，分别处理 suffix；缓存长度为 0 时直接建立请求 | 默认测量包含热前缀条件，不代表互不共享前缀的冷请求 |
| decode 吞吐使用 `SPS × K × (1+nextn) / CP` | 这是满接受计数。有效输出吞吐需要实际接受数和完成时间；绘图时输入接受率只是条件换算 |
| 结果只由 `dp_rank=0` 写出 | 是该 rank 的窗口统计及折算，不能无条件视为逐 rank 实际完成量汇总后的全局服务吞吐 |

例如 nextn=5、假设总体 draft 接受比例为 0.75 时，每步计数由 6 变成 4.75；代码可完成此换算，但并未通过这个参数测得接受率。不能把任意逐位置条件概率直接代入后称为有效吞吐。

当前 JSON 记录了很多配置，但没有完整记录强制路由、统计周期和初始化 profiler 等条件。我们的 baseline identity 应同时固定数学语义、输入和路由、精度、库和 kernel、调度方式、硬件拓扑、计时边界及观测开关。

源码：[强制设置](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L164)、[metadata](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/input_metadata/prepare_tensor_input_base.cpp#L82)、[gate 参数](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/v4_modules/mlp.py#L359)、[工作负载说明](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/README.md#L34)、[结果字段](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/benchmark.py#L521)。

## 4. 重跑和配置检查还有改进空间

`plot_pareto.py::load_points()` 对同一 batch 只保留最后一次记录，再检查部分固定字段；这是重跑替换规则，不是重复实验的统计汇总。输入两条 100 / 90 SPS 的同 batch 人工记录，输出只剩 90；换记录顺序也可能让曲线变高，因此不能据此指控选择性报喜。

本地回放还确认：

- 不同 batch 的 `prompt_mode=text/image` 可以通过当前一致性检查。
- 同 batch 的旧 `nextn=3` 记录会先被新 `nextn=5` 覆盖，旧配置不会进入后续检查。
- 不同 batch 的 `ep_size` 不一致会被拒绝，说明已有部分检查确实有效。

我们的随机候选顺序、多独立进程、保留原始样本方案应继续使用。重跑应保留 run_id，并明确作废理由；以独立进程为重复单位汇总，不能把一个进程内连续迭代当成大量独立实验。当前小批每候选仅 3 个进程，适合检查方向和稳定性，不足以声称可靠的服务 p99。

源码：[绘图输入及检查](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/plot_pareto.py#L23)。

## 5. 优先验证的优化

### A. 我们的 runtime：先减少重复的主机工作

`CudaBackend::execute()` 中，各 transfer、GEMM 和 fused op 都分别调用 `accesses(op,false/true)`；每次从头扫描整张访问表并创建一个 vector。`op_devices()` 也在依赖和完成事件路径上反复构造列表。矩阵和 buffer 绑定在 `prepare()` 后已确定，这些工作可以提前完成。

对现有小批包第一组 P=4 的生成 Program，JSON 计数与生成 C++ 的 `AccessDesc` 数量一致：

| 当前路径 | 所有 op 数 | 访问表长度 | 每次 execute 的 accesses 调用 | 扫描记录总数 |
|---|---:|---:|---:|---:|
| per-source host | 36 | 116 | 64 | 7,424 |
| direct fused | 16 | 48 | 32 | 1,536 |

这是静态操作数，不是分配器测量、CPU 时间或性能瓶颈证明。两条路径都承受这些开销，但数目不同，可能影响小 M 下的完整算子比较。

**下一版改法：**在准备阶段保存每个 op 的输入、输出、设备集合、字节数及必要依赖，执行阶段使用已验证的记录；同时应用到 host 和 fused。先保持 CUDA/NCCL 调用及依赖顺序不变，避免把主机预处理改进和通信调度改进混在一起。可在 CPU 上检查准备结果与原 Program 一致；真正加速和 GPU 正确性仍须实测。

CUTLASS host launcher 每次还会构建参数并调用 `can_implement`。可以继续检查能否在准备阶段验证并缓存固定参数，但不能凭这一点断言 CUTLASS 初始化占主要时间。

来源：[cuda_backend.cpp](../../EDM-IR/runtime/src/cuda_backend.cpp)、[cutlass_gemm.cu](../../EDM-IR/runtime/src/cutlass_gemm.cu)、[测量入口](../../EDM-IR/runtime/tools/benchmark_selector.cpp)。

### B. 保留 merged host，区分执行策略和 kernel 优势

已有运行包包含 per-source CUTLASS、per-source cuBLAS、merged CUTLASS 和 merged cuBLASLt。merged host 每个目的 rank 做一个 PM 行 GEMM，减少调度和小 GEMM 数量；它应继续作为强基线。

合并只对当前“同一目的 rank 的所有 source 共用同一个 W”的 uniform 语义成立，不能直接推广成任意路由 MoE。merged 路径同时改变矩阵形状和派发次数，因此若它更快，首先说明完整策略更好；若我们的方法仍更快，再判断证据能否支持通信/计算机制的收益。

改主机执行器后，host 和 fused 都应测新版本，并给新实现 ID、源码哈希和预测器 profile。旧样本不能改名后混入；原 GPU 小批包和独立计划保持冻结。

### C. DeepInfer：内部计时的开关和事件维护

CUDA Graph timer 在事件缓冲切换时调用 `set_event()` 更新图中的 event 节点，读取时同步事件并计算耗时。这里值得做“仅最终完成计时”与“内部细分计时”两种运行模式的对照，检查小 batch 的 CPU 开销。

但 `_post_process_step()` 先等待 `sample_end_evt`，然后才读取 graph timer。所以不能从源码中看到 `synchronize()` 就断言它在重复等待 GPU，更不能删除保证输出就绪的完成事件。优化对象是可选计时节点及其主机管理，收益尚未测量。两种模式可能生成不同的 graph，应分别构造后测试。

来源：[事件切换及读取](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/timer.py#L169)、[输出就绪依赖](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py#L247)。

### D. DeepInfer：统计缓冲区复用

LB 统计 flush 每次新建 `gathered_tokens`，随后做类型转换和归约。固定 EP、统计周期和层数下，可检查是否能预分配复用，或者用显式完成依赖管理统计读取。首先减少内存分配；是否值得异步化应由完整窗口耗时和负载均衡统计正确性决定。

这与“删掉 flush 区间”是不同操作。若保持统计频率及值一致，而包含 flush 的总时间下降，才是实际优化。改变统计频率或禁用统计则要作为另一种部署配置报告。本轮没有实施这些 GPU 路径改动。

## 6. 下一轮实验的收敛范围

先继续使用现有四个形状，验证强 host 基线和完整算子计时，不因研究 DeepInfer 扩大到完整模型或 SM100 租赁。保留已有 FP16 输入、CPU FP32 参考、前后全输出校验、随机顺序和独立进程重复。

若实施主机元数据预处理，建立新的版本化运行包，比较以下两组：

1. 相同 per-source 数学与调度语义下，旧执行器和预处理后的执行器；host、fused 都改，检查收益归属。
2. 预处理后的 per-source host、现有 merged host 与 fused；比较完整执行策略，单独注明 merged 改变了 GEMM 的 M。

主指标仍由 `steady_clock` 围住 `execute()+synchronize()` 取得，使用原始样本，保留进程内中位数及均值，再在独立进程层汇总。平均耗时用于完整工作量/总时间换算；中位数回答典型迭代延迟，两者不可互换。旧冻结评分协议继续按原规则执行；新协议要先冻结再收集标签。

诊断采集使用另一个进程。设备侧事件只能解释部分临界路径，多 stream/多 GPU 上不能把各段 event 时间简单相加，也不从 wall time 中直接减去估计的 profiler 或主机开销。若收益与进程间波动同量级，增加重复，暂不将其作为已证实优化。

完成以上后再判断预测器问题究竟来自特征表达、基线身份，还是主机执行成本。公开 DeepInfer 结果可以帮助提出机制假设，但不能代替同硬件、同语义的独立验证。
