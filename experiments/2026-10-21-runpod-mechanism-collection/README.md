# RunPod 首轮机制采集包

这轮先采集可解释的对照数据，回答 M512、large-N、grouped-N 和 host 基线的问题。没有训练模型，也没有把历史误选宣布为已修复。包内包含冻结源码、固定运行顺序、FP16 输入、FP32 CPU 参考输出、执行计数和依赖源码。

## 机器要求

| 项目 | 本轮要求 / 建议 |
|---|---|
| GPU | 单机 4× NVIDIA A100-SXM4-80GB，SM80 / 108 SM |
| 卡间连接 | 所选四卡两两 `NV12`，CUDA peer access 全通；连接后检查，不能只看平台商品名 |
| CPU / RAM | 建议至少 16 vCPU、64GB RAM；构建默认并行 4，避免编译挤占内存 |
| 磁盘 | 建议 60GB 可用空间，包含 CUDA 工具链、源码、构建产物和少量时间线；输入及参考仅 590 MiB |
| 系统 | Linux x86_64；Python 3.10+、CMake 3.20+、支持 C++20 的编译器、Git、make |
| CUDA | 优先旧实验的 CUDA 12.8 开发环境，必须有 `nvcc`；不要只装运行库。GCC 12 是可选搭配，真实兼容性以编译为准 |
| 库 | 包内 CUTLASS `dcf215af68a2d08d305076c152a06f201728cd53`；NCCL 2.28.9-1 `dbc86fd06e8b0c4517b95d8958a09ccacf9520c9`，单独构建并记录动态库哈希 |
| 诊断 | Nsight Systems CLI；主测量不需要 profiler。首轮不要求 Nsight Compute 权限 |
| 运行条件 | 独占所选四卡，避免其他计算任务；不并发执行采集脚本 |

80GB 和 NV12 是复查已有实验的可比性要求，不是这些矩阵的显存容量需求。单卡不能替代 P=4 的通信与 fused 对照。支持 P2P 的部分 PCIe 拓扑也能使用 NCCL LSA，但会改变这轮要复查的成本条件，本包不接受直接替换。NCCL 从 2.28 引入 Device API，其 LSA 的 P2P 条件见 [官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)。这里不使用跨节点 RDMA，也不要求 H100 / Blackwell。

RunPod 选择带开发工具链的模板，并提供 SSH over exposed TCP 连接信息，方便传输执行包及结果；连接方式见 [RunPod 官方说明](https://www.runpod.io/blog/how-to-achieve-true-ssh-in-runpod)。实际拓扑通过预检查决定是否继续。

## 测什么

形状统一为 `(P,M,K,N)`，P=4；M 是每个源—目标分块的行数。

| 工作负载 | M | K | N | 作用 |
|---|---:|---:|---:|---|
| w000 | 64 | 1024 | 512 | 原优先形状：短 K 对照 |
| w001 | 64 | 4096 | 2048 | 原优先形状：大 N 的 host / fused 误判 |
| w002 | 64 | 8192 | 512 | 原优先形状：长 K、q8 对照 |
| w003 | 256 | 8192 | 512 | 原优先形状：grouped-N 跨 M |
| w004 | 512 | 4096 | 512 | M512 的 q2 / q4 误选 |
| w005–w009 | 64/192/512 | 3072 | 512/1536 | 可选交叉设计，完整形状见 `bundle/plan.json` |

| 阶段 | 配置数 | 独立进程数 | 要回答的问题 |
|---|---:|---:|---|
| `anchors` 必做 | 10 | 30 | w001/w004 的 host、direct、q2、q4、q8 在 legacy 路径是否保留旧排序 |
| `mechanisms` 必做 | 59 | 177 | 五个重点形状的 metadata 开销、grouped-N、基线强度 |
| `geometry` 可选 | 25 | 75 | 区分 M/N tile 几何、工作集与 K 的影响 |

两个必做阶段合计 69 个配置、207 次进程运行；加可选阶段共 282 次。每配置 3 个新进程，每进程预热 20 次、计时 100 次，共 20,700 / 28,200 个原始计时样本。迭代样本不能当作独立重复。跨阶段同一配置可能重复，这是各阶段内部对照所需，不合并成额外独立区组。

`mechanisms` 在每个重点形状都测：host/q2/q4 的 legacy 与 prepared 两种路径；prepared q2/q4 的 g1/g2；per-source cuBLAS、merged CUTLASS、merged cuBLASLt。两个 K=8192 的形状另测 prepared q8 g1/g2。全部候选在同区组固定随机顺序交错执行，不能拿上一阶段的 host 耗时代替本阶段的 host。

这些候选覆盖当前问题，不构成完整搜索最优值。未测 direct-grouped、q16 等候选不能补标签。新测形状用于机制分析后也属于开发数据，不能再称独立验证集。

## 每项数据的用途

| 数据 | 如何采集 | 能判断什么 |
|---|---|---|
| 全程序耗时 | steady_clock 包围 execute + synchronize，保留全部样本、均值、中位数及极值 | 实际候选排序、慢了多少 |
| q2/q4 同区组差分 | 每个重复单独报告微秒差和比值 | M512 分割 K 后的净代价，不能单凭差分断言全是 semaphore |
| legacy/prepared 对照 | 同 family、相同 GPU kernel、分别记录实现身份 | CPU metadata 准备是否改变净收益；不等于完整 CPU/GPU 时间分解 |
| g1/g2 对照 | 相同 M/K/N/q，分别测量 | 减少 N tile 与增加 CTA 线程数后的实际净收益 |
| 四种 host 实现 | per-source CUTLASS/cuBLAS、merged CUTLASS/cuBLASLt，单独记录算法和分区 | 优势是否仍能超过更强的 host；merged 不共用 per-source 标签 |
| 编译资源 | ptxas 输出保存在 build.stderr.txt；真实二进制哈希 | 寄存器、共享内存等静态资源，不能直接当成实测 occupancy |
| CPU/GPU 时间线 | 主测完成后最多先选 4 个配置，单独 nsys CUDA/OS runtime trace | CPU 提交间隙、核函数时长、stream/event 等待的时序证据 |
| 正确性及环境 | 计时前后全输出检查、GPU UUID/拓扑、驱动/库哈希、区组边界状态 | 数据是否可接纳、是否存在环境漂移 |

`trace-candidates.json` 给出 M512 q2/q4 两种 metadata 模式和大 N host/q2/g2 的索引，只在主测结果需要解释时选择。第一轮不自动采集全部 trace，也不直接运行多 GPU NCU 全量回放。时间线不会给出完整 HBM/NVLink 字节量，执行台账中的逻辑字节也不能替代硬件计数。

Profiler 输出在单独目录，外部 `diagnostic.json` 明确标记 `profiler=true`；benchmark stdout 自身的默认未插桩声明在该目录无效。不能把 profiled stdout 送进主标签。参数根据 [Nsight Systems 官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，实际报告产生后再检查哪些区间可用。

## 服务器执行

解压后在本目录执行。命令彼此独立，每一步成功后再进行下一步；GPU 编号根据拓扑选择。若服务器已经提供合适的 NCCL，仍须核对源码及实际动态库身份。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 server.py verify
python3 server.py inventory
```

查看 `work/inventory-01.json`，确认有 nvcc、四卡型号及两两 NV12，再开始编译。包内包含依赖 Git 浅快照，不需要从 GitHub 拉取；默认单独构建 NCCL，不替换系统库。

```bash
python3 server.py build-deps --jobs 4
python3 server.py build --jobs 4
python3 server.py preflight
python3 server.py run --stage anchors --budget-seconds 900
```

先看 anchors 的正确性、环境和排序。没有错误再运行机制对照。首轮默认不自动进入 geometry。

```bash
python3 server.py run --stage mechanisms --budget-seconds 1800
```

需要补充交叉设计时：

```bash
python3 server.py run --stage geometry --budget-seconds 900
```

时间预算不足时返回码 75，保存已完成的每一行，使用同一 `--work`、`--label` 和阶段续跑：

```bash
python3 server.py run --stage mechanisms --budget-seconds 1800 --resume
```

每个测量进程上限 120 秒；失败立即停止，保存 stdout/stderr，不自动重试或跳过。预算检查会在剩余时间不足以容纳下一个进程及状态采集时提前结束。预算约束采集调度，不负责停止 RunPod 计费。修改源码、二进制、环境或卡选择后，不允许续写旧结果目录，应重新构建并使用新工作目录。

可选时间线例子：

```bash
python3 server.py trace --job-index 0 --label m512q2
python3 server.py trace --job-index 9 --label m512q4
```

先备份 `work/` 整体：它包含输入身份、原始样本、完整错误、构建日志、库/二进制身份和 profiler 报告。采集完成后下载结果再释放机器，不把唯一结果留在 Pod 内。

## 时间与停止条件

建议先安排约 2 小时，这是作业安排余量，不是已实测的运行时长或费用承诺。上传、NCCL/CUDA 编译、进程初始化和诊断通常比矩阵计时本身更占时间；不能用“282×100×1ms”估算整轮租用时间。

- 准备环境 / 预检查：先用约 5–10 分钟判断机器是否合适；型号、NV12 或 P2P 不满足就不做全量采集。
- NCCL 构建上限 20 分钟；本项目编译上限 30 分钟。日志持续写盘，失败停止，不在循环里重编。
- 必做采集先给 anchors 15 分钟、mechanisms 30 分钟上限。根据已完成进程数和耗时决定续跑，不以 timeout 总和当预计时长。
- 每个 trace 最多 180 秒，首批最多先选 4 个。
- 任一数值、身份或原始样本校验失败：停止该阶段；不产生合法标签。
- 三个进程中位数 max/min 超过 1.10：报告待复核，不删数据。此门槛不是显著性检验，也不能保证小于 10% 的差异就是稳定事实。

本轮结束时要形成“已知计数—实测净差—时间线解释—仍缺证据”四列记录，再决定成本模型哪些项必须改。不会因为新数据能拟合旧错误就自动验收模型。

## 本地验证范围

验收结果见 `verification.json`。本地检查覆盖打包身份、区组配对、算法/正确性政策、真实本地子进程超时与续跑、profiler 隔离、生成程序布局与 CUDA API 调用映射。CPU 调用映射使用替身，不执行 CUDA kernel。参考输出另做整数点积抽查。

当前尚未完成 CUDA 编译、GPU 全输出正确性或真实性能验证。旧台账、旧模型和旧执行包不改写。
