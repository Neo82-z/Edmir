# split-K 的四组对照与 GPU 小批运行包

本轮把可以在本地完成的四步落实到了代码、逐点结果和运行包。**没有启动租卡，没有 GPU 实测结果；四个 split-K 版本都未通过预先写下的替换条件。** 原 relative 模型继续作为离线比较项，新模型没有接入编译器自动选择。

## 1. 改动范围和判断规则

只改 `g=1` 的 split-K/direct 时间比值模型。host/direct 判断、grouped/plain 模型、grouped-N 的支持范围、仅由 direct/host 得到的校准偏移均保持原样。

[design.json](design.json) 在本轮回放前写入。四组组合为：

- 原四个特征，拟合各 q 相对 direct 的对数时间比。
- 原四个特征，拟合同一形状内所有 q 两两之间的对数时间比。
- 新四个特征，拟合各 q 相对 direct 的对数时间比。
- 新四个特征，拟合同一形状内所有 q 两两之间的对数时间比。

四组都使用 ridge=1；没有搜索超参数。每个 workload 的总拟合权重为 1，同一特征组在两种损失下使用同一训练集 RMS 缩放。q=1 的分量恒为 0，不设截距。P4 按完整 group 留出，拟合和缩放均只使用该折训练数据。所有 P8 结果仅供诊断，不参与选择版本。

新特征令 `G=ceil(M/64)*ceil(N/128)`，使用：

```text
log2(ceil(G*q/108) / (q*ceil(G/108)))
log2(q)
(q-1)/16
32*(q-1)/K
```

第一项描述理想 CTA 批次与 K 分区工作量的组合；其余项允许分区和短 K 分块带来额外代价。它们是预测假设，不是实测 occupancy，也不是时延恒等式。108 SM、64×128×32 tile 都是本轮 A100/kernel 的固定条件。

原特征/原损失重现了上一版拆分模型的结果。独立 NumPy 实现手工重建特征、权重和缩放，用增广矩阵 SVD 求解；与生产代码正规方程求解的最大系数差约 `2.25e-14`。因此，目前没有证据把明显退步归因于线性求解算错。见 [numerical_audit.json](numerical_audit.json)。这项检查不证明预测有效。

## 2. 本地结果：改善存在，但不足以替换

regret 定义为 `选中候选的实测时间 / 当前候选集最快实测时间 - 1`，越低越好。这里都是历史点估计，不是新测量或置信区间。

P4 留组，41 个形状；完整候选集包括 host 和实际测过的 fused 候选。涉及 grouped-N 的两个预测器使用相同支持限制。

| 方法 | 平均 regret | 最大 regret | 慢于 host 的形状数 |
|---|---:|---:|---:|
| 训练集选出的固定策略（各折为 split-K4） | 5.43% | 18.18% | 3 |
| 原 relative + 相同 grouped 支持范围 | 3.54% | 29.24% | 2 |
| 原特征 / 相对 direct 拟合 | 25.29% | 87.86% | 13 |
| 原特征 / 成对拟合 | 18.34% | 54.29% | 2 |
| 新特征 / 相对 direct 拟合 | 12.12% | 39.57% | 1 |
| 新特征 / 成对拟合 | 9.88% | 39.57% | 0 |

只看 g=1 内部 q 排序，不含 host/grouped：

| 方法 | 平均 regret | 最大 regret |
|---|---:|---:|
| 训练集选出的固定 q | 4.34% | 16.45% |
| 原特征 / 相对 direct 拟合 | 28.08% | 87.86% |
| 原特征 / 成对拟合 | 21.86% | 71.02% |
| 新特征 / 相对 direct 拟合 | 7.20% | 31.25% |
| 新特征 / 成对拟合 | 3.20% | 17.83% |

**排序的改善没有同步修复时间比值的幅度。** 例如留出的 `(P,M,K,N)=(4,128,8192,512)`：

- 新特征/成对拟合把 split-K4 相对 direct 的时间比预测为约 0.72，实测约 0.39。
- 与 host/direct 分量相加后，预测 split-K4/host 约 1.40，实测约 0.72。
- 模型因此退回 host，损失为 39.57%。这一点的 host/direct 对数预测为 0.669，实测为 0.613；主要差距仍在 split-K 增益幅度。

所以“没有选出比 host 更慢的候选”也不能单独证明可靠。过多退回 host 同样会失去本可获得的收益。下一轮应单独检验 split/direct 增益随 K 的表达及倍率拟合，不能只优化 q 排名；本轮不据 P8 结果继续调参。

P8 校准后的平均 regret 依次为 34.76%、26.88%、9.99%、14.76%；原 relative 为 8.65%。这些只是迁移诊断，全部逐点输出保存在 [results.json](results.json)。没有因某一列较好而更换预设标准。

## 3. host 身份和执行方式

新增 [merged_host_backend.cpp](../../EDM-IR/runtime/src/merged_host_backend.cpp)，与原 `CudaBackend` 分开。原编译器回退和历史 host 执行器未被替换。

| 候选 | 每个目的 rank 的 GEMM | 实现 |
|---|---|---|
| `host` | P 个 M×K GEMM，串行 | 原 host NCCL + CUTLASS 64×128×32 |
| `host_compiler_cublas` | P 个 M×K GEMM，串行 | 原 host NCCL + cuBLAS |
| `host_merged_cutlass` | 1 个 PM×K GEMM | 新 merged host + 同一 CUTLASS kernel |
| `host_merged_cublaslt` | 1 个 PM×K GEMM | 新 merged host + cuBLASLt |

合并合法的前提是每个目的 rank 对所有 source 使用同一个 W。新实现直接接收到按 source 排列的连续 `[P*M,K]` 缓冲区，本地块也要复制；没有在计时范围外偷偷做拼接。每个目的 rank 等自己的通信完成后执行一个 GEMM。NCCL send/recv 的循环顺序和每个远端 pair 的分组方式与原 host 一致，但 merged 路径减少了原调度器中的逐 GEMM 事件/派发；因此它比较的是完整执行策略，不是单纯 kernel 用时。

cuBLASLt 使用 FP16 输入/输出、FP32 compute，32 MiB workspace 上限，在准备阶段选择一个 heuristic 结果。记录每个 rank 的算法 ID、tile、split-K、reduction、swizzle、custom option、stages 和 workspace。**没有按实测时延调优，不能将它称为 cuBLASLt 的性能上限。** 接口依据：[NVIDIA cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/)。没有可用算法时直接失败，不改名回退到别的库。

`uniform_workload.cpp` 检查完整 source/destination 对、相同权重、未分块 M、FP16 矩阵大小及输出映射；拒绝不符合此实验语义的 Program。

## 4. 已准备的小批实验

[gpu_pilot/package.json](gpu_pilot/package.json) 包含源码与生成文件哈希、随机顺序和全部任务；[gpu_pilot/inputs.json](gpu_pilot/inputs.json) 包含真实输入及 CPU 参考文件的哈希。输入/reference 二进制按仓库规则不进 Git，复制目录时应带上 `data/`，或在本地重新生成。

| P | M | K | N | 目的 |
|---:|---:|---:|---:|---|
| 4 | 64 | 1024 | 512 | 短 K、额外分区开销 |
| 4 | 64 | 4096 | 2048 | 大 N 下 q 排序 |
| 4 | 64 | 8192 | 512 | 长 K 的 split-K/grouped-N |
| 4 | 256 | 8192 | 512 | 与 M=64 成对检查 grouped-N |

每个形状 14 个候选：四个 host，以及 q∈{1,2,4,8,16}、g∈{1,2} 的十个 fused。每个候选 3 个独立进程，每个进程 20 次 warmup、100 次计时，共 **168 个进程、16,800 个计时样本**。cuBLASLt 和 merged CUTLASS 复用 host 的语义描述，因此只需 48 个生成 Program。

所有候选共享同一组非零 FP16 输入。CPU 先用这些 FP16 数值转成 FP32 计算参考值；每次进程在计时前后检查全部 `P²MN` 个输出。NaN/Inf、错误候选身份、不完整样本、误差不达标均停止。计时边界仍为 host `steady_clock` 围住 `execute()+synchronize()`；准备、CPU 参考和输入上传均在计时外。

先核对型号、108 SM、P2P、物理 PCI 对应的 NV12 拓扑，再执行测量。记录实际运行库版本及 Linux loader 解析到的库哈希，检查 NCCL 与构建记录一致。单个进程 180 秒超时；失败保留 stdout/stderr，可使用 `--resume` 接着做未完成的项。

目标是 **4 张 A100-SXM4-80GB、相互 NV12**，例如符合该拓扑的 8 卡节点中的 4 卡。普通 PCIe A100 或其他 4 卡互联不能直接冒充原实验环境。无需先租 H100；单卡也不能验证这项完整通信算子。

### 本地检查和服务器入口

以下命令都从仓库根目录执行。`<...>` 为服务器真实路径。

```bash
# 仅本地 CPU 布局检查，已完成
cmake -S EDM-IR/runtime -B /tmp/edmir-selector-local \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEDMIR_EXPERIMENT_DIR="$PWD/experiments/2026-10-20-splitk-local-revision/gpu_pilot"
cmake --build /tmp/edmir-selector-local -j 4
ctest --test-dir /tmp/edmir-selector-local --output-on-failure

# 服务器构建：记录真实 CUTLASS/NCCL commit、头文件和动态库
python3 EDM-IR/tools/run_selector_gpu.py build \
  --package experiments/2026-10-20-splitk-local-revision/gpu_pilot \
  --build-dir /tmp/edmir-selector-gpu \
  --cutlass <CUTLASS源码目录> --nccl-source <NCCL源码目录> \
  --nccl-include <NCCL头文件目录> --nccl-library <libnccl.so实际路径>

# 运行小批；失败后在同一命令末尾加 --resume
python3 EDM-IR/tools/run_selector_gpu.py run \
  --package experiments/2026-10-20-splitk-local-revision/gpu_pilot \
  --build-dir /tmp/edmir-selector-gpu --out <新的结果目录>
```

NCCL 需要支持当前 Device API 的版本（2.28+）；CUDA/CUTLASS 仍需完成真实构建。构建脚本记录 NCCL 源码 commit 和所链接库的哈希，但二者的构建来源仍应由服务器构建日志确认。不得填写猜测的 commit。

若只通过 Git 取得运行包，缺少 `data/*.bin`，可在有 NumPy 的 CPU 环境生成完整的新包：

```bash
python3 EDM-IR/tools/prepare_selector_gpu.py prepare \
  --out <新的小批运行包目录> --with-inputs
```

本次四组输入和参考文件共 253,755,392 字节（242 MiB），已在本地生成并核对哈希。不要人为改哈希来绕过核对。修改代码后也需重新准备新包，原包会拒绝继续使用。

### 独立验证保持冻结

原 [P4](../2026-10-20-predictor-contracts/independent_p4_plan.json) 和 [P8](../2026-10-20-predictor-contracts/independent_p8_plan.json) 的文件与来源哈希均未改变，仍各有 45 个新 M/K/N 组合。运行包工具可读取冻结计划：

```bash
python3 EDM-IR/tools/prepare_selector_gpu.py prepare \
  --plan experiments/2026-10-20-predictor-contracts/independent_p4_plan.json \
  --out <新独立实验目录> --with-inputs
```

此模式使用冻结预测和顺序，5 个独立进程重复，共 3,150 条候选中位数；收集完成后调用已有严格评分器。小批中若改了预测器，必须先冻结新模型和新计划，再获取独立标签，不能沿用旧模型的预测文件替它背书。P4/P8 使用相同 M/K/N 网格，不能把看过 P4 后再调模型的 P8 同网格叫作第二次全新独立测试。

## 验证边界

- 203 项 Python 测试通过，包括旧逻辑回归、split-K 对照、输入/参考的标量复核、采集失败续跑。
- 49 项 C++ 主机端检查通过，包括 48 个生成 Program 的矩阵与 merged 行映射，以及原 runtime 测试。
- 新 merged backend 和统一测量入口通过 Apple Clang C++20 的仅语法检查，使用 NVIDIA 官方 CUDA 12.8、cuBLAS 12.8 和 NCCL 2.28.9 头文件。
- **未完成 Linux/NVCC 构建、GPU 链接、NCCL 运行、GPU 数值校验、计时和独立可靠性验证。语法检查不等于上述验证。**

完整检查记录见 [verification.json](verification.json)。原始时间、旧模型、原冻结协议均保留；失败的四组结果也保留。
