# A100 P4 边界诊断执行包

本轮只完成本地准备，未租用服务器、未运行 GPU。执行包可以独立搬到服务器；真实 CUDA 编译、GPU 数值正确性和计时尚未验证。当前预测器仍未通过 M512 与大 N 的联合稳定性要求，没有进入编译器默认选择路径。

## 本次要回答的问题

旧数据主要沿 M=64 或 N=512 两条轴变化，不能充分区分 M、N 与工作集的耦合。本次保留两个旧形状作为环境对照，增加五个诊断形状。先看固定候选实际怎么变，再决定模型表达式如何改；测量前已经保存四个既有模型的条件预测，没有重新拟合。

形状顺序统一为 **(P,M,K,N)**。P=4；M 是每个源—目标分块的行数，不能直接当作服务端请求 batch size。

| 类型 | M | K | N | 用途 |
|---|---:|---:|---:|---|
| 旧形状 | 512 | 4096 | 512 | M512 误选对照 |
| 旧形状 | 64 | 4096 | 2048 | 大 N 边界对照 |
| 新形状 | 64 | 3072 | 512 | 交叉设计起点 |
| 新形状 | 64 | 3072 | 1536 | 增大 N |
| 新形状 | 192 | 3072 | 512 | 与上一行具有相同输出 tile 数，改变 A/B 工作集 |
| 新形状 | 192 | 3072 | 1536 | 打破原有 M/N 轴混淆；当前模型对该点拒绝预测 |
| 新形状 | 512 | 3072 | 512 | M512 下改变 K |

每个形状测 `host / direct / splitk2 / splitk4 / splitk8`，共 **35 个配置、3 个独立进程区组、105 次进程运行**。每次预热 20 轮，保留全部 100 个计时样本，按预先固定的随机顺序执行。100 个迭代样本不是 100 次独立重复。

这批是模型诊断数据。若用来修改模型，之后必须另留验证数据。未纳入 grouped-N 和 splitk16，不能把本批最快者称为全部候选的最优解。

## 固定的实验口径

- Host 是 NCCL 搬运加每源 CUTLASS GPU GEMM，身份为 `host_nccl_cutlass_sm80_per_source_v1`；不等于 CPU 计算、cuBLAS 或 merged GEMM。
- 所有主测量显式使用 `legacy` metadata 路径。当前目标是诊断旧模型，不能混入 prepared 路径的额外收益。
- 计时边界是主机 `steady_clock` 包围 `execute()` 和 `synchronize()`；不启用 profiler/NVTX、不丢弃尾部样本。
- 硬件校验要求 **4× A100-SXM4-80GB、SM80/108 SM、所选卡之间 NV12 链路及 peer access**。这是匹配已有实验的约束，不表示输入本身需要 80 GB 显存。普通 PCIe A100 不满足本包口径。
- 使用新环境 ID，记录 GPU UUID、拓扑、实际动态库和编译依赖身份。不得把新数据重标为旧归档环境。
- 四个冻结预测来自旧环境，比较预测误差前需检查两个旧形状的环境漂移。它们不是新服务器上已经验证的预测。

## 包内内容

`bundle/` 包含 124 个冻结源码文件、35 套生成程序、运行顺序、四个模型的拟合状态与条件预测，以及 7 组 FP16 输入和 FP32 CPU 参考输出。输入和参考输出合计 **428 MiB**。`package.json`、`inputs.json`、`seal.json` 记录哈希和身份。

执行包 SHA-256（规范化 package 对象，不是压缩文件哈希）：

```text
3569fb7a6fa01ca05b84c9addac73823bb4ba2d25b7e5386e3614cbb3308a8a1
```

CUDA、NCCL、CUTLASS 和编译器不随包分发。服务器需安装 Linux、Python 3.10+、CMake 3.20+、支持当前源码的 C++20/CUDA 工具链、CUTLASS 源码及带 Device API 头文件的 NCCL。启动时直接使用包内冻结源码，不依赖原仓库，也不需要 NumPy/SciPy；仅重新生成参考值或拟合模型才需要额外 Python 依赖。尚不能保证任意 CUDA/NCCL/CUTLASS 版本组合都能编译。

## 搬到服务器后的命令

先核对压缩文件旁的 `.sha256`，解压。以下路径均需替换为服务器真实路径；将匹配的 `nvcc`、`c++`、`cmake` 放入 PATH。NCCL 头文件和动态库应来自所指定的 NCCL 源码构建，构建器记录身份但无法仅凭文件名证明其来源。CUTLASS/NCCL 源码需处于无已跟踪修改的 Git 提交。

```bash
export BOUNDARY_PACKAGE=/srv/boundary-diagnosis-a100-p4/bundle
export BOUNDARY_BUILD=/srv/boundary-build-01
export BOUNDARY_RESULTS=/srv/boundary-results-01

python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/prepare_boundary_gpu.py" verify \
  --package "$BOUNDARY_PACKAGE"

python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/run_boundary_gpu.py" build \
  --package "$BOUNDARY_PACKAGE" --build-dir "$BOUNDARY_BUILD" \
  --cutlass /srv/deps/cutlass \
  --nccl-source /srv/deps/nccl \
  --nccl-include /srv/deps/nccl/build/include \
  --nccl-library /srv/deps/nccl/build/lib/libnccl.so \
  --jobs 4

python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/run_boundary_gpu.py" preflight \
  --package "$BOUNDARY_PACKAGE" --build-dir "$BOUNDARY_BUILD" \
  --out /srv/boundary-preflight-01.json

python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/run_boundary_gpu.py" run \
  --package "$BOUNDARY_PACKAGE" --build-dir "$BOUNDARY_BUILD" \
  --out "$BOUNDARY_RESULTS"
```

构建目录使用新路径；预检查文件和首次测量目录不能已存在。预检查不一致或数值校验失败会停止，不会继续写入合法计时标签。确保所选四卡没有其他实验并发占用；采集器会在区组边界记录状态，但不会替代服务器资源隔离。

单个测量进程超时 120 秒即终止进程组，保留错误、进度和 stdout/stderr。问题处理后，只有包、构建和环境身份一致才能续跑：

```bash
python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/run_boundary_gpu.py" run \
  --package "$BOUNDARY_PACKAGE" --build-dir "$BOUNDARY_BUILD" \
  --out "$BOUNDARY_RESULTS" --resume
```

续跑只执行未完成项，不会自动重跑失败前已经校验并落盘的数据。若修改源码、依赖、GPU 选择或影响计时的环境变量，应使用新的构建/结果目录。超时上限是故障保护，不是正常运行时间估计。

## 单独采集时间线

仅在主测量结束后，对需要解释的配置单独运行。`job-index` 是 `plan.json` 中的零基序号。以下示例只检查第 0 项，按需替换，不必给所有 105 项开 profiler。

```bash
python3 "$BOUNDARY_PACKAGE/source/EDM-IR/tools/run_boundary_gpu.py" trace \
  --package "$BOUNDARY_PACKAGE" --build-dir "$BOUNDARY_BUILD" \
  --out /srv/boundary-trace-01 --job-index 0 --nsys nsys
```

这里调用 Nsight Systems 的 CUDA/OS runtime 时间线采集，关闭 CPU sampling 和 context-switch tracing；参数语义见 [NVIDIA 官方说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。输出目录独立，最长 180 秒，只产生诊断记录和报告，不产生 `measurements.jsonl` 或可导入的主测量 `run.json`。被包裹的 benchmark stdout 中仍有自身默认计时策略声明，应以外部 `diagnostic.json` 的 `profiler: true` 为准，禁止将该 stdout 作为无扰动标签。此步骤不采集 HBM/NVLink 硬件流量计数。

## 如何判断结果

`measurements.jsonl` 保存每个进程的全部 100 个样本及全输出正确性检查结果；`context.json` 保存环境；`summary.json` 保留三个进程区组各自的中位数、同区组 fused/host 比值、相同输出 tile 数对照，以及固定 K 的 M/N 交互差分。区组边界的 `telemetry_*.xml` 在主计时之外采集。

先检查两个旧形状的变化、环境和数值正确性，再解释新点。重点观察 M512 随 K 改变的 host/split-K 排序，以及 M/N 交叉点的代价变化。端到端差分能区分候选解释，但不能单独证明哪个物理资源是原因。任何模型修改之后，还要重新过原有稳定性门槛和真正独立数据，采集器不会自动认可或发布模型。

## 已完成的本地验收

- 17 项 Python 测试通过：6 项新包测试、11 项既有打包/采集安全测试；包括真实本地子进程超时、已完成数据保留、续跑只执行剩余项和 profiler 目录隔离。GPU 检查在这些测试里由替身提供。
- 72 项 C++ 检查通过：35 项布局、35 项 CUDA API 调用映射、2 项运行时检查。使用 CPU 测试替身，不执行 CUDA kernel。
- 7 组输入中抽取的 1,008 个参考元素通过独立整数点积精确核对；同位置模拟 q=1/2/4/8 的 FP16 分段累加，最大绝对误差为 0.01171875。这是参考值和误差风险抽查，不是 GPU 全量正确性结论。
- 旧模型、旧审查证据和旧 GPU 包按原哈希核对；本轮没有重新拟合或改写旧结论。

详细验收记录见 `verification.json` 和 `evidence/`（压缩包内）。交付目录旁的 `delivery.json` 记录压缩文件哈希及异路径解压校验结果。
