# 四形状 A100 实验：主机元数据预处理与测量整理

已完成主机元数据预处理、强 host 基线的对照编排，以及保留全部样本的计时与汇总。**没有租卡或运行 GPU；尚不能判断加速比。** 硬件目标仍是相互 NV12 的 4 张 A100-SXM4-80GB，SM80、108 SM，不运行 DeepInfer，也不引入 SM100。

[design.json](design.json) 在修改执行器前写入，限定四个形状和对照范围。[verification.json](verification.json) 保存本地验证结果、源码哈希及剩余验证事项。

## 已完成的改动

### 1. 把固定元数据处理移到准备阶段

[prepared_execution.cpp](../../EDM-IR/runtime/src/prepared_execution.cpp) 在准备阶段确定：

- 每个 op 的输入、权重及输出 buffer；
- transfer 的字节数和元素数；
- op 涉及的逻辑设备及原来的遍历顺序；
- 去重后的前置依赖、需要等待的事件及所属 rank；
- 与原执行器相同的拓扑执行顺序。

[cuda_backend.cpp](../../EDM-IR/runtime/src/cuda_backend.cpp) 的 host 和 fused 路径都可以直接读取这些记录。原 P=4 host 每轮对访问表扫描 7,424 条记录，direct fused 扫描 1,536 条；预处理路径在 `execute()` 中不再调用这些访问表扫描，也不再为参与设备和依赖查询临时构造列表。CUDA/NCCL 派发、完成事件、必要等待和 kernel 参数保持原来的顺序。

这减少的是重复主机工作，不是取消通信或同步。适用条件是 Program 准备后重复执行，其 backing arrays 必须保持存活和不可变。准备工作仍然有成本；本轮沿用原来的稳态计时边界，不声称冷启动或一次性执行获得同样收益。

使用 `CudaBackendOptions.prepare_metadata=true` 开启。通用执行器默认仍为 false，避免旧调用方和旧预测器 profile 静默改变身份。新实验为预处理实现增加 `_prepared_metadata_v2` 后缀，不能改名混入旧标签。

### 2. 保留两个 merged host 强基线

仍包含 per-source CUTLASS、per-source cuBLAS、merged CUTLASS 和 merged cuBLASLt。后两个继续使用已有的连续接收布局，每个目的 rank 执行一个 PM 行 GEMM，没有增加新的合并、通信或 kernel 优化因素。

合并的前提仍是同一目的 rank 的各 source 共用一个 W。它与 per-source 执行的 GEMM 行数及派发次数不同，因此比较的是完整执行策略，不能直接解释为单一 kernel 机制的优势。cuBLASLt 仍只使用准备阶段选出的一个 heuristic 算法，不是调优后的性能上限。

### 3. 保留完整测量结果

[benchmark_selector.cpp](../../EDM-IR/runtime/tools/benchmark_selector.cpp) 继续用主机 `steady_clock` 包围 `execute()+synchronize()`。准备、输入上传及数值检查在计时外，所有候选使用同一边界；计时样本数组提前分配容量。

每个进程保存全部 100 个样本，以及中位数、均值、总耗时、最小值、最大值、元数据模式和观测配置。没有删除慢样本，也没有用设备侧 kernel 时间替换完整算子时间。

[run_selector_gpu.py](../../EDM-IR/tools/run_selector_gpu.py) 校验统计值确实由全部样本计算；拒绝 prepared/legacy 身份混淆、计时边界不符及声明过滤样本的记录。新实验运行前检查 `CUDA_LAUNCH_BLOCKING`、CUDA profiler 注入、已知 profiler preload 和 NCCL INFO/TRACE 日志配置，并记录相关环境变量。主计时进程不请求 profiler 或 NVTX；这些检查不能穷尽外部附加的采集器或系统负载。

汇总保留各独立进程的均值和中位数，并逐重复编号给出 legacy/prepared 比值，不把重复记录覆盖成最后一次。100 个进程内样本不能当作 100 次独立实验。

## 四个形状与有限对照

| P | M | K | N |
|---:|---:|---:|---:|
| 4 | 64 | 1024 | 512 |
| 4 | 64 | 4096 | 2048 |
| 4 | 64 | 8192 | 512 |
| 4 | 256 | 8192 | 512 |

M 是每个 source→destination 块的行数；merged GEMM 使用 PM 行，不能把它与原 per-source GEMM 的 M 混为一谈。

每个形状的主要比较项仍是 14 个：2 个预处理后的 per-source host、10 个预处理后的 fused（q∈{1,2,4,8,16}、g∈{1,2}），以及 2 个现有 merged host。

仅增加以下 4 个 legacy 元数据路径对照：`host`、`host_compiler_cublas`、`direct`、`splitk4_groupedn2`。legacy 和 prepared 在同一新版二进制内显式切换，用于隔离本轮元数据改动；legacy 对照不是历史二进制性能的重放。

合计 **4 形状 × 18 候选 × 3 独立进程 = 216 个进程**，每进程 20 次 warmup、100 次计时，共 21,600 个计时样本。候选顺序提前随机化并写入运行包，不扩大形状或开展整模型实验。只有 3 次进程重复，适合初步判断方向，不支持可靠的服务 p99 或自动替换决定。

## 本地验证

- **206 项 Python 测试通过**。最终采集脚本调整后，相关 8 项测试再次通过，覆盖完整候选表、实现身份、尾部样本保留、统计一致性、重复配对及已知测量干扰配置。
- **97 项 C++ 检查通过**：48 项生成程序的布局与元数据检查、48 项真实执行器的 CPU API 调用序列对照、1 项原 runtime 检查。
- 元数据检查还覆盖访问表重排、依赖顺序改变、重复依赖、无依赖及缺失操作数拒绝。
- API 对照使用官方 CUDA/NCCL/cuBLAS 头文件和 CPU 测试替身。在正常及逆序设备映射、连续两轮执行下，legacy/prepared 的准备阶段及执行阶段调用序列、参数、指针绑定、事件和同步均一致。它不执行 kernel，不模拟 GPU 并发，也不能证明 GPU 数值正确性。
- `cuda_backend.cpp`、merged backend 和测量入口通过 Apple Clang C++20 的 CUDA API 头文件语法检查，包含相关条件编译分支；没有完成 NVCC 编译或 GPU 链接。
- 四组输入及 CPU FP32 参考与旧包的元数据、文件哈希完全一致，总计 242 MiB。新包 109 个源码文件的哈希已核对。

日志在 [checks/](checks/)。实际加速、GPU 数值、通信依赖正确性和重复运行稳定性仍待 A100 验证。

## 运行包和入口

[gpu_pilot/package.json](gpu_pilot/package.json) 固定源码、生成程序、候选身份及运行顺序。包 ID：

```text
a7db891a99a98a2e7099bacdb7a2485565bf1f0b556a823179c3f09d042a0a2d
```

以下命令从仓库根目录运行，服务器路径需替换为实际值；本轮没有执行这些 GPU 命令。

```bash
python3 EDM-IR/tools/run_selector_gpu.py build \
  --package experiments/2026-10-20-host-metadata-optimization/gpu_pilot \
  --build-dir /tmp/edmir-metadata-gpu \
  --cutlass <CUTLASS源码目录> --nccl-source <NCCL源码目录> \
  --nccl-include <NCCL头文件目录> --nccl-library <libnccl.so实际路径>

python3 EDM-IR/tools/run_selector_gpu.py run \
  --package experiments/2026-10-20-host-metadata-optimization/gpu_pilot \
  --build-dir /tmp/edmir-metadata-gpu --out <新的结果目录>
```

NCCL 仍需支持当前 Device API。采集器检查 A100 型号、SM 数、P2P、NV12 拓扑、实际动态库、完整输出及前后数值误差；单个进程超时保留失败信息，同一环境可显式使用 `--resume`。

`data/*.bin` 按仓库规则不进 Git，复制运行包时应带上数据。仅通过 Git 获得代码时，可用 NumPy 重新生成一个新目录：

```bash
python3 EDM-IR/tools/prepare_selector_gpu.py prepare \
  --out <新的运行包目录> --metadata-ablation --with-inputs
```

普通 CPU 布局和元数据检查：

```bash
cmake -S EDM-IR/runtime -B /tmp/edmir-metadata-cpu \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEDMIR_EXPERIMENT_DIR="$PWD/experiments/2026-10-20-host-metadata-optimization/gpu_pilot"
cmake --build /tmp/edmir-metadata-cpu -j 4
ctest --test-dir /tmp/edmir-metadata-cpu --output-on-failure
```

若要加上 CPU API 调用对照，在配置时开启 `EDMIR_CUDA_API_TRACE_TESTS=ON`，并通过 `EDMIR_CUDA_API_HEADER_DIRS` 提供分号分隔的官方 CUDA runtime、nvcc、CCCL、cuBLAS、NCCL 和 NVTX 头文件目录。该选项与真正的 `EDMIR_ENABLE_CUDA` 互斥，测试替身不会链接进 GPU 实验程序。

## 历史记录与判断边界

修改前的 104 个源码文件保存在 [legacy_source/](legacy_source/)，由 [legacy_source_manifest.json](legacy_source_manifest.json) 核对。原 GPU 小批包可对照这份源码通过验证；当前改进后的源码必须使用新包，不能修改旧包哈希绕过检查。

P4/P8 独立计划及其模型源码校验仍通过，未修改预测器参数或借旧 GPU 标签证明新执行器。它们保持原来的实验身份；未来若要让新实现参与独立预测验证，需要先冻结相应的新模型、基线和计划。

后续先看 4 组 legacy/prepared 对照是否在独立进程间表现一致，再看预处理后的 fused 是否仍能胜过最快的强 host。报告均值与中位数的差别，并保留所有重复；若收益与波动同量级，增加重复或报告不确定，不继续扩大形状来寻找胜出的结果。CUTLASS 参数缓存、通信重排和 kernel 改动均不混入这轮对照。
