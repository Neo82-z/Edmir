# CPU 准备、模型候选与硬件一致性

日期：2026-10-29。状态：三个模型权重和 GPU 软件缓存准备完成，CPU 已停机，500 GB 持久网络卷保留在 US-NE-1；尚未通过 GPU 验证。78 个权重分片的大小及 SHA256 通过，186 个 wheel 的离线依赖解析通过。GLM 分词器仍有正则警告，GPU 驱动、引擎运行、路由采集和性能继续单独验收。实际交付见 [资产交接](../../experiments/2026-10-29-cpu-preparation/ASSET_HANDOFF.md) 与 [RunPod 状态记录](../../experiments/2026-10-29-cpu-preparation/RUNPOD_STATUS.md)。补充并调整 [P8 与 serving 计划](NEXT_P8_AND_SERVING_PLAN_2026-10-29.md) 的单模型准备建议；不改既有实验协议、评分器或冻结六形状。

## 1. CPU 先行，但准备完成与 GPU 验收分开

同意先使用无 GPU 的 Linux x86_64 环境，下载已选资产、构建镜像、准备采集程序。建议起点为 8 vCPU、32 GB RAM、约 400–500 GB 持久空间；下载本身不需要高 CPU，源码编译可按需要增加 RAM/并行度。该容量面向两个约 30B–35B BF16 主候选、一个轻量备选及软件缓存；若再保留 Qwen3-Next 80B 或 Llama4 Scout，需重新按实际文件总量核算。不是租用报价或订单。

- CPU 可做：固定 revision 的权重/tokenizer/config 下载与校验、镜像和 wheel 准备、显式目标架构的部分离线编译、输入生成、路由统计/守恒/超时恢复测试、报告解析。
- GPU 才能做：驱动/库实际兼容、JIT 和 autotune、峰值显存、真实路由、拓扑/P2P、正确性、性能及计数器权限。CPU 的 mock 成功不能把这些标记为通过。
- macOS 本机适合控制和分析，不直接把本机 arm64 安装物当 Linux x86_64 GPU 包。目标镜像 digest、Python/CUDA/库 ABI 与 GPU 节点一致。
- 预先准备 GPU 版软件；不用 CPU-only vLLM/PyTorch 代替最终 GPU 环境。JIT 缓存按模型、源码、SM 架构、编译器和运行时区分，跨架构不能默认复用。
- 存储必须能交付到最终 GPU 地点。RunPod 普通 network volume 的地点会限制可用 GPU，创建前核对 A100/H100 库存地区。CPU 节点与 GPU 节点是否能挂同一卷需核实；可使用 S3 接口在不开 GPU 时预填充卷。不要假设 CPU Pod 可以原地升级成目标 GPU Pod。

参考：[RunPod network volumes](https://docs.runpod.io/storage/network-volumes)。已获授权的实际规格为 8 vCPU / 16 GB，CPU $0.24/hour，20 GB 容器盘约 $0.003/hour；500 GB 标准网络卷 $35/月独立收费。CPU 只用于资产准备，不执行内存密集的引擎源码编译。

## 2. 模型预备池可以多家，正式测量分主次

候选选择依据是 MoE 结构差异、真实模型质量、可固定的引擎支持、显存与集成成本，不是只按模型厂商或榜单名次。先准备两个正式候选和一个轻量备选；保留其余模型元数据，避免全部权重、全部配置做笛卡尔积。

| 模型 | 建议用途 | 规模与限制 |
| --- | --- | --- |
| GLM-4.7-Flash | 非 Qwen 的主要 MoE 候选 | 约 31.2B 实际张量参数，BF16 权重文件约 62.4 GB；64 routed experts、top-4、专家 intermediate 1536 |
| Qwen3.6-35B-A3B | 第二种 MoE 架构候选 | 约 71.9 GB BF16 张量；256 experts、top-8、intermediate 512；只测文本路径，固定视觉模块配置 |
| DeepSeek-V2-Lite-Chat | 低成本真实 MoE 调试与结构对照 | 约 31.4 GB BF16 张量；64 experts、top-6、intermediate 1408；明确它是较早模型，不作为当前最强能力代表 |
| Qwen3-Next-80B-A3B-Instruct | 保留上一计划的连续性，可替代上述一个主候选 | 约 162.65 GB BF16 张量；不能因已写进计划而强制同时跑全部模型 |
| Llama-3.1-8B-Instruct | 可选 dense 引擎/采集链路对照 | 约 16.1 GB BF16 张量估算；不是 MoE；本次匿名接口读取配置被门控 |
| Llama-4-Scout-17B-16E-Instruct | 可选大模型 MoE 扩展 | 约 217.3 GB BF16 张量估算；需要账户授权和内存/并行配置审查，暂不进入首批权重下载 |
| DeepSeek-V4.1-Flash-1017 | 阅读与接口参考 | 当前发布权重文件约 510.3 GB，精度混合且含 FP4 专家；不属于单卡/四卡 80GB 全 GPU 驻留实验 |

单卡 H100 80GB 不保证能以目标上下文/并发运行一个 62–72 GB checkpoint：还需非专家权重、KV/状态、workspace、CUDA Graph 和加载峰值空间。EP 分片也不是简单地将所有权重除以 GPU 数；非专家部分可能复制。正式容量以目标配置 smoke 为准。

现有元数据来源、revision、分片大小、LFS 哈希及引擎检查见 [metadata-audit.json](../../experiments/2026-10-29-cpu-preparation/metadata-audit.json)。可用同目录 `audit_metadata.py` 重做公开元数据审查；它不下载权重、不运行模型代码、不连接 GPU。

发现的源数据异常：GLM 索引 `metadata.total_size=31,221,488,576`，但 safetensors 文件大小总和为 `62,444,175,504` 字节。以实际文件大小规划下载；不能把前者当作 BF16 字节总量。下载完成后还需核对实际文件及校验和。

模型与权重文件来源：[GLM](https://huggingface.co/zai-org/GLM-4.7-Flash)、[Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、[DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)、[Llama4](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)、[V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。R1-Distill-Qwen/Llama 不能因为带 DeepSeek 名称就当成原生 DeepSeek MoE。

## 3. 软件分成主实验环境与可选研究环境

### 主实验候选

以 **vLLM v0.20.1** 为首个待验证共同版本。该 tag 的 registry 包含 GLM4 MoE Lite、Qwen3.5 MoE（用于 Qwen3.6）、DeepSeek V2、Qwen3 Next 和 Llama 对应架构；注册存在不是所有 EP/后端组合已可运行。Qwen3.6 官方推荐 vLLM >=0.19.0；GLM 发布时卡片里的 nightly 提示应结合现有固定 tag 源码判断，不能永久依赖滚动 nightly。

已读该 tag 的 CUDA requirements：torch/torchaudio 2.11.0、torchvision 0.26.0、FlashInfer 0.6.8.post1、TileLang 0.1.9。准备时以该版本整套依赖和镜像为单位，解析后保存锁文件、wheel 哈希和镜像 digest。当前只是源码候选，未取得/构建镜像，未完成安装与 CPU/GPU smoke；CUDA/NCCL/驱动兼容仍待目标镜像核对。不得直接覆盖旧 A100 算子测量包的 CUDA/NCCL。

源码：[registry](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/registry.py)、[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.20.1/requirements/cuda.txt)。需要修复引擎时另建版本并在正式对照前冻结；不能只更新优化组。

### 观测工具与职责

| 工具 | 准备用途 | 运行规则 |
| --- | --- | --- |
| Nsight Systems + NVTX | CPU 提交、CUDA kernels、stream、事件和通信时间线 | 真正的全程序流水图以此为主；短窗口、多进程/rank 标识；诊断与正式计时分开 |
| Nsight Compute | 少量 kernel 的驻留、资源、访存、Tensor Core 和 stall | 不能拿普通 kernel replay 的结果当原始多卡 overlap 耗时；只分析可安全采集的控制 |
| PyTorch profiler / vLLM stats | 算子关联、请求阶段和调用路径 | 不与另一套 CUPTI profiler 同时包裹正式计时；按需独立运行 |
| nvidia-smi、可用时的 DCGM | 拓扑、显存、频率、温度、功率与错误背景 | GPU 才能确认可见性与采样权限 |
| NCCL tests、P2P/带宽检查 | 环境通信能力与异常排查 | 单独的系统诊断，不替代我们的完整算子基线 |

CPU 预装 CLI、报告解析器和本地 GUI/导出路径。GPU 开始时先确认 profiler/CUPTI/性能计数器权限；平台不可用时登记 unavailable，不在正式跑分中盲目叠加工具。正式计时无 profiler，输出原始样本；`.nsys-rep` 和 `.ncu-rep` 单独归档。时间线中的重叠面积不是因果收益，仍需同工作量对照。

参考：[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#workload-durations)。

### 可选研究组件

- **TileLang**：在主环境沿用 vLLM 依赖版本；如需新算子研究，用独立环境，不全局升级破坏基线。按目标 kernel 区分 SM80/SM90，不能假设同一示例全架构适用。
- **DeepEP**：当前主分支要求 Hopper+、CUDA >=13.1、PyTorch >=2.10、NCCL >=2.32.3 和节点内 NVLink。可在 CPU 构建 host 部分，但 GPU kernels 运行时 JIT；SM 预算可借鉴，是否接入固定 vLLM 需核对 API 代际。不能因为都是 DeepEP 名称就替换依赖。
- **DeepGEMM**：当前文档要求 SM90/SM100 和 CUDA >=12.9；具体 kernel、精度与 layout 另核对。H100 可进入兼容性审查，并不证明 MegaMoE 等全部接口可用。
- **DeepInfer**：当前官方公开支持 NVIDIA **SM100** / Ascend950，H100 是 **SM90**，不能直接满足。保存源码与 benchmark 方法作参考，不把移植 DeepInfer 变成本轮交付前提。
- 可选组件独立镜像/环境，且先准备小型正确性入口；不能将所有“最新”包强行装进同一个环境。DeepEP/DeepGEMM 是通信/计算组件，DeepInfer 是引擎，都不是通用观测工具。

来源：[DeepEP](https://github.com/deepseek-ai/DeepEP#requirements)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM#requirements)、[DeepInfer](https://github.com/deepseek-ai/DeepInfer#readme)、[Hopper SM90](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。版本要求是本次阅读时的主分支状态，实际安装前须固定 commit，而不是运行时重新拉 main。

## 4. 八卡 A100 与一/四卡 H100 如何保持一致

本方案按“**H100 最多一或四卡，八卡 A100 仍是原主线**”理解。硬件矩阵如下：

| 环境 | 负责的证据 | 不能代替的证据 |
| --- | --- | --- |
| A100 ×8，原通信域 | 原 P8 算子/候选验证，以及原协议允许的应用实验 | 不代表 Hopper 性能 |
| H100 ×1 | 模型加载、单卡 kernel、正确性、hook 和 profiler 入口 | 多 GPU 通信争用、EP4/EP8 或 P8 收益 |
| H100 ×4，同一 NVLink 域优先 | 明确标注 P4 的机制或应用验证 | 八卡扩展性与八卡冻结评价 |
| A100 ×4，可在八卡机器取子集 | 如要做跨硬件对照，连接 A100 P8 与 H100 P4 的共同 P4 条件 | 不能直接用两种硬件原始耗时比值归因方法改进 |

八卡与四卡可以出现在同一论文中，但必须标注不同的任务/证据，不能把 A100×8 的 FP16 算子收益拼接为 H100×4 BF16 serving 收益。单卡没有非平凡的跨 GPU EP 通信，不能用于证明本文核心 overlap。

当前源码 `EDM-IR/runtime/CMakeLists.txt` 指定 CUDA architecture 80，CUTLASS 使用 Sm80/half_t。H100 并非自动可验收：先确认编译/PTX/指令兼容，再记录采用了什么 kernel；“旧路径能运行”与“面向 H100 优化的实现”分开。H100 上必须有该硬件的强 host/引擎基线，不能靠未优化的基线宣布胜出。新增 SM90 优化需要独立实现与校验，不为使用新工具而默认增加移植工程。

比较时固定数学语义、dtype、输入/路由、token/请求定义、正确性要求、计时边界和基线角色；硬件、GPU 数、拓扑、kernel/库身份作为显式条件。共享模型结构可以保留，硬件标定参数不能偷用 A100 数字。跨硬件零样本迁移和重新标定后性能分别报告。

真实 serving 以每个模型/硬件内部的 matched A/B 为主，不将不同模型原始 tokens/s 作优化优劣比较。GPU 数变化时，固定全局 workload 与固定每卡 workload 是两个不同问题，事先选定，不混在一个 scaling 图里。FP16 机制数据与 BF16 主应用数据分开，FP8/FP4 作为新精度条件，不能悄悄替换。

冻结六形状保持不变、不解析、不评分。若最终所有平台都只能四卡，需要正式新增 P4 确认性协议并收窄论文声明；原 P8 待验收项仍标为未完成，不把原冻结集修改成四卡后当作同一验收。

## 5. 进入 GPU 阶段的准备完成条件

1. 两个主候选及可选轻量备选的实际权重完整，revision/分片哈希核对；Llama 等门控模型有访问权才下载。
2. 主环境的 image digest、依赖和模型注册确认，启动命令按 GPU 数/dtype/EP或TP 分组分别冻结；必要 JIT 留 GPU smoke，不谎称全预编译。
3. 多候选运行器、真实 router hook、请求语料、统计守恒、超时、恢复和回收在 CPU 已演练。
4. profiler 短窗口与正式计时分开，rank/layer/step ID 明确，图形报告解析可以本地完成。
5. 预算按硬件和存储重新核对，GPU 探测/启动失败有提前结束规则，平台停止状态可核验。

实际准备环境是 Ubuntu 20.04 x86_64。已核对 vLLM 0.20.1 的发布 wheel 为 `manylinux_2_35_x86_64`，因此不能在该下载机上直接验收运行；软件缓存面向 Ubuntu 24.04 / CPython 3.12，后续 GPU 主机至少需满足 glibc 2.35。PyTorch 2.11.0 的默认 Linux 依赖使用 CUDA 13.0.2，必须单独核对目标 GPU 驱动，不能把它覆盖到原 A100 CUDA 12.8 测量环境。

网络卷位于 US-NE-1，同区列有 H100 SXM；这不保证将来的四卡容量，异区 A100 八卡实验仍需数据复制。软件 wheel 缓存、固定镜像 digest、源码归档、CLI 自检及 GPU 运行验收分别记录；不能把“下载好”写成“GPU 就绪”。
