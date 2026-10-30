# 本轮参考源码

2026-10-21 核查。微信公众号正文 `https://mp.weixin.qq.com/s/X41mKH4Ds-VXUAnK6M8Eww` 返回验证页，未读到正文；以下按用户截图中的六个仓库逐一查阅 README 和相关源码，不推测文章未提供的论述。版本及文件哈希见 `sources.json`。外部代码只读，没有安装、执行或引入运行时依赖，也没有把其他硬件的耗时混入 A100 标签。

| 仓库 | 读哪处 | 对本项目的价值与边界 |
|---|---|---|
| [TileLang](https://github.com/tile-ai/tilelang/blob/d85a70a97218fec25856f8b5c969f167ad96dc1e/tilelang/autotuner/tuner.py) | 调优器的配置超时、缓存兼容性、正确性检查状态 | 候选搜索要有预算，跳过正确性检查不能标为通过。本轮固定三方案、拟合截止时间和进程超时；模型缓存包含训练行、标签与方法。没有运行 GPU autotuning。 |
| [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/csrc/jit_kernels/heuristics.hpp) | `HWDesc`、`make_gemm_candidate`、分阶段代价与候选搜索 | 最值得继续读的物理代价结构：把硬件描述、可执行性约束、waves、搬运与计算代价分开。Ascend 的 L0/L1、带宽和指令常数不能搬到 A100。这里的 M-grouped 也不能当作我们的 grouped-N。 |
| [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/utils/testing.py) | `bench` 的事件、warmup、barrier、完成回调边界 | 计时必须包含完整异步操作，区分 rank 本地事件耗时与完整算子 wall time。其 NPU 通信时间不是本项目 NCCL 基线。本轮未改 GPU 计时包。 |
| [TileKernels](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/testing/pytest/benchmark.py) | 正确性与 benchmark 标记、JSONL 结果、基线对照 | 参考测试与性能验收的分离、结果身份及回归记录。当前 README 的 CUDA 要求是 SM90/SM100，不能视作现成 SM80 基线。 |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/tests/kernelkit/bench.py) | profiler 范围与 kernel 事件提取 | 可参考性能归因方法；attention 内核与本任务不同，profiler kernel 时间不能替代现有完整算子标签。 |
| [DeepSelect](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/csrc/cuda_kernels/config.h) | TopK 配置及 README 的支持范围 | 是稀疏注意力/采样用 TopK 算子，不是通用决策学习框架。候选的 dtype、输出排序、返回值等语义影响成本；本轮不把它作为学习模型基线。 |

与学习方法直接相关的参考：

- [Laya 模型](https://github.com/NandhaKishorM/laya/blob/ffc142012af63d015fa19f05f29f9c5ff329a2cc/laya/common.py#L575)：共享候选评分；本项目改为小型数值网络，不使用文本编码器。
- [Laya 训练](https://github.com/NandhaKishorM/laya/blob/ffc142012af63d015fa19f05f29f9c5ff329a2cc/laya/train.py#L493)：明确决策训练损失及交叉熵对照。我们的已测候选有完整成本，直接对期望选择损失求导，不复用它的随机策略梯度。
- [Laya 顺序处理](https://github.com/NandhaKishorM/laya/blob/ffc142012af63d015fa19f05f29f9c5ff329a2cc/laya/common.py#L280)：提示我们检查候选重排。新评分器不输入候选位置，逐行共享参数；平局按明确规则处理。

这些参考为结构、工程方法和测量边界提供依据，不构成本预测器准确性的外部验证。新增训练损失、梯度和 NumPy 小网络是本项目实现，没有复制上述仓库的实现代码。
