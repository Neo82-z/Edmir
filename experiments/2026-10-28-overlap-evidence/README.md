# 从公开通信实现和性能记录检查误选

2026-10-28。本轮先读性能论文，再核对代码、测量方法和公开数据。新增的是**选择差距诊断与 overlap 证据检查**，没有重训、替换默认评分器或启动 GPU。冻结六形状没有进入本轮分析。M512 仍未解决。

## 先看选择，不能让 MAPE 掩盖失败

重新审计上一轮固定实验结果，未调整权重。四卡 41 个形状、346 个正确候选记录。下表的“误选”覆盖全部 provisional choice，包括支持检查未通过的预测；不能靠拒绝选择降低错误数。

| 上轮模型 | 误选形状 | host/fused 方向错误 | 平均选择损失 | 最坏选择损失 | 耗时 MAPE |
|---|---:|---:|---:|---:|---:|
| 原损失/原底座 + AdamW | 10/41 | 1 | 0.544% | 7.357% | 4.590% |
| 新损失/原底座 + AdamW | 10/41 | 1 | 0.492% | 7.357% | 2.941% |
| 新损失/原底座 + Muon | 10/41 | 1 | 0.503% | 7.357% | 2.875% |
| 新损失/工作量底座 + Muon | 7/41 | 1 | 0.433% | 7.357% | 2.990% |

这些不是本轮改进成绩。支持覆盖均为 28/41；简单旧底座的平均选择损失仍更低（0.361%）。八卡已测子集上零误选，但支持覆盖为零，不能据此宣称八卡验收通过。当前数据已多次用于设计，只能作为开发回放。

最少误选的上轮分支仍有七个错误：

| `[P,M,K,N]` | 预测候选 → 实测最快 | 实际损失 |
|---|---|---:|
| `[4,64,512,128]` | splitk2 → direct | 2.632%，8.0 μs |
| `[4,64,512,2048]` | splitk2 → direct | 0.584%，1.9 μs |
| `[4,512,4096,512]` | host → q2 | **7.357%，74.0 μs** |
| `[4,64,512,1024]` | splitk2 → direct | 0.577%，1.8 μs |
| `[4,256,4096,512]` | splitk4 → q2 | 0.938%，6.4 μs |
| `[4,64,1024,128]` | q2g2 → q2 | **4.082%，13.8 μs** |
| `[4,128,512,512]` | q2 → direct | 1.580%，4.9 μs |

没有重复测量的不确定性证据，不能把较小差距直接归为噪声。也不应该把它们与 M512 的 74 μs 损失混为同一机制。

M512 实测 host/q2 为 1079.90/1005.90 μs，差值 `host−q2=+74.00 μs`；该分支预测 995.65/1071.84 μs，差值 **−76.18 μs**。两个候选的相对误差把差值推错了 150.18 μs。

令 `e_h=log(pred_h/actual_h)`、`e_f=log(pred_f/actual_f)`，则：

```
预测 log(host/fused) − 实测 log(host/fused) = e_h − e_f
```

本例共同 log 偏差 `(e_h+e_f)/2≈−0.00887` 很小，差分偏差 `e_h−e_f≈−0.14471` 却足以翻转选择。统一缩放两者耗时不能修复顺序。这是误选的数学诊断，尚不能确定底层原因是通信、计算还是启动等待。

## 经过核对的外部依据

筛选标准是原作者说明、可检查实现和可辨认的实验条件，而非仅看机构。下面区分论文、官方工程说明和公开数据；没有把博客中的速度宣传当成复现结果。

| 优先资料 | 本轮实际核对 | 直接借鉴 | 使用边界 |
|---|---|---|---|
| [Demystifying NCCL v3](https://arxiv.org/html/2507.04786v3)，含 NVIDIA 作者 | 论文的启动方式、协议、channel 和分块机制；对应 [NCCL](https://github.com/NVIDIA/nccl) / [nccl-tests](https://github.com/NVIDIA/nccl-tests) | 通信不只是字节/峰值带宽；启动组织、消息粒度和协议影响延迟/吞吐切换 | v3 仍以 NCCL 2.19.1 为分析对象，不能把其常数移到当前版本 |
| [DeepEP 固定版本](https://github.com/deepseek-ai/DeepEP/tree/f6aa38fe06cd4a0489e7730365d4e0114970a2f3) 与 [公开 profile](https://github.com/deepseek-ai/profile-data/tree/449602428a1b023acb8a505d4f34fef536535db6) | `get_theoretical_num_sms`、计时函数；下载并解析 prefill/decode 原始时间线 | 分别算读、写、NVLink、RDMA 流量；资源需求受瓶颈链路与路由影响 | 当前 SM 估计假设均匀路由，部分选项尚未建模；默认常数与安全余量不可直接移植 A100 |
| [T-CCL](https://arxiv.org/html/2610.07098v1) 与 [artifact](https://github.com/ooverlap/ooverlap/tree/9abd9cf93286fc8ca5a0a041db6234c4e86e1635) | 论文实验与 artifact 的 `evalution/run_flashoverlap_sm90.sh` | 在受限 CTA 预算下测通信曲线，配合计算曲线；固定调度比较后端，保留 overlap 负收益 | 作者非 NVIDIA；论文标注 SC'26 Workshops。硬件 H100/GH200，TMA 路径不适用 A100；未实现本文所需 All-to-All，不能直接当基线 |
| [Alibaba InferSim](https://github.com/alibaba/InferSim/tree/d12ec47f83d5896efd77007ad5d5f2ad044c2876) | 通信估算、MFU 查询、三份公开 GEMM/grouped GEMM 表 | 经验服务速率需要按硬件和工作负载查询，MoE 形状含专家/路由信息 | 当前公开表不提供我们的 A100 host/q2/q4 标签；CSV 没有完整精度/计时/版本合同。技术报告下载失败，本轮结论来自 README/代码/数据 |

另外核对三处工程材料：

- [NVIDIA Green Contexts，2026-10-06](https://developer.nvidia.com/blog/control-how-your-gpu-shares-work-with-green-contexts)：资源配额和 stream priority 不是同一回事。示例是 Blackwell 上的关键/批量 kernel，不是我们的 GEMM–通信实验，不能套用其加速比。
- [DeepGEMM MegaMoE 测试](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/tests/test_mega_moe.py)：融合路径与旧基线分开计时，旧基线依赖缺失时会跳过。必须记录基线是否真正执行及计时范围，不能由融合测试成功推断基线比较成立。公开路径面向 SM90/SM100，非 A100。
- [GLM 工程记录](https://z.ai/blog/glm-built-its-inference-infrastructure)及已合并 [FLA PR #1180](https://github.com/fla-org/flash-linear-attention/pull/1180)：PR 是 opt-in TF32x3 精度修复并增加测试，其说明并未宣称性能提升。它支持我们要求精度合同与正确性验证，不能用来佐证博客中其他 kernel 的速度数字。

NCCL 的 [Zero-CTA 官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#zero-cta-optimization)还说明了版本、buffer、操作和路径条件。我们源码中的 window 注册本身不能证明启用了 Zero-CTA；当前 `cuda_backend.cpp` 没有设置 `CTAPolicy=ZERO`。即使不占 CTA，也不能推断不占显存或链路带宽。

## 用公开时间线检验一个容易错误的特征

已下载 DeepSeek 固定版本的 `prefill.json` 与 `decode.json`，共 104,839 条事件。按同一 GPU/context 合并 kernel 区间；EP internode kernel 标记为通信，其他非 NCCL kernel 标记为计算/辅助，NCCL kernel 单独排除。保留逐 kernel 分类明细供复核。

| 时间线 | EP 通信 kernel 区间并集 | 与计算/辅助 kernel 交集 | 交集/通信 kernel 并集 |
|---|---:|---:|---:|
| prefill | 1,782,646 μs | 1,624,782 μs | 91.144% |
| decode | 17,059 μs | 1 μs | 0.00586% |

这是捕获窗口内的区间统计，不是每请求耗时、SM 占用或隐藏通信比例。官方说明 decode 发出 RDMA 消息后释放 SM，网络可以继续传输；因此 **GPU kernel 不同时运行，也可能存在网络与计算重叠**。两份时间线不能单独测出 overlap 加速比，缺少相同工作的孤立运行对照与传输完成证据。

版本同样关键：这些 V3/R1 profile 来自 2025 年；本轮读取的 DeepEP v2.5 明确说明 EP 需要 SM，不支持 zero-SM EP。不能把旧 profile 的解释套给新 API。profile 使用 H800、prefill EP32/decode EP128、模拟完全均衡路由；只能验证分析方法和机制区分，不能作为 A100 四/八卡训练标签。

## 由证据约束下一步模型结构

对于**相同工作、相同实现、已测出完整阶段结束时刻**的两阶段对照，设孤立计算/通信时间为 `C0,R0`，并发时为 `C1,R1`，完整阶段区间交集为 `O`，区间并集以外的其他耗时为 `G0,G1`：

```
Tsolo = C0 + R0 + G0
Tjoint = C1 + R1 − O + G1
节省时间 = O − [(C1−C0) + (R1−R0)] − (G1−G0)
```

这只是满足条件的计数恒等式，不能用测量后得到的 O 预测同一次执行，也不能把 issuing kernel 的区间当作 RDMA 完整阶段。Opaque fused kernel 未暴露阶段边界时，不强行拆出 C/R。真实调度有多个节点，继续使用现有 `conditional_dependency_span`，补齐相关串行边后才能讨论总时间；不再建立第二套图模型。

模型修改收敛为一条路线：**执行身份 → 有证据的服务曲线及并发竞争 → 依赖传播 → 候选耗时差与选择**。

1. 执行身份先分清 FP16/BF16、kernel/NCCL 版本、P/拓扑、host/fused 路径和资源策略。缺失身份不套用外部系数。现有合同继续使用。
2. 计算服务时间按实际 tile、尾部与资源需求建模；通信服务时间按消息粒度、路径与资源预算建模。T-CCL 提供受限资源曲线的实验方式，DeepEP 提供读/写/链路需求分开算的实现例子。当前本地总耗时数据没有唯一确定这些曲线。
3. 共享学习项若继续使用，只修正有组件数据支持的服务项或候选差分；候选分数仍来自同一个耗时估计。不能给 M512 加专属偏置，也不能在相同数据上反复增大网络直到排对。共同偏差和差分偏差分别检查。
4. 验收优先看完整候选集误选数、host/fused 双向错误、平均/最坏 regret，同时报告覆盖率；MAPE、CE/KL 是辅助项。近似平局仍保留原始误选，只有独立重复测量支持后才额外报告不确定性口径。

这支持“现有表达没有区分某些执行机制”的研究方向，但**尚未证实 M512 就是资源竞争引起，也没有证明参数量不足**。本轮没有凭外部论文为任意拟合项补造标签。

## 接下来的最小可证伪检查

优先 M512 的 host/q2；然后 `[4,64,1024,128]` 的 q2/q2g2；再检查 direct/split-K 的小差距。不要立即扩大形状网格或训练轮数。

| 竞争解释 | 区分它需要什么 | 否定条件 |
|---|---|---|
| host 独立服务时间估错 | 同一执行身份的 GEMM-only、通信-only、launch/等待范围 | 分项均对而联合执行仍偏差，不能继续归咎独立服务项 |
| 并发竞争或阶段就绪表达缺失 | 相同工作孤立/联合时间，完整传输完成点，资源/消息配置 | 联合服务没有变慢且依赖图解释不了差距，应检查其他项 |
| grouped-N 资源或 host API 工作变化 | g1/g2 的寄存器/共享内存、实际配置 API 次数和计时边界 | 仅增加 ceil-wave 不能解释当前所有网格的乐观容量下界均为一波 |
| 旧计时/执行身份混合 | 原始日志、构建哈希、计时器范围、正确性与精度 | 身份一致仍稳定复现错误，不能归为脚本差异 |

先检查已有采集资料能提供哪些证据，缺项再补采；不把 H800/H100 表混入 A100 标签。`NCCL_MIN/MAX_CTAS` 或资源划分只在路径实际支持时作为受控实验，不因为设置了环境变量就认定生效。采样和 profiler 分开，最终 wall-time 比较保持相同数值合同。冻结六形状留待方案确定后的独立验收。

## 代码、数据与复现

- `EDM-IR/edmir/compiler/overlap_evidence.py`：差分误差、区间统计、匹配工作/精度/身份及完整阶段结束的证据检查。未知的网络隐藏时间、SM 占用和因果加速比输出 null。
- `EDM-IR/tools/review_overlap_evidence.py`：公开 profile 和已有模型结果审计；检查 22 个受保护文件前后哈希，冻结形状不进入数据集。
- [decision-margin-audit.json](decision-margin-audit.json)、[public-trace-analysis.json](public-trace-analysis.json)、[review.json](review.json)：可机器读取的失败与证据。
- [source-pins.json](source-pins.json)、[source-catalog.json](source-catalog.json)、[downloads.json](downloads.json)：版本、链接、下载哈希与证据范围。外部代码仅阅读，未安装运行；代码许可证保存在对应目录。profile 按官方公开材料保留来源，本记录不为其添加许可证。

在仓库根目录执行，无需 GPU 或 PyTorch：

```sh
python3 EDM-IR/tests/overlap_evidence_test.py
python3 EDM-IR/tools/review_overlap_evidence.py
```

9 项逻辑测试覆盖候选差分、公共偏差、重叠负收益、异构/精度错误、异步完成证据、不同 GPU 区间隔离、排列不变性与有界遍历。通过测试只说明诊断逻辑符合合同，不说明预测器已经选对。公开时间线分析约 0.2 秒；每份最多 20 万事件，超出时明确失败，避免无界扫描。
