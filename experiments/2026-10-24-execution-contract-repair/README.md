# 执行合同修复交付

日期：2026-10-24。修改在当前源码工作区；未租机、未连接 GPU、未训练模型。旧实测包和此前审查报告保留。

## 已完成

1. **输入与工作区就绪。** 两个 backend 的 `copy_from_host()` 在所属设备上等待默认上传流完成后才返回，主机临时输入的生命周期与设备可见性都有明确边界。`CudaBackend.prepare()` 同样等待 split-K 工作区初始化完成。这些等待在主计时之外，不在每次稳态执行中增加 device-wide 同步。
2. **执行身份 v2。** 源码清单覆盖 `Op.resources()`、策略、Program 类型、codegen、uniform 适配器和两个 backend。身份另含执行图、实际 GEMM 维度、q/g、metadata 模式和算法摘要。源码变化会要求重新审查，图或证据不匹配会拒绝比较。v1 JSON 不自动升级；旧耗时不能直接贴到新身份。
3. **受限精简策略。** `uniform_immutable_peer_v1` 只接受未分块的标准 FP16 uniform fused 程序（2..8 ranks），采用目标 compute 队列顺序。Python 校验标准程序，C++ 再校验访问、独立输出和完整队列边。运行时要求全部矩阵完整上传，每轮全部完成后才能再次执行或复用；禁止可写原始指针入口。split-K workspace 逐操作独立，设备完成后才释放。默认仍为原策略，不自动进入候选选择。
4. **强 host 台账接通。** 使用现有采集器命名 `host`、`host_compiler_cublas`、`host_merged_cutlass`、`host_merged_cublaslt`。merged 描述自身的搬运/执行图和每卡 PM 行 GEMM，不把输入 manifest 的 P² 次 GEMM 当作实际执行。库内部 CTA 和流量保持未知；cuBLASLt 未记录实际每 rank 算法时不能接收耗时证据。共同测量条件与各候选执行身份分开，仍要求同输入/测量上下文、匹配区组、完整正确性检查和未插桩计时。

生成新机制采集包时，四种 host 都会携带各自台账。已冻结采集包不会被改写。精简策略必须显式生成新 manifest，并使用它自己的源码/图身份采集；不能改写旧包中的候选标签来运行。

## 本地验收

- **86 项 Python 测试通过**，含资源源码变异、图摘要篡改、旧证据失配、候选顺序、双向 host/fused 比较、producer/复用风险拒绝、包封存与超时恢复。记录：`python-tests-final.log`。
- **3 个 C++ CTest 目标通过**。编译真实 runtime 源码，CUDA/NCCL/cuBLAS 接口使用 CPU 测试替身；不模拟张量运算或设备耗时。记录：`local-check-02/2.log`。
- 新策略在 legacy/prepared × 正常/反向映射四种组合各执行 **1,000 轮**，每轮 164 次 execute/synchronize API 调用，未随迭代增长。上传完整性采用计数状态，稳态入口不再遍历外部缓冲区。
- 用旧冻结 header 编译新 runtime，旧策略 legacy/prepared 的 API 序列一致，每轮仍为 240 次；两种模式各运行 1,000 轮。记录：`legacy-api-test.log`。
- 每轮均保留 16 次 GEMM、28 次 event record、12 次 stream synchronize；精简策略的 wait 从 74 降到 36。工作区独立、重入拒绝、输入重复上传、释放前等待均有测试。
- 40,000 节点 / 159,992 条含重复边的图准备测试通过，本机该次约 39 ms。它只验证本地准备逻辑的可运行性，不代表 GPU 性能。
- GPU collector 用官方头文件完成 C++ 语法检查；本机没有执行 CUDA kernel 编译与 GPU 数值验证。
- 原四个优先形状和 M512 × 两种策略 × 14 个候选，共 **140 份静态台账**，全部保留待测结果。`candidate-accounts.json` 含完整材料，`candidate-summary.json` 含身份汇总。
- 原 GPU 压缩包 SHA-256 仍为 `0c28679a0a0052131c46ef86718f7e0f91cb743b19f0abcd031aae813f8f4856`。

## 使用

在仓库根目录：

```sh
python3 EDM-IR/tools/explain_execution.py \
  --m 512 --k 4096 --n 512 --environment planned-a100-p4 \
  --families host host_compiler_cublas host_merged_cutlass host_merged_cublaslt splitk2 splitk4 \
  --metadata-mode prepared --dependency-policy uniform_immutable_peer_v1 \
  --out /tmp/m512-execution-v2.json
```

输出路径必须不存在。`--dependency-policy` 仅作用于 fused；host 保持自身策略。这里只生成台账，不启动测量或选择快慢。

CPU runtime 检查入口为 `EDM-IR/tools/check_execution_contract.py`，需要新输出目录和本地官方 API 头文件。完整已执行命令在 `local-check-02/commands.json`；所有测试均有超时限制。

详细接口和新 manifest 的生成方式见 [EXECUTION_EVIDENCE.md](../../EDM-IR/docs/EXECUTION_EVIDENCE.md)。

## 尚未成立的结论

P=4 的依赖从 31 条降到 12 条，最长操作链从 11 缩为 4，这是静态执行结构变化。**不能换算成 GPU 加速比，不能宣称 M512/q4 或大 N 的误选已解决。** 并发增加后的争用、split-K 等待和数值正确性仍需在同机用新身份做小批 GPU 对照。此次没有添加拟合参数，也没有把旧标签用于证明新策略更快。
