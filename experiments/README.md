# Experiments

每个 experiment 都应该是一个自包含的 slice。

推荐目录结构：

```text
experiments/YYYY-MM-DD-short-name/
  README.md
  commands.sh
  env.md
  results.csv
  figures/
  raw/      # 默认被 git 忽略，除非手动 force add
```

规则：

- 一个 slice 只回答一个问题。
- 保存 exact commands 和 environment。
- raw logs 如果很大，就保存在本地或外部存储。
- 处理后的 tables、figures 和 written interpretation 可以提交到 Git。
- 每次只改变一个主要变量。

## 平台分层

- 4x4090 / Ada / PCIe-only：只作为 NCCL/PyTorch distributed、PCIe/NUMA topology 和 profiling 流程热身。
- A100 / SM80：作为 NCCL、PyTorch `all_to_all`、NVSHMEM legacy 对照平台；进入 NCCL Device API 实验前必须先通过 `probe-runtime`，当前接口要求 NCCL 2.28+。
- H100 / H800 / H20 / SM90：作为 DeepEP V2、MoE EP dispatch/combine 和 overlap 的主实验平台。

## 已完成的 A100 闭环

- `2026-08-26-a100-runtime-capabilities`：记录系统 NCCL 2.25.1 不具备 Device API 的负向门槛。
- `2026-08-26-a100-nccl-device-fused-reference`：隔离构建 NCCL 2.28.9，运行官方 Device API AllReduce，并验证 Edmir LSA 融合参考核的数值正确性。该 slice 不宣称性能收益。
- `2026-08-26-a100-nccl-device-cutlass-fusion`：将 LSA 指针注入 CUTLASS device kernel，验证单 kernel tensor-core 路径，并用 N-sweep 定位 direct-remote fusion 的收益边界与远端 A 重读问题。
