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
- A100 / SM80：作为 NCCL、PyTorch `all_to_all`、NVSHMEM legacy 对照平台。
- H100 / H800 / H20 / SM90：作为 DeepEP V2、MoE EP dispatch/combine 和 overlap 的主实验平台。
