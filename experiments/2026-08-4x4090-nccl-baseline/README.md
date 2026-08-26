# 4xRTX 4090 NCCL baseline and compiler calibration

Status: **ready to run; no result has been collected**.

这个实验的目的不是把 4090 数据冒充 A800/H100 论文结果，而是利用当前空闲机器完成两件事：

1. 查清 4 张卡的实际 PCIe/P2P 排布以及 NCCL 是否经过 host staging；
2. 给 Edmir compiler 提供逐 GPU pair、逐消息尺寸的通信成本曲线。

任何关于互联的结论都必须来自归档的 `nvidia-smi topo`、CUDA P2P 和 NCCL 日志，不能只根据“RTX 4090”型号判断。

## 为什么只测 18 组

| 层次 | 排布 | 操作 | 模式 | 数量 | 用途 |
|---|---|---|---|---:|---|
| pairwise | 01, 02, 03, 12, 13, 23 | `sendrecv` | default | 6 | compiler 的 `(src,dst,size)` 主成本曲线 |
| pairwise control | 同上 | `sendrecv` | `NCCL_P2P_DISABLE=1` | 6 | 判断默认路径是否真的获得 P2P 收益 |
| collective | 0123 | `alltoall`, `all_gather`, `all_reduce` | default | 3 | 4 卡 NCCL 基线与整体健康检查 |
| collective control | 0123 | 同上 | `NCCL_P2P_DISABLE=1` | 3 | host-staged 对照 |

总计 18 组。当前不测 3 卡、不扫 `NCCL_ALGO`、不测多节点，因为这些变量不回答静态 4 卡 `All2All -> GEMM` compiler 的当前问题。

主结果是 default pairwise `sendrecv` 和 default 4-card `alltoall`。AllReduce/AllGather 只是辅助诊断，不能替代 AllToAll 基线。

## 运行

在服务器上的 Edmir 仓库根目录执行：

```bash
RESULT_ROOT=/root/edmir-4090-baseline-$(date +%Y%m%d-%H%M%S)

bash experiments/2026-08-4x4090-nccl-baseline/commands/00_capture_environment.sh \
  "$RESULT_ROOT"

NCCL_TESTS_DIR=/root/edmir-bench/nccl-tests \
bash experiments/2026-08-4x4090-nccl-baseline/commands/10_run_nccl_baselines.sh \
  "$RESULT_ROOT"
```

默认 sweep 为 `1K -> 512M`、倍率 2、5 次 warmup、20 次 iteration。可以通过环境变量覆盖：

```bash
MIN_BYTES=1K MAX_BYTES=256M WARMUP_ITERS=5 TEST_ITERS=20 \
NCCL_TESTS_DIR=/path/to/nccl-tests \
bash experiments/2026-08-4x4090-nccl-baseline/commands/10_run_nccl_baselines.sh \
  "$RESULT_ROOT"
```

脚本不会因为单个测试失败而丢掉其他数据；每项退出状态写入 `manifest.tsv`。失败本身也是结果，特别是 P2P 被平台或驱动禁用时。

## 将 pairwise 数据接入 compiler

`fit-nccl` 只读取命名为 `pair_<src>_<dst>_default_sendrecv.txt` 的默认模式日志。它不会用禁用 P2P 的对照曲线污染 compiler profile，也不会把 collective bus bandwidth 当成单个 transfer 的成本。

```bash
cd EDM-IR
python -m edmir fit-nccl \
  --base-profile examples/synthetic_profile.json \
  --logs "$RESULT_ROOT/nccl" \
  --out "$RESULT_ROOT/4090-pairwise-profile.json"
```

这个输出只替换 pairwise communication 部分。GEMM throughput、launch 和 sync 参数仍继承 base profile；在完成 GEMM calibration 前，它只能用于检查通信排布和 compiler 数据通路，不能作为最终策略选择结论。

## 阅读顺序

1. `environment/nvidia-smi-topo-m.txt`：先识别 PIX/PXB/PHB/SYS 和 CPU affinity；
2. `environment/nvidia-smi-topo-p2p-p.txt` 与 `p2pBandwidthLatencyTest.txt`：确认 CUDA peer access；
3. 六组 default pairwise：比较相同 size 下的 latency/algbw；
4. 六组 no-P2P 对照：判断默认路径的机制差异；
5. `all4_default_alltoall.txt`：形成当前 compiler workload 的 NCCL 顺序基线；
6. AllGather/AllReduce：只用于判断 NCCL 环境是否存在普遍退化。

## 完成条件

- 四张卡、PCI bus ID、CPU/NUMA affinity 和软件版本已归档；
- 6 个 pair 均有 default 日志或明确失败记录；
- 4-card AllToAll 有 default 日志或明确失败记录；
- `manifest.tsv` 没有未解释的非零状态；
- `fit-nccl` 生成 12 条有向 pair curve，并能被 `edmir simulate` 使用。

