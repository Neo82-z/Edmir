# Experiment：NCCL 4x4090 Baseline

## 问题

在 PCIe-only、4x4090、dual-NUMA 机器上，near-GPU pair、cross-NUMA pair 和全 4 GPU 的 NCCL collective baseline 有什么差异？

## 已知 Topology

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NODE    SYS     SYS     0-27,56-83      0
GPU1    NODE     X      SYS     SYS     0-27,56-83      0
GPU2    SYS     SYS      X      NODE    28-55,84-111    1
GPU3    SYS     SYS     NODE     X      28-55,84-111    1
```

解释：

- GPU0 / GPU1 在 NUMA node 0 内相对更近。
- GPU2 / GPU3 在 NUMA node 1 内相对更近。
- cross pair 需要经过 `SYS`，理论上更慢或更不稳定。
- 这个 topology 没有 NVLink。

## Collectives

至少运行：

- all_reduce
- all_gather
- reduce_scatter
- alltoall

## Device Cases

```text
0,1       NUMA 0 内的 near pair
2,3       NUMA 1 内的 near pair
0,2       cross-NUMA pair
1,3       cross-NUMA pair
0,1,2,3   全 4-GPU run
```

## Commands

```bash
# 在这里填写 exact commands。
```

## Results

| Collective | Devices | Size Range | Peak AlgBW | Peak BusBW | Notes |
|---|---|---:|---:|---:|---|

## Observations

1. 
2. 
3. 

## Next

- 对关键 runs 增加 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH`。
- 为一个 small-message case 和一个 large-message case 增加 Nsight Systems trace。
- 与 `NCCL_P2P_DISABLE=1` 结果对比。
