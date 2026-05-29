# Experiment：NCCL 4x4090 Baseline

## 当前状态

**状态：暂缓 clean NCCL baseline。**

当前机器上已有 vLLM 服务常驻，占用了多张 GPU 和大量显存。此时直接跑 `nccl-tests` 得到的结果会混入线上服务干扰，不适合作为硬件 / NCCL 的干净 baseline。

本文件当前先记录：

- 4x4090 topology。
- 当前 GPU / container 占用情况。
- 为什么暂缓 NCCL baseline。
- 后续干净窗口中应该如何跑第一版 NCCL baseline。

原始终端输出保存在：[`raw/2026-05-29-precheck.log`](raw/2026-05-29-precheck.log)

## 原始问题

在 PCIe-only、4x4090、dual-NUMA 机器上，near-GPU pair、cross-NUMA pair 和全 4 GPU 的 NCCL collective baseline 有什么差异？

## 环境快照

采集时间：`Fri May 29 18:35:44 2026`

| 项目 | 值 |
|---|---|
| Driver | `580.126.18` |
| CUDA | `13.0` |
| GPU | 4x NVIDIA GeForce RTX 4090 |
| 架构 | Ada Lovelace |
| 单卡显存 | `24564 MiB` |
| 当前结论 | GPU0 / GPU2 / GPU3 被 vLLM 明显占用，GPU1 也有两个 Python 进程 |

## 当前 GPU 占用

| GPU | Memory Used | GPU Util | 主要进程 | 备注 |
|---|---:|---:|---|---|
| GPU0 | `21429 / 24564 MiB` | `0%` | `VLLM::EngineCore` x3 | embedding 服务占用 GPU0 |
| GPU1 | `3428 / 24564 MiB` | `0%` | `/usr/bin/python3` x2 | 相对空闲，但不是 clean |
| GPU2 | `23703 / 24564 MiB` | `0%` | `VLLM::Worker_TP0` | Qwen3-32B-FP8 TP worker |
| GPU3 | `23703 / 24564 MiB` | `0%` | `VLLM::Worker_TP1` | Qwen3-32B-FP8 TP worker |

进程细节：

| GPU | PID | Process | GPU Memory |
|---|---:|---|---:|
| 0 | `297682` | `VLLM::EngineCore` | `9756 MiB` |
| 0 | `3332855` | `VLLM::EngineCore` | `3732 MiB` |
| 0 | `3553571` | `VLLM::EngineCore` | `7922 MiB` |
| 1 | `1328209` | `/usr/bin/python3` | `1706 MiB` |
| 1 | `3632600` | `/usr/bin/python3` | `1706 MiB` |
| 2 | `1217466` | `VLLM::Worker_TP0` | `23694 MiB` |
| 3 | `1217467` | `VLLM::Worker_TP1` | `23694 MiB` |

## vLLM 容器占用

### `be07844ccc06`：Qwen3-32B-FP8

Device request：

```json
[{"Driver":"","Count":0,"DeviceIDs":["2","3"],"Capabilities":[["gpu"]],"Options":{}}]
```

启动命令：

```bash
vllm serve /data/models/Qwen3-32B-FP8 \
  --served-model-name qwen3-32b-fp8 \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

结论：该容器占用 `GPU2,GPU3`，并且显存接近打满，不适合同时参与 NCCL baseline。

### `3471fe9d72fc`：Qwen3-Embedding-4B

Device request：

```json
[{"Driver":"","Count":0,"DeviceIDs":["0"],"Capabilities":[["gpu"]],"Options":{}}]
```

启动命令：

```bash
vllm serve --model /data/models/Qwen3-Embedding-4B \
  --served_model_name Qwen3-Embedding-4B \
  --trust-remote-code \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.4 \
  --max-num-batched-tokens 4096
```

结论：该容器占用 `GPU0`。

## 已知 Topology

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    SYS     SYS     0-27,56-83      0               N/A
GPU1    NODE     X      SYS     SYS     0-27,56-83      0               N/A
GPU2    SYS     SYS      X      NODE    28-55,84-111    1               N/A
GPU3    SYS     SYS     NODE     X      28-55,84-111    1               N/A
```

解释：

- GPU0 / GPU1 在 NUMA node 0 内相对更近。
- GPU2 / GPU3 在 NUMA node 1 内相对更近。
- cross pair 需要经过 `SYS`，理论上更慢或更不稳定。
- 这个 topology 没有 NVLink，是 PCIe-only commodity multi-GPU baseline。

## GPU1 单卡硬件快照

由于 GPU1 目前相对空闲，它适合先做 single-GPU Ada / tinygrad / CUDA 预实验，但不适合作为 clean communication baseline。

| 字段 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Architecture | Ada Lovelace |
| Bus Id | `00000000:65:00.0` |
| FB Memory | `24564 MiB` total, `3428 MiB` used, `20654 MiB` free |
| BAR1 Memory | `256 MiB` total, `6 MiB` used |
| PCIe Max | Gen4 x16 |
| PCIe Current | Gen1 x16 at idle |
| Idle Clocks | Graphics / SM `210 MHz`, Memory `405 MHz` |
| Max Clocks | Graphics / SM `3105 MHz`, Memory `10501 MHz` |
| Power Limit | `450 W` |
| Idle Power | around `10-11 W` |
| Compute Mode | Default |
| MIG | N/A |

注意：`PCIe Current = Gen1` 是 idle 状态下的读数，不能直接当作通信测试时的有效链路速率。后续跑 workload 时需要重新采集或观察是否升到 Gen4。

## 当前决策

本次不把当前机器状态下的 `nccl-tests` 结果作为正式 baseline。

原因：

1. GPU2 / GPU3 被 Qwen3-32B-FP8 TP=2 服务占满。
2. GPU0 被 Qwen3-Embedding-4B 服务占用。
3. GPU1 仍有两个 Python 进程，不是完全 clean。
4. NCCL baseline 需要尽量排除 vLLM 对显存、HBM、PCIe、CUDA context 和调度的干扰。

短期动作：

- 暂缓 4-GPU NCCL baseline。
- 可先在 GPU1 上做 single-GPU Ada / tinygrad 预实验。
- 等申请到 vLLM 暂停窗口后，再跑 clean NCCL baseline。

## 后续 Clean NCCL Baseline 计划

### 需要先确认

```bash
nvidia-smi
nvidia-smi topo -m
```

目标状态：4 张 GPU 尽量无 vLLM / Python 常驻进程，显存占用接近 idle。

### 建议运行的 collectives

至少运行：

- all_reduce
- all_gather
- reduce_scatter
- alltoall

### Device Cases

```text
0,1       NUMA 0 内的 near pair
2,3       NUMA 1 内的 near pair
0,2       cross-NUMA pair
1,3       cross-NUMA pair
0,1,2,3   全 4-GPU run
```

### Commands

```bash
# near pair: NUMA 0
CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2

# near pair: NUMA 1
CUDA_VISIBLE_DEVICES=2,3 ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2

# cross-NUMA pair
CUDA_VISIBLE_DEVICES=0,2 ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2
CUDA_VISIBLE_DEVICES=1,3 ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 2

# full 4-GPU baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/all_gather_perf -b 8 -e 512M -f 2 -g 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/reduce_scatter_perf -b 8 -e 512M -f 2 -g 4
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/alltoall_perf -b 8 -e 512M -f 2 -g 4
```

### Debug / Trace

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./build/all_reduce_perf -b 1M -e 64M -f 4 -g 4
```

后续再补：

- small-message case 的 Nsight Systems trace。
- large-message case 的 Nsight Systems trace。
- `NCCL_P2P_DISABLE=1` 对照实验。

## Results

当前暂无 clean NCCL baseline 结果。

| Collective | Devices | Size Range | Peak AlgBW | Peak BusBW | Notes |
|---|---|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | 等 vLLM 暂停窗口后补 |

## Observations

1. 当前机器不是 clean benchmark 环境，不能直接产出正式 NCCL baseline。
2. 4x4090 topology 是 dual-NUMA、PCIe-only；后续重点对比 `NODE` pair 和 `SYS` pair。
3. GPU1 可以先用于 single-GPU Ada / tinygrad 预实验，但不能回答 GPU-GPU communication 问题。
4. Qwen3-32B-FP8 服务使用 `GPU2,GPU3`，`--tensor-parallel-size 2`，是后续 vLLM TP trace 的真实 workload 候选。

## Next

- [ ] 申请 vLLM 暂停窗口，获取 clean 4-GPU 环境。
- [ ] 跑完整 NCCL baseline。
- [ ] 记录 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` 输出。
- [ ] 为 small-message / large-message 各采一次 Nsight Systems trace。
- [ ] 如继续研究单卡 4090 / Ada / tinygrad，另开一个 experiment slice，避免和 NCCL baseline 混在一起。
