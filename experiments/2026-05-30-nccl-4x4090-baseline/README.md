# Experiment：NCCL 4x4090 Baseline
root@ubuntuserver:/ops/nccl-tests/nccl# nvidia-smi
Fri May 29 18:35:44 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.18             Driver Version: 580.126.18     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:4B:00.0 Off |                  Off |
| 47%   35C    P8             13W /  450W |   21429MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off |   00000000:65:00.0 Off |                  Off |
| 31%   41C    P8             10W /  450W |    3428MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090        Off |   00000000:B1:00.0 Off |                  Off |
| 44%   43C    P8             24W /  450W |   23703MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 4090        Off |   00000000:E3:00.0 Off |                  Off |
| 31%   46C    P8             19W /  450W |   23703MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          297682      C   VLLM::EngineCore                       9756MiB |
|    0   N/A  N/A         3332855      C   VLLM::EngineCore                       3732MiB |
|    0   N/A  N/A         3553571      C   VLLM::EngineCore                       7922MiB |
|    1   N/A  N/A         1328209      C   /usr/bin/python3                       1706MiB |
|    1   N/A  N/A         3632600      C   /usr/bin/python3                       1706MiB |
|    2   N/A  N/A         1217466      C   VLLM::Worker_TP0                      23694MiB |
|    3   N/A  N/A         1217467      C   VLLM::Worker_TP1                      23694MiB |
+-----------------------------------------------------------------------------------------+

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


利用docker查看容器占用的显卡：

nvidia-smi
docker inspect be07844ccc06 --format '{{json .HostConfig.DeviceRequests}}'
docker inspect 3471fe9d72fc --format '{{json .HostConfig.DeviceRequests}}'

查看启动命令：

docker inspect be07844ccc06 --format '{{.Path}} {{range .Args}}{{.}} {{end}}'
docker inspect 3471fe9d72fc --format '{{.Path}} {{range .Args}}{{.}} {{end}}' ##
重点看：


** CUDA_VISIBLE_DEVICES、--tensor-parallel-size、--gpu-memory-utilization**

后续重启vllm：

root@ubuntuserver:/ops/nccl-tests/nccl# docker inspect be07844ccc06 --format '{{json .HostConfig.DeviceRequests}}'
[{"Driver":"","Count":0,"DeviceIDs":["2","3"],"Capabilities":[["gpu"]],"Options":{}}]
root@ubuntuserver:/ops/nccl-tests/nccl# docker inspect 3471fe9d72fc --format '{{json .HostConfig.DeviceRequests}}'
[{"Driver":"","Count":0,"DeviceIDs":["0"],"Capabilities":[["gpu"]],"Options":{}}]
<cker inspect be07844ccc06 --format '{{.Path}} {{range .Args}}{{.}} {{end}}'
vllm serve /data/models/Qwen3-32B-FP8 --served-model-name qwen3-32b-fp8 --max-model-len 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes
<cker inspect 3471fe9d72fc --format '{{.Path}} {{range .Args}}{{.}} {{end}}'
vllm serve --model /data/models/Qwen3-Embedding-4B --served_model_name Qwen3-Embedding-4B --trust-remote-code --dtype half --max-model-len 2048 --gpu-memory-utilization 0.4 --max-num-batched-tokens 4096





