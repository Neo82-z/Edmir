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

nvidia-smi -q -i 1

==============NVSMI LOG==============

Timestamp                                              : Fri May 29 18:45:46 2026
Driver Version                                         : 580.126.18
CUDA Version                                           : 13.0

Attached GPUs                                          : 4
GPU 00000000:65:00.0
    Product Name                                       : NVIDIA GeForce RTX 4090
    Product Brand                                      : GeForce
    Product Architecture                               : Ada Lovelace
    Display Mode                                       : Requested functionality has been deprecated
    Display Attached                                   : No
    Display Active                                     : Disabled
    Persistence Mode                                   : Disabled
    Addressing Mode                                    : None
    MIG Mode
        Current                                        : N/A
        Pending                                        : N/A
    Accounting Mode                                    : Disabled
    Accounting Mode Buffer Size                        : 4000
    Driver Model
        Current                                        : N/A
        Pending                                        : N/A
    Serial Number                                      : N/A
    GPU UUID                                           : GPU-905f4679-3bbe-1073-d04f-c571527e7991
    GPU PDI                                            : 0x1747ff2ac279a52f
    Minor Number                                       : 1
    VBIOS Version                                      : 95.02.3C.00.02
    MultiGPU Board                                     : No
    Board ID                                           : 0x6500
    Board Part Number                                  : N/A
    GPU Part Number                                    : 2684-301-A1
    FRU Part Number                                    : N/A
    Platform Info
        Chassis Serial Number                          : N/A
        Slot Number                                    : N/A
        Tray Index                                     : N/A
        Host ID                                        : N/A
        Peer Type                                      : N/A
        Module Id                                      : 1
        GPU Fabric GUID                                : N/A
    Inforom Version
        Image Version                                  : G002.0000.00.03
        OEM Object                                     : 2.0
        ECC Object                                     : 6.16
        Power Management Object                        : N/A
    Inforom BBX Object Flush
        Latest Timestamp                               : N/A
        Latest Duration                                : N/A
    GPU Operation Mode
        Current                                        : N/A
        Pending                                        : N/A
    GPU C2C Mode                                       : N/A
    GPU Virtualization Mode
        Virtualization Mode                            : None
        Host VGPU Mode                                 : N/A
        vGPU Heterogeneous Mode                        : N/A
    GPU Recovery Action                                : None
    GSP Firmware Version                               : 580.126.18
    IBMNPU
        Relaxed Ordering Mode                          : N/A
    PCI
        Bus                                            : 0x65
        Device                                         : 0x00
        Domain                                         : 0x0000
        Base Classcode                                 : 0x3
        Sub Classcode                                  : 0x0
        Device Id                                      : 0x268410DE
        Bus Id                                         : 00000000:65:00.0
        Sub System Id                                  : 0x16F310DE
        GPU Link Info
            PCIe Generation
                Max                                    : 4
                Current                                : 1
                Device Current                         : 1
                Device Max                             : 4
                Host Max                               : 4
            Link Width
                Max                                    : 16x
                Current                                : 16x
        Bridge Chip
            Type                                       : N/A
            Firmware                                   : N/A
        Replays Since Reset                            : 0
        Replay Number Rollovers                        : 0
        Tx Throughput                                  : 488 KB/s
        Rx Throughput                                  : 390 KB/s
        Atomic Caps Outbound                           : N/A
        Atomic Caps Inbound                            : N/A
    Fan Speed                                          : 31 %
    Performance State                                  : P8
    Clocks Event Reasons
        Idle                                           : Active
        Applications Clocks Setting                    : Not Active
        SW Power Cap                                   : Not Active
        HW Slowdown                                    : Not Active
            HW Thermal Slowdown                        : Not Active
            HW Power Brake Slowdown                    : Not Active
        Sync Boost                                     : Not Active
        SW Thermal Slowdown                            : Not Active
        Display Clock Setting                          : Not Active
    Clocks Event Reasons Counters
        SW Power Capping                               : 1126906464 us
        Sync Boost                                     : 0 us
        SW Thermal Slowdown                            : 333448846 us
        HW Thermal Slowdown                            : 0 us
        HW Power Braking                               : 0 us
    Sparse Operation Mode                              : N/A
    FB Memory Usage
        Total                                          : 24564 MiB
        Reserved                                       : 484 MiB
        Used                                           : 3428 MiB
        Free                                           : 20654 MiB
    BAR1 Memory Usage
        Total                                          : 256 MiB
        Used                                           : 6 MiB
        Free                                           : 250 MiB
    Conf Compute Protected Memory Usage
        Total                                          : 0 MiB
        Used                                           : 0 MiB
        Free                                           : 0 MiB
    Compute Mode                                       : Default
    Utilization
        GPU                                            : 0 %
        Memory                                         : 0 %
        Encoder                                        : 0 %
        Decoder                                        : 0 %
        JPEG                                           : 0 %
        OFA                                            : 0 %
    Encoder Stats
        Active Sessions                                : 0
        Average FPS                                    : 0
        Average Latency                                : 0
    FBC Stats
        Active Sessions                                : 0
        Average FPS                                    : 0
        Average Latency                                : 0
    DRAM Encryption Mode
        Current                                        : N/A
        Pending                                        : N/A
    ECC Mode
        Current                                        : Disabled
        Pending                                        : Disabled
    ECC Errors
        Volatile
            SRAM Correctable                           : N/A
            SRAM Uncorrectable Parity                  : N/A
            SRAM Uncorrectable SEC-DED                 : N/A
            DRAM Correctable                           : N/A
            DRAM Uncorrectable                         : N/A
        Aggregate
            SRAM Correctable                           : N/A
            SRAM Uncorrectable Parity                  : N/A
            SRAM Uncorrectable SEC-DED                 : N/A
            DRAM Correctable                           : N/A
            DRAM Uncorrectable                         : N/A
            SRAM Threshold Exceeded                    : N/A
        Aggregate Uncorrectable SRAM Sources
            SRAM L2                                    : N/A
            SRAM SM                                    : N/A
            SRAM Microcontroller                       : N/A
            SRAM PCIE                                  : N/A
            SRAM Other                                 : N/A
        Channel Repair Pending                         : No
        TPC Repair Pending                             : No
    Retired Pages
        Single Bit ECC                                 : N/A
        Double Bit ECC                                 : N/A
        Pending Page Blacklist                         : N/A
    Remapped Rows
        Correctable Error                              : 0
        Uncorrectable Error                            : 0
        Pending                                        : No
        Remapping Failure Occurred                     : No
        Bank Remap Availability Histogram
            Max                                        : 192 bank(s)
            High                                       : 0 bank(s)
            Partial                                    : 0 bank(s)
            Low                                        : 0 bank(s)
            None                                       : 0 bank(s)
    Temperature
        GPU Current Temp                               : 41 C
        GPU T.Limit Temp                               : 43 C
        GPU Shutdown T.Limit Temp                      : -7 C
        GPU Slowdown T.Limit Temp                      : -2 C
        GPU Max Operating T.Limit Temp                 : 0 C
        GPU Target Temperature                         : 84 C
        Memory Current Temp                            : N/A
        Memory Max Operating T.Limit Temp              : N/A
    GPU Power Readings
        Average Power Draw                             : 10.64 W
        Instantaneous Power Draw                       : 10.67 W
        Current Power Limit                            : 450.00 W
        Requested Power Limit                          : 450.00 W
        Default Power Limit                            : 450.00 W
        Min Power Limit                                : 150.00 W
        Max Power Limit                                : 450.00 W
    GPU Memory Power Readings
        Average Power Draw                             : N/A
        Instantaneous Power Draw                       : N/A
    Module Power Readings
        Average Power Draw                             : N/A
        Instantaneous Power Draw                       : N/A
        Current Power Limit                            : N/A
        Requested Power Limit                          : N/A
        Default Power Limit                            : N/A
        Min Power Limit                                : N/A
        Max Power Limit                                : N/A
    Power Smoothing                                    : N/A
    Workload Power Profiles
        Requested Profiles                             : N/A
        Enforced Profiles                              : N/A
    Clocks
        Graphics                                       : 210 MHz
        SM                                             : 210 MHz
        Memory                                         : 405 MHz
        Video                                          : 1185 MHz
    Applications Clocks
        Graphics                                       : N/A
        Memory                                         : N/A
    Default Applications Clocks
        Graphics                                       : N/A
        Memory                                         : N/A
    Deferred Clocks
        Memory                                         : N/A
    Max Clocks
        Graphics                                       : 3105 MHz
        SM                                             : 3105 MHz
        Memory                                         : 10501 MHz
        Video                                          : 2415 MHz
    Max Customer Boost Clocks
        Graphics                                       : N/A
    Clock Policy
        Auto Boost                                     : N/A
        Auto Boost Default                             : N/A
    Fabric
        State                                          : N/A
        Status                                         : N/A
        CliqueId                                       : N/A
        ClusterUUID                                    : N/A
        Health
            Summary                                    : N/A
            Bandwidth                                  : N/A
            Route Recovery in progress                 : N/A
            Route Unhealthy                            : N/A
            Access Timeout Recovery                    : N/A
            Incorrect Configuration                    : N/A
            Partition Assigned                         : N/A
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1328209
            Type                          : C
            Name                          : /usr/bin/python3
            Used GPU Memory               : 1706 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 3632600
            Type                          : C
            Name                          : /usr/bin/python3
            Used GPU Memory               : 1706 MiB
    Capabilities
        EGM                                            : disabled


nvidia-smi -q -d CLOCK,POWER,MEMORY -i 1
Timestamp                                              : Fri May 29 18:47:09 2026
Driver Version                                         : 580.126.18
CUDA Version                                           : 13.0

Attached GPUs                                          : 4
GPU 00000000:65:00.0
    FB Memory Usage
        Total                                          : 24564 MiB
        Reserved                                       : 484 MiB
        Used                                           : 3428 MiB
        Free                                           : 20654 MiB
    BAR1 Memory Usage
        Total                                          : 256 MiB
        Used                                           : 6 MiB
        Free                                           : 250 MiB
    Conf Compute Protected Memory Usage
        Total                                          : 0 MiB
        Used                                           : 0 MiB
        Free                                           : 0 MiB
    GPU Power Readings
        Average Power Draw                             : 10.76 W
        Instantaneous Power Draw                       : 10.97 W
        Current Power Limit                            : 450.00 W
        Requested Power Limit                          : 450.00 W
        Default Power Limit                            : 450.00 W
        Min Power Limit                                : 150.00 W
        Max Power Limit                                : 450.00 W
    Power Samples
        Duration                                       : 117.97 sec
        Number of Samples                              : 119
        Max                                            : 12.16 W
        Min                                            : 9.62 W
        Avg                                            : 10.48 W
    GPU Memory Power Readings
        Average Power Draw                             : N/A
        Instantaneous Power Draw                       : N/A
    Module Power Readings
        Average Power Draw                             : N/A
        Instantaneous Power Draw                       : N/A
        Current Power Limit                            : N/A
        Requested Power Limit                          : N/A
        Default Power Limit                            : N/A
        Min Power Limit                                : N/A
        Max Power Limit                                : N/A
    Clocks
        Graphics                                       : 210 MHz
        SM                                             : 210 MHz
        Memory                                         : 405 MHz
        Video                                          : 1185 MHz
    Applications Clocks
        Graphics                                       : N/A
        Memory                                         : N/A
    Default Applications Clocks
        Graphics                                       : N/A
        Memory                                         : N/A
    Deferred Clocks
        Memory                                         : N/A
    Max Clocks
        Graphics                                       : 3105 MHz
        SM                                             : 3105 MHz
        Memory                                         : 10501 MHz
        Video                                          : 2415 MHz
    Max Customer Boost Clocks
        Graphics                                       : N/A
    SM Clock Samples
        Duration                                       : Not Found
        Number of Samples                              : Not Found
        Max                                            : Not Found
        Min                                            : Not Found
        Avg                                            : Not Found
    Memory Clock Samples
        Duration                                       : Not Found
        Number of Samples                              : Not Found
        Max                                            : Not Found
        Min                                            : Not Found
        Avg                                            : Not Found
    Clock Policy
        Auto Boost                                     : N/A
        Auto Boost Default                             : N/A

nvidia-smi topo -m


        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    SYS     SYS     0-27,56-83      0               N/A
GPU1    NODE     X      SYS     SYS     0-27,56-83      0               N/A
GPU2    SYS     SYS      X      NODE    28-55,84-111    1               N/A
GPU3    SYS     SYS     NODE     X      28-55,84-111    1               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks


  



