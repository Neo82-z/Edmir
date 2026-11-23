# 当前执行口径：八卡 A100 上的单 GLM 投影

以此目录的 `design/queue.csv` 和论文仓库 `latex/review/FINAL_12H_PLAN_ZH_2026-11-23.txt`
（11-23 修订方案）为准；`2026-11-20-final-round/` 只作为只读的旧包和评分器来源。真实模型只有
GLM-4.7-Flash。FP16 网格是其 gate/up 几何的均衡代理；BF16 回放来自两个固定路由快照。
硬件单机 8×A100 SXM4 80 GB，NVSwitch 全 NV12。历史数据和冻结评分器使用原有契约。

## 租前确认与资产复用

- **优先 Shadeform / Verda / helsinki-finland-2。** 确认有单节点八卡，且下单时能选现有独立卷
  edmir-moe-assets-20261117（500 GB，卷 ID 5c647cb1-fed1-48ed-b7bf-52fd04c329b0）。
  最后一份本地平台回执确认保留。库存、卷能否挂载与报价以本次核对为准。
- 租前向供应商确认完整 Linux 驱动版本，以及该型节点上的 Fabric Manager 已运行并和
  宿主驱动配套。优先上一轮实测的 580.178.04 或更新的匹配 R580。门槛检查
  Linux CUDA 13.0 GA 对应版本 ≥580.65.06。若服务由平台管理，先确认能取得
  绑定该实例八张 GPU UUID 的宿主证据。
- 同区 CPU 准备时按卷 ID 和 ext4 UUID 34464844-e985-4623-a0e2-3e7cdf43624c 找卷，再从
  卷里的 edmir-assets/ 复用资产。GLM 模型和 tokenizer 按固定 revision 与 54 文件清单校验，
  48 个权重分片合计 62,444,175,504 字节。Nsight Systems 使用卷上的 2025.1.3.140 与原清单。
- 镜像从 vllm-0.20.1-amd64.docker.tar 加载。归档 SHA256 为
  3e52284234d31e1e94c3a8f42ccf913b13f3beb1287d0e1be782fda9688f0725。
  OCI manifest digest 为
  sha256:bf0486e51765766a069f6eb26a63a06748859a568b24200acd3aa4edb58f3d35。
  用既有 verify_image_archive.py 校验后 docker load。标签 edmir/vllm:0.20.1-locked。
  CPU 节点的 Docker 缓存随实例走，GPU 节点仍需从卷上的归档本地加载。
- 在远端 CPU 用此镜像（--pull never、--network none、无 GPU 暴露）运行既有
  EDM-IR/tools/prepare_real_moe.py，重发绑定新采集源码的回执。SM80 构建也放在 CPU：
  build_lt.sh 需要 Linux amd64 CUDA 开发工具链和固定 CUTLASS。若运行镜像缺 nvcc，
  用受控的开发环境编译后将共享库和构建回执写回卷，并在锁定运行镜像内检查动态链接。
- 若同区没有合格八卡，先在目标供应商的远端 Linux amd64 CPU 和持久卷上完成迁移。
  能从原卷直传就复制已验收资产。模型缺件按原 revision 补齐，镜像按原 digest 重建归档。
  重新校验模型各文件、镜像 manifest/各层和工具哈希。目标卷挂载与离线回执验收后再租 GPU。
  Verda 跨区克隆仅在当前账户明确支持时采用。不同供应商各自准备卷。

## 开机验收

宿主上执行：

    python3 experiments/2026-11-23-glm-path-replay/hw_gate.py host --out /path/to/receipts/host.json

若只能看到来宾 VM，用 --provider-receipt 提供供应商工单事实。JSON 字段为
source=provider_host、reference（工单标识）、gpu_uuids（实例八卡 UUID）、
host_driver、fm_version、active=active、substate=running。若用 R620 或更新驱动，
该工单还须提供与独立 FM 版本匹配的 fm_library_version。
A100 的 Fabric State/Status 可以是 N/A，仍须有宿主回执。

再在锁定镜像中执行：

    docker run --rm --pull never --network none --gpus all \
      -v "/path/to/llm-serving:/repo:ro" -v "/path/to/receipts:/receipts" \
      --entrypoint python3 edmir/vllm:0.20.1-locked \
      /repo/experiments/2026-11-23-glm-path-replay/hw_gate.py runtime \
      --host-receipt /receipts/host.json --out /receipts/gate.json

核对驱动和八卡 UUID，验证镜像实际运行 PyTorch 2.11.0+cu130、CUDA 13.0、vLLM 0.20.1。
gate.json PASS 还要求全 NV12、所有卡对双向 P2P、功率和八卡 BF16 负载通过。回放和分析读取此格式。

## 前置顺序

所有设备命令都在目标节点运行。先备好已校验的模型、CPU 回执、源锁、
vLLM 0.20.1 环境和固定 CUTLASS 工作树。各设备作业回执写入新的实验目录。
启动后依次执行 `design/queue.csv` 的前置任务：

1. 用上面的双阶段 `hw_gate.py` 取得 `gate.json`。通过后进入编译检查和冒烟。
2. 装入 CPU 侧编好的 SM80 共享库。构建入口是 `CUTLASS_ROOT=... ./build_lt.sh /abs/path/libedmir_glm.so`。
3. 合成冒烟 `torchrun --standalone --nproc-per-node=8 replay.py --library /abs/path/libedmir_glm.so smoke --gate gate.json --out smoke.json`。
   - BF16 五臂使用真实 K/N，验证 symmetric rendezvous、Triton 对端读、AllToAll、原生
     pynccl all-gatherv 和 vLLM 投影调用。
   - FP16 网格七臂（D/BNEW 的 q1/q4、SLT、D_Q1_G1、BOLD）在 M=1 和 M=64 上各跑一次准入，
     覆盖 FP16 Triton、cuBLASLt heuristic 和 CUTLASS 桥接。
   - 某个 arm 抛异常时记为 runtime_error，其余 arm 照常跑完；有任何失败，命令以非零退出并列出失败项。
   - 失败项是修代码的信号：能修就先修，修完重跑冒烟（回执绑定源码和库哈希）。
   - `prepare` 只要求回执与当前 replay.py、gate_up.py、native.py、balanced.py 和库文件哈希一致。
     失败的 arm 记入 RUN.json，由正式准入裁决（BNEW 失败会开启桥接），不会挡住能跑的 arm。
   - 先看完冒烟结果，再加载模型。
4. `timeout 900 python native.py --out /abs/path/moe_configs` 做单模型原生调优：
   E=8、N=1536（配置文件命名维度，gate/up 实际输出 3072），7 个 M，每格至多 4 个配置。
   保留候选的数值、event 计时与失败状态。采用返回的 vLLM 配置文件。
5. `torchrun --standalone --nproc-per-node=8 replay.py microbench --gate gate.json --out microbench.json`
   八卡同时测，留下全部八卡样本：
   - 单 peer 读：5 个 CTA 档（27–432），每卡读下一张卡的 64 MiB；
   - 七 peer 同时读：每个 peer 4–64 个 CTA（总 28–448），每张卡的 7 个读者各取其缓冲区的不重叠八分之一，
     共 56 MiB，超过 L2；
   - HBM 拷贝、单个极小 kernel、对等 barrier；
   - 11 个 M 档上的等分 `all_to_all_single`（每卡 8M×2048 FP16），与 BNEW/SLT 的搬运相同。
   这些数用来对照 scenario 里的 `b_in`（1 和 7 peer）、`b_hbm`、`G_launch` 和 `a2a`，只作解释，不重封预测。
6. `torchrun --standalone --nproc-per-node=8 replay.py --library /abs/path/libedmir_glm.so tune-slt --gate gate.json --out slt.json`。
   每格每卡最多查 8 个 cuBLASLt heuristic，对 ≤32 MiB 的可用候选用独立输入数值检查及 event 计时。
   冻结最快合格候选和完整日志。无合格项的卡将使对应 SLT 不准入。
7. 采集 GLM 两个快照。采集器会核对 CPU 回执里 7 个采集文件的哈希，所以 CPU 回执必须用封存提交
   之后的这一版代码重新生成，节点上的仓库也必须正好是封存提交。不使用 guard，租期由操作者管理。

       docker run --rm --pull never --network none --gpus all --ipc=host \
         -v /path/to/llm-serving:/code:ro -v /path/to/model:/model:ro \
         -v /path/to/cpu-records:/cpu-records:ro -v /path/to/moe_configs:/moe_configs:ro \
         -v /path/to/capture-root:/work \
         -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
         -e HF_HOME=/work/cache/hf -e XDG_CACHE_HOME=/work/cache -e VLLM_CACHE_ROOT=/work/cache/vllm \
         -e TRITON_CACHE_DIR=/work/cache/triton -e TOKENIZERS_PARALLELISM=false \
         -e PYTHONPATH=/code/EDM-IR -e PYTHONUNBUFFERED=1 -e VLLM_HOST_IP=127.0.0.1 \
         --entrypoint python3 edmir/vllm:0.20.1-locked /code/EDM-IR/tools/run_real_moe.py --execute \
         --mode routes --snapshot-replay --native-tune "/moe_configs/E=8,N=1536,device_name=NVIDIA_A100-SXM4-80GB.json" \
         --corpus /code/experiments/2026-11-17-real-moe-local/corpus/requests-source.json \
         --tokens /cpu-records/<新回执目录>/requests-tokenized.json \
         --cpu-receipt /cpu-records/<新回执目录>/cpu-receipt.json \
         --manifest /code/experiments/2026-11-16-next-window/model-manifest.json --model-dir /model \
         --source-lock /code/experiments/2026-11-17-real-moe-local/vllm-source-lock.json --out /work/capture

   - 只跑 S32（32 个请求，每卡 4 个）：首个纯 prefill 是第 1 步，每卡 2048 个 token；纯 decode 第 8 步每卡 4 个。
   - 每卡保存第 23 层的 BF16 输入、本卡专家权重、top-k、原生 launch 的对齐张量和实际命中的配置。
     路由布局按实际记录：2026-11-17 在同一 vLLM 上观察到的是本卡路由（`local`），汇集后路由也接受。
   - 调优失败时去掉 `--native-tune`，采集回执记为 default，作为诊断单列。
   - 快照写在 `<capture-root>/capture/workers/snapshots`，即后面的 `--snapshots`。
   - 采集修一次仍失败，就用 `prepare --without-snapshots "<原因>"` 只登记网格（见下节），不要卡住网格。

## 容器内运行

除采集外，GPU 作业都在锁定镜像里跑。回放和准入要用 NCCL 和对称内存，必须带 `--gpus all --ipc=host`：

    RUN="docker run --rm --pull never --network none --gpus all --ipc=host \
      -v /path/to/llm-serving:/repo:ro -v /path/to/work:/work -v /path/to/capture-root:/capture:ro \
      -e TRITON_CACHE_DIR=/work/cache/triton -e PYTHONUNBUFFERED=1 \
      -w /repo/experiments/2026-11-23-glm-path-replay --entrypoint python3 edmir/vllm:0.20.1-locked"
    T="-m torch.distributed.run --standalone --nproc-per-node=8 replay.py --library /work/libedmir_glm.so"

    $RUN $T smoke --gate /work/gate.json --out /work/smoke.json
    $RUN native.py --out /work/moe_configs
    $RUN $T microbench --gate /work/gate.json --out /work/microbench.json
    $RUN $T tune-slt --gate /work/gate.json --out /work/slt.json

`admit`、`lock-admission`、`time-all` 和 `analyze.py` 是父进程，用 `$RUN replay.py ...` 直接运行，
它们自己按需启动八卡 torchrun。

## 统一准备与运行

统一设置 `--run <new-run> --library <libedmir_glm.so> --snapshots <capture>/workers/snapshots`
作为 `replay.py` 的全局参数，放在子命令前。RUN 登记了快照时，`admit`、`time-all` 缺 `--snapshots`
会在启动任何子进程之前报错；`time` 在占用进程编号之前先核对快照目录和回执哈希，
参数写错不会消耗正式进程的机会。

**prepare 之后不得再改包里的任何文件。** 运行时核对全部 `*.py`、库和冻结回执的哈希，
改动任何一个（包括 analyze.py），之后的作业都会拒绝运行。

- `verify` 在 CPU 上检查两个快照、哈希和保存的原生对齐。
- `prepare --gate gate.json --smoke-receipt smoke.json --prediction preboot_prediction --slt-receipt slt.json --microbench microbench.json --native-tune <实际捕获配置> --native-trials <调优目录>/trials.json`
  一次登记 33 个均衡 seed、2 个快照以及全部 build。后续不得覆盖 RUN 或其中冻结的回执。
  - 只有硬件门槛和设计会挡住 prepare。冒烟回执必须是当前代码和库的，但其中失败的 arm 只记进
    RUN.json 的 `smoke_failures`，由正式准入裁决。
  - `--slt-receipt` 缺失时 SLT 记为 unsupported；`--microbench` 缺失或个别探针失败时只少一份解释数据。
    缺什么都写进 RUN.json 的 `missing`。default 捕获省略 `--native-tune`，`--native-trials` 照样给。
  - 采集拿不到时，用 `--without-snapshots "<原因>"` 代替 `--snapshots`：只登记 11 格网格，
    快照进程不跑，快照比较在结果里显示为不完整。
- 顺序执行父进程 `admit --unit GLM_PREFILL`、`admit --unit GLM_DECODE`、`admit`，
  然后 `lock-admission`。快照的每臂独立 torchrun，一个 arm 失败不会阻塞其他 arm。
  均衡默认每个 seed 一个进程，异常后由独立子进程完成剩余 arm。完整主准入记录后才锁定。
- 若任意均衡 BNEW seed 不合格，开启同一 11 格的 g1 桥接。直接臂使用 64×128 tile，
  BOLD 使用逐对 NCCL 搬运接既有 CUTLASS 64×128 kernel。两臂算法和传输共同变化，
  只作为弱传输桥接单列。匹配 B-a 的缺测格仍为空。桥接门在正式计时前锁定。
- `time-all` 按队列运行，或按进程 ID 用
  `torchrun --standalone --nproc-per-node=8 replay.py <全局参数> time --process <id>`。
  - 只有队列里第一个进程失败且没有写出任何计时行时，`time-all` 才停下来让你检查；
    再次运行会跳过已启动的进程继续往下跑。
  - **重试规则（开机前定，机械执行）**：首轮跑完整个队列后，每个进程最多重试一次，
    条件只看首次尝试是否少记了 arm，或记有 runtime_error/timeout；不看任何计时数值。
    重试整体重跑该进程的全部 arm，分析只用重试那次，首次尝试的原始行保留。
    数值失败（准入不合格）不重试。重试后仍失败的就是缺测。
  - 不另建 run，也不跨 run 拼配对。
- `python analyze.py --run <run> --out <results>` 输出有效配对、交叉方向、离线选择
  和论文 JSON。队列共 78 个正式进程、390 个主臂计时。桥接只在触发后多测最多 132 个臂。
- BF16 direct 和 BNEW 使用同一逐源、跨本地专家的 tile 工作表。q1 每个非空源一次
  Triton 投影，q4 每源另有一次 FP32 归并，最终输出 BF16。原生直接使用捕获的
  aligned IDs、expert IDs 与 padding extent。统一输出重排只在准入执行。
- Direct 的读取前发布 barrier 与读取后的释放 barrier 都在计时区，保护下一轮复用。
  所有臂同步后记录 max-rank 墙钟和 CUDA event，CUDA event 视作设备时间线经过时间，
  包括开始和结束记录之间的空档。另记每次迭代 CPU 发出全部工作所用的时间（enqueue）；
  分析表里的 `enqueue_over_wall` 接近 1 表示该格受 Python 发射开销限制，解读时单独标注。逐卡记录输入、有效投影、DP padding、零写和对齐槽位。
  回放范围到 gate/up 输出为止。静态路由、投影对齐预计算、正确性重排在计时外。

## 事前预测

`preboot_prediction/` 使用原解析器与原 scenario 参数在 CPU 重新封存。主 R1 只检查
四条网格主对照。宽区间命中要求完整的实测严格交叉落在区间内并且方向相同，单列为
`hit_loose`。严格预测使用原来的一档容差。实测覆盖不全或方向未定不可判中。
微基准用于解释并给出有效带宽尺度，原 scenario 预测不按实测重封。
P0 在 11 格全部选 SLT，R2 仅报离线选择和测后 regret，同选择规则的相等结果按原值报告。

重封是原封存的子集：用冻结的 `predict.py` 和 `scenario.json` 在本设计上重算，四个输出逐字节复现，
GLM 的 77 行预测和 44 个比较类别与 11-21 的封存完全相同。

scenario 参数是对着历史 CUTLASS J1 的 P1 M512 实测拟合的，本轮网格的 D/BNEW 改用 Triton 逐源消费者。
两者 tile 几何相同（64×128·g，split-K 1 或 4），重读次数 ρ 和字节账本不变，但 kernel 效率不同。
论文须写明：预测按 CUTLASS 标定，实测用 Triton；`D_Q4_G2` 的"P2 固定策略"只指 q4/g2 这组参数，
不再是 P2 的那个 kernel。

## arm 状态列

`design/arms.csv` 的 `gpu_status` 只说明该 arm 的计算 kernel 在本轮之前是否在 A100 上跑过：
`gpu_run` 只有 BOLD（CUTLASS 64×128）和 NATIVE（vLLM fused MoE Triton kernel）。
三个 Triton D arm、BNEW、SLT 和全部快照直读/staged arm 都是 `new`。各 arm 的主机侧调度代码
都是本轮新写的，一律以冒烟和准入为准。
