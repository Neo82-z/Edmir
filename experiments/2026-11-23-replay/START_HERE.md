# 当前执行口径：八卡 A100 上三个模型的第一投影

以此目录的 `design/queue.csv` 和论文仓库 `latex/review/FINAL_12H_PLAN_ZH_2026-11-23.txt`
为准；`2026-11-20-final-round/` 只作为只读的旧包和评分器来源。

三个模型都测，协议相同：

| 模型 | 宽度代号 | 路由专家 / top-k | gate/up 的 K×N | 采集层 |
|---|---|---|---|---|
| GLM-4.7-Flash | GLM | 64 / 4 | 2048×3072 | 23 |
| Qwen3.5-35B-A3B（只跑文本） | QWEN35 | 256 / 8 | 2048×1024 | 20 |
| DeepSeek-V2-Lite-Chat | DSV2L | 64 / 6 | 2048×2816 | 13 |

每个模型有两类证据：11 个 FP16 均衡格（该模型 gate/up 几何的代理，M=1…1024），以及两个 BF16 快照
（S32 负载下采集层的首个纯 prefill 和第八个纯 decode，真实激活、权重和路由）。
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
  48 个权重分片合计 62,444,175,504 字节。Qwen3.5-35B-A3B（revision 2508b24，14 个分片共
  71,903,878,016 字节）和 DeepSeek-V2-Lite-Chat（revision 8586474）也要放到卷上，
  清单的生成与校验见"开机前在 CPU 上为每个模型准备的东西"。Nsight Systems 使用卷上的 2025.1.3.140 与原清单。
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

    python3 experiments/2026-11-23-replay/hw_gate.py host --out /path/to/receipts/host.json

若只能看到来宾 VM，用 --provider-receipt 提供供应商工单事实。JSON 字段为
source=provider_host、reference（工单标识）、gpu_uuids（实例八卡 UUID）、
host_driver、fm_version、active=active、substate=running。若用 R620 或更新驱动，
该工单还须提供与独立 FM 版本匹配的 fm_library_version。
A100 的 Fabric State/Status 可以是 N/A，仍须有宿主回执。

再在锁定镜像中执行：

    docker run --rm --pull never --network none --gpus all \
      -v "/path/to/llm-serving:/repo:ro" -v "/path/to/receipts:/receipts" \
      --entrypoint python3 edmir/vllm:0.20.1-locked \
      /repo/experiments/2026-11-23-replay/hw_gate.py runtime \
      --host-receipt /receipts/host.json --out /receipts/gate.json

核对驱动和八卡 UUID，验证镜像实际运行 PyTorch 2.11.0+cu130、CUDA 13.0、vLLM 0.20.1。
gate.json PASS 还要求全 NV12、所有卡对双向 P2P、功率和八卡 BF16 负载通过。回放和分析读取此格式。

## 上机清单

带上节点的东西（都在卷上或仓库里）：

| 项目 | 要求 |
|---|---|
| 仓库 | 检出封存提交，工作区干净 |
| 锁定镜像 | `edmir/vllm:0.20.1-locked`，由卷上的归档加载并校验 |
| 三个模型目录 | GLM 按既有 54 文件清单校验；Qwen3.5 和 DSV2-Lite 用 `make_model_manifest.py` 生成清单并校验（见下） |
| 每个模型的 CPU 回执和 token 文件 | 用**当前**采集代码、按模型各生成一份；2026-11-17 的旧回执会被拒绝 |
| `libedmir_replay.so` | CUDA 13 工具链编译，链接 `libcublasLt.so.13` |
| 宿主机 gate 回执 | `host.json` |

两段 gate 通过后，先跑一次只读的自检，再碰 GPU（`$RUN` 的定义见"容器内运行"，其中已挂载 `/cpu-records`）：

    $RUN preflight.py --library /work/libedmir_replay.so \
      --cpu-receipt /cpu-records/GLM/cpu-receipt.json --cpu-receipt /cpu-records/QWEN35/cpu-receipt.json \
      --cpu-receipt /cpu-records/DSV2L/cpu-receipt.json --out /work/preflight.json

它检查设计表与封存预测是否完好、归档文件是否齐全、仓库是否有未提交的改动、CPU 回执是否对应当前采集代码、
镜像里的 torch/vLLM 版本、共享库能否解析到 `libcublasLt.so.13` 并加载、Triton 缓存目录是否可写。
任何一项 FAIL 都先修好。容器里没有 git 时，提交这一项显示 SKIP，手工记下检出的提交。

整轮的顺序是：gate → 自检 → 冒烟 → 三个模型的原生调优 → 微基准 → 标定版预测 → SLT 调优 → 三个模型的采集 → prepare →
准入 → 锁定 → 计时 → 分析 → 把 run 目录、采集目录和全部回执拷到机外。
每一步的产物都是后一步的输入；决策点只有四个：
- gate 不通过：停。
- 冒烟有失败项：能修就修（修代码要重新提交并重跑冒烟），修不了就继续。
- 某个模型的采集修一次仍失败：prepare 时不传它的快照目录，该模型只保留 11 格网格，另外两个模型不受影响。
- 第一个正式进程失败且没有写出计时行：`time-all` 会停下来，查日志后再运行它会继续。

## 前置顺序

所有设备命令都在目标节点运行。先备好已校验的模型、CPU 回执、源锁、
vLLM 0.20.1 环境和固定 CUTLASS 工作树。各设备作业回执写入新的实验目录。
启动后依次执行 `design/queue.csv` 的前置任务：

1. 用上面的双阶段 `hw_gate.py` 取得 `gate.json`。通过后进入编译检查和冒烟。
2. 装入 CPU 侧编好的 SM80 共享库。构建入口是 `CUTLASS_ROOT=... ./build_lt.sh /abs/path/libedmir_replay.so`。
3. 合成冒烟 `torchrun --standalone --nproc-per-node=8 replay.py --library /abs/path/libedmir_replay.so smoke --gate gate.json --out smoke.json`。
   - BF16 五臂按三个模型各自的几何（本卡专家数、宽度、top-k）各跑一遍，验证 symmetric rendezvous、
     Triton 对端读、AllToAll、原生 pynccl all-gatherv 和 vLLM 投影调用。
   - FP16 网格七臂（D/BNEW 的 q1/q4、SLT、D_Q1_G1、BOLD）在每个宽度的 M=1 和 M=64 上各跑一次准入，
     覆盖 FP16 Triton、cuBLASLt heuristic 和 CUTLASS 桥接。
   - D/BNEW 的消费者启动在这里第一次以 CUDA graph 捕获并回放，回执里每个 arm 记有 `launch_mode`。
   - 某个 arm 抛异常时记为 runtime_error，其余 arm 照常跑完；有任何失败，命令以非零退出并列出失败项。
   - 失败项是修代码的信号：能修就先修，修完重跑冒烟（回执绑定源码和库哈希）。
   - `prepare` 只要求回执与当前 replay.py、gate_up.py、native.py、balanced.py 和库文件哈希一致。
     失败的 arm 记入 RUN.json，由正式准入裁决（BNEW 失败会开启桥接），不会挡住能跑的 arm。
   - 先看完冒烟结果，再加载模型。
4. 三个模型各做一次原生调优，写进同一个配置目录：

       for w in GLM QWEN35 DSV2L; do timeout 900 $RUN native.py --out /work/moe_configs --width $w; done

   每个模型调自己的形状（本卡专家数 E、中间维 N/2）：GLM 是 E=8、N=1536，Qwen3.5 是 E=32、N=512，
   DSV2-Lite 是 E=8、N=1408；7 个 M，每格至多 4 个配置。保留候选的数值、event 计时与失败状态
   （`trials-<模型>.json`）。某个模型调优失败时，它的采集不带 `--native-tune`，按默认配置记录。
5. `torchrun --standalone --nproc-per-node=8 replay.py microbench --gate gate.json --out microbench.json`
   八卡同时测，留下全部八卡样本：
   - 单 peer 读：5 个 CTA 档（27–432），每卡读下一张卡的 64 MiB；
   - 七 peer 同时读：每个 peer 4–64 个 CTA（总 28–448），每张卡的 7 个读者各取其缓冲区的不重叠八分之一，
     共 56 MiB，超过 L2；
   - HBM 拷贝、单个极小 kernel（即时启动）、对等 barrier；
   - graph 回放下多发一个 kernel 的开销（8 个极小 kernel 录成一个 graph，回放时间除以 8）；
   - 共用消费者在本地行上的计算速率（M=1024，g=1 和 g=2 各一次，所有 SM 槽位占满，graph 回放）；
   - 11 个 M 档上的等分 `all_to_all_single`（每卡 8M×2048 FP16），与 BNEW/SLT 的搬运相同。
   这些数有两个用途：解释结果；作为第二版封存预测的参数（下一步）。
5b. 封存第二版预测（CPU 步骤，不用 GPU）：

        $RUN calibrate_node.py --microbench /work/microbench.json --out /work/calibrated

   用冻结的 `calibrate.py` 和 `predict.py`，只把情景参数换成本机微基准的实测值，在任何正式计时之前封存。
   探针到参数的对应关系在开机前定死，写在 `calibrate_node.py` 开头：
   - `b_in`：单 peer 和七 peer 读带宽随 CTA 数的表；`b_hbm`：HBM 拷贝；`a2a`：11 个 M 档的线性拟合；
   - `mma_peak`：消费者的本地计算速率；`G_launch`：graph 回放下的单次启动开销；
   - `F_direct`：两次对等 barrier；`F_staged`：定义为 1 µs（staged 除 AllToAll 外没有固定步骤）；
   - 没有探针的参数（SM 数、tile、占用、L2、`lt_launch` 等）沿用情景值，回执里逐项列出。
   必需的探针失败时命令返回 3，不产生第二版预测，实验照常进行，只对开机前那版预测计分。
6. `torchrun --standalone --nproc-per-node=8 replay.py --library /abs/path/libedmir_replay.so tune-slt --gate gate.json --out slt.json`。
   每格每卡最多查 8 个 cuBLASLt heuristic，对 ≤32 MiB 的可用候选用独立输入数值检查及 event 计时。
   冻结最快合格候选和完整日志。无合格项的卡将使对应 SLT 不准入。
7. 采集三个模型的快照，每个模型一次，各写进自己的目录。模型由环境变量 `EDMIR_MOE_MODEL` 选择
   （GLM、QWEN35、DSV2L），采集代码是同一套。采集器会核对 CPU 回执里 7 个采集文件的哈希，
   所以每个模型的 CPU 回执都必须用当前这版代码生成，节点上的仓库也必须正好是封存提交。
   不使用 guard，租期由操作者管理。

       capture() {  # capture <模型> <模型目录> <清单> <调优配置文件名>
         docker run --rm --pull never --network none --gpus all --ipc=host \
           -v /path/to/llm-serving:/code:ro -v "$2":/model:ro \
           -v /path/to/cpu-records:/cpu-records:ro -v /path/to/moe_configs:/moe_configs:ro \
           -v /path/to/capture-root:/work \
           -e EDMIR_MOE_MODEL=$1 \
           -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
           -e HF_HOME=/work/cache/hf -e XDG_CACHE_HOME=/work/cache -e VLLM_CACHE_ROOT=/work/cache/vllm \
           -e TRITON_CACHE_DIR=/work/cache/triton -e TOKENIZERS_PARALLELISM=false \
           -e PYTHONPATH=/code/EDM-IR -e PYTHONUNBUFFERED=1 -e VLLM_HOST_IP=127.0.0.1 \
           --entrypoint python3 edmir/vllm:0.20.1-locked /code/EDM-IR/tools/run_real_moe.py --execute \
           --mode routes --snapshot-replay --native-tune "/moe_configs/$4" \
           --corpus /code/experiments/2026-11-17-real-moe-local/corpus/requests-source.json \
           --tokens /cpu-records/$1/requests-tokenized.json --cpu-receipt /cpu-records/$1/cpu-receipt.json \
           --manifest "$3" --model-dir /model \
           --source-lock /code/experiments/2026-11-17-real-moe-local/vllm-source-lock.json --out /work/$1
       }
       capture GLM    /path/to/glm    /code/experiments/2026-11-16-next-window/model-manifest.json "E=8,N=1536,device_name=NVIDIA_A100-SXM4-80GB.json"
       capture QWEN35 /path/to/qwen35 /cpu-records/manifests/QWEN35.json "E=32,N=512,device_name=NVIDIA_A100-SXM4-80GB.json"
       capture DSV2L  /path/to/dsv2l  /cpu-records/manifests/DSV2L.json  "E=8,N=1408,device_name=NVIDIA_A100-SXM4-80GB.json"

   - 只跑 S32（32 个请求，每卡 4 个）：首个纯 prefill 是第 1 步，每卡 2048 个 token；纯 decode 第 8 步每卡 4 个。
   - 每卡保存采集层的 BF16 输入、本卡专家权重、top-k、原生 launch 的对齐张量和实际命中的配置。
     路由布局按实际记录：2026-11-17 在 GLM 上观察到的是本卡路由（`local`），汇集后路由也接受。
   - 调优失败的模型去掉 `--native-tune`，采集回执记为 default，作为诊断单列。
   - 快照写在 `<capture-root>/<模型>/workers/snapshots`，后面每个模型给一次 `--snapshots`。
   - Qwen3.5 带视觉塔，档案里已设 `language_model_only`，只跑文本。
   - **只有 GLM 的采集钩子在这台机型上跑过（11-17）。** Qwen3.5 和 DSV2-Lite 在 vLLM 0.20.1 里用的是同一个
     FusedMoE 层、同一个 Triton kernel、同样的线性专家放置，钩子按源码核对过、在 CPU 上用桩测试过，
     但没有在 GPU 上跑过。它们的采集如果失败：先看 `workers/capture-failure-rank*.json` 里的原因，修一次；
     仍失败就不传它的快照目录，该模型只保留网格。
   - 11-20 方案里登记的"TP2/DP4 等并行配置兜底、重建源归属"没有实现：采集器只接受 TP1/DP8/EP8。
     某个模型在这个配置下起不来时，按上一条处理，不用合成路由替代。

### 开机前在 CPU 上为每个模型准备的东西

在锁定镜像里、不暴露 GPU。仓库只读挂在 `/code`，模型只读挂在 `/model`，输出写进 `/cpu-records`：

    cpu() {  # cpu <模型> <模型目录> <脚本和参数...>
      local m=$1 dir=$2; shift 2
      docker run --rm --pull never --network none \
        -v /path/to/llm-serving:/code:ro -v "$dir":/model:ro -v /path/to/cpu-records:/cpu-records \
        -e EDMIR_MOE_MODEL=$m -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e HF_HOME=/cpu-records/cache/hf -e XDG_CACHE_HOME=/cpu-records/cache -e TOKENIZERS_PARALLELISM=false \
        -e PYTHONPATH=/code/EDM-IR -w /code --entrypoint python3 edmir/vllm:0.20.1-locked "$@"
    }
    mkdir -p /path/to/cpu-records/manifests

    # 清单：GLM 用仓库里现成的；另外两个模型由本地文件生成，并与仓库里钉住的 config 哈希
    #（Qwen3.5 另加上游的 14 个分片 LFS 摘要）核对。清单放在 manifests/ 下，不能放进
    # /cpu-records/<模型>：下一步要求那个目录事先不存在。
    cpu QWEN35 /path/to/qwen35 EDM-IR/tools/make_model_manifest.py --model-dir /model \
      --sources experiments/2026-11-20-final-round/models/SOURCES.json \
      --weights experiments/2026-11-20-final-round/models/QWEN35.weights.json --out /cpu-records/manifests/QWEN35.json
    cpu DSV2L /path/to/dsv2l EDM-IR/tools/make_model_manifest.py --model-dir /model \
      --sources experiments/2026-11-20-final-round/models/SOURCES.json --out /cpu-records/manifests/DSV2L.json

    # 每个模型一份 CPU 回执和 token 文件（输出目录 /cpu-records/<模型> 不能已存在）
    cpu GLM /path/to/glm EDM-IR/tools/prepare_real_moe.py --model-dir /model \
      --manifest experiments/2026-11-16-next-window/model-manifest.json \
      --source-lock experiments/2026-11-17-real-moe-local/vllm-source-lock.json \
      --corpus experiments/2026-11-17-real-moe-local/corpus/requests-source.json --out /cpu-records/GLM
    for m in QWEN35 DSV2L; do   # 模型目录按实际路径替换
      cpu $m /path/to/$m EDM-IR/tools/prepare_real_moe.py --model-dir /model --manifest /cpu-records/manifests/$m.json \
        --source-lock experiments/2026-11-17-real-moe-local/vllm-source-lock.json \
        --corpus experiments/2026-11-17-real-moe-local/corpus/requests-source.json --out /cpu-records/$m
    done

`prepare_real_moe.py` 会核对：架构已在 vLLM 里注册、引擎参数被接受、原始/HF/vLLM 三种 tokenizer 对固定
请求的切分一致、每条请求能精确截到 512 token。GLM 还要 4096 token 的长输入，另外两个模型只做 512。
这一步如果在某个模型上报错（最可能是 tokenizer 一致性或聊天模板），要在开机前解决，它不占 GPU 机时。
Qwen3.5 是多模态配置，`vocab_size` 在 `config.json` 的 `text_config` 里；CPU 准备和采集都按这个位置读取
（2026-11-25 修正，此前会报 `KeyError`）。这条路径只在 CPU 上用仓库里钉住的三份 config 测过，
Qwen3.5 和 DSV2-Lite 的整个 CPU 准备还没有在锁定镜像里跑通过，请先跑它。

## 容器内运行

除采集外，GPU 作业都在锁定镜像里跑。回放和准入要用 NCCL 和对称内存，必须带 `--gpus all --ipc=host`：

    RUN="docker run --rm --pull never --network none --gpus all --ipc=host \
      -v /path/to/llm-serving:/repo:ro -v /path/to/work:/work -v /path/to/capture-root:/capture:ro \
      -v /path/to/cpu-records:/cpu-records:ro \
      -e TRITON_CACHE_DIR=/work/cache/triton -e PYTHONUNBUFFERED=1 \
      -w /repo/experiments/2026-11-23-replay --entrypoint python3 edmir/vllm:0.20.1-locked"
    T="-m torch.distributed.run --standalone --nproc-per-node=8 replay.py --library /work/libedmir_replay.so"

    $RUN $T smoke --gate /work/gate.json --out /work/smoke.json
    for w in GLM QWEN35 DSV2L; do timeout 900 $RUN native.py --out /work/moe_configs --width $w; done
    $RUN $T microbench --gate /work/gate.json --out /work/microbench.json
    $RUN $T tune-slt --gate /work/gate.json --out /work/slt.json

`admit`、`lock-admission`、`time-all` 和 `analyze.py` 是父进程，用 `$RUN replay.py ...` 直接运行，
它们自己按需启动八卡 torchrun。

## 统一准备与运行

统一设置 `--run <new-run> --library <libedmir_replay.so> --snapshots <capture>/workers/snapshots`
作为 `replay.py` 的全局参数，放在子命令前。RUN 登记了快照时，`admit`、`time-all` 缺 `--snapshots`
会在启动任何子进程之前报错；`time` 在占用进程编号之前先核对快照目录和回执哈希，
参数写错不会消耗正式进程的机会。

**prepare 之后不得再改包里的任何文件。** 运行时核对全部 `*.py`、库和冻结回执的哈希，
改动任何一个（包括 analyze.py），之后的作业都会拒绝运行。

- `verify` 在 CPU 上检查给出的各个快照、哈希和保存的原生对齐，并列出设计里还缺哪些快照。
- `prepare --gate gate.json --smoke-receipt smoke.json --prediction preboot_prediction --slt-receipt slt.json --microbench microbench.json --calibrated-prediction /work/calibrated --native-tune <某模型的调优配置> --native-trials <调优目录>/trials-<模型>.json`
  一次登记 99 个均衡 seed、通过校验的快照（最多 6 个）以及全部 build。后续不得覆盖 RUN 或其中冻结的回执。
  - `--snapshots`、`--native-tune`、`--native-trials` 每个模型各给一次（重复这个参数）。
    没有给出快照目录的模型记入 `missing`，它的快照进程不跑，网格照常。给出了但校验不过的快照会报错：
    修好它，或者不传那个目录。
  - `--calibrated-prediction` 必须出自同一份 `microbench.json`（核对哈希）。它在这里被复制进 RUN 并登记哈希，
    所以第二版预测一定早于准入和计时。没有它时只对开机前那版预测计分，并记入 `missing`。
  - 只有硬件门槛和设计会挡住 prepare。冒烟回执必须是当前代码和库的，但其中失败的 arm 只记进
    RUN.json 的 `smoke_failures`，由正式准入裁决。
  - `--slt-receipt` 缺失时 SLT 记为 unsupported；`--microbench` 缺失或个别探针失败时只少一份解释数据。
    缺什么都写进 RUN.json 的 `missing`。default 捕获的模型不给 `--native-tune`，`--native-trials` 照样给。
  - 三个模型的采集都拿不到时，用 `--without-snapshots "<原因>"` 代替 `--snapshots`：只登记 33 格网格。
- 运行父进程 `admit`（也可以用 `--unit <快照或格>` 分批），
  然后 `lock-admission`。快照的每臂独立 torchrun，一个 arm 失败不会阻塞其他 arm。
  均衡默认每个 seed 一个进程，异常后由独立子进程完成剩余 arm。完整主准入记录后才锁定。
- 若任意均衡 BNEW seed 不合格，开启全部 33 格的 g1 桥接。直接臂使用 64×128 tile，
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
  和论文 JSON。把 `<results>/paper_results.json` 复制到论文仓库的 `latex/data/2026-11-boundary/`，
  再运行 `make boundary-results`，论文第 8 节的表和图就由它生成（步骤见论文仓库 `latex/submission/README.md`）。队列共 234 个正式进程（快照 36 个、网格 198 个）、1170 个主臂计时；桥接只在触发后多测最多 396 个臂。
  队列按"先快照、后网格"排，各自内部按 GLM、Qwen3.5、DSV2-Lite 的顺序整模型排列：租期不够时从队尾放弃，
  丢的是排在后面的整个模型，不会让每个模型都缺一部分。
- BF16 direct 和 BNEW 使用同一逐源、跨本地专家的 tile 工作表。q1 每个非空源一次
  Triton 投影，q4 每源另有一次 FP32 归并，最终输出 BF16。
- **消费者启动以 CUDA graph 回放。** D/BNEW 每个 arm 的逐源投影和归并启动在计时前录成一个 CUDA graph，
  每次迭代回放一次；pack、AllToAll 和对等 barrier 仍即时执行。kernel、参数和启动次数都不变，
  只去掉 Python 逐个发射的开销（vLLM serving 的 decode 本来就在 graph 里）。
  一个进程里要计时的消费者 arm 在计时前一起捕获，准入也用同一方式。任何一张卡捕获失败时，
  所有卡上这些 arm 都改为即时执行，同一进程的 D 与 BNEW 始终同一启动方式；
  这些行的 `launch_mode` 为 eager，在结果的 `eager_consumer_rows` 中单列。
  **开机前定的判据：主分析只用预定的启动方式，即 graph 回放。** D/BNEW 或 NATIVE 退回即时执行的行不进入
  效应、交叉、计分、阶段和 regret；它所在的配对因此不完整，该格记为 incomplete。这些行之间的比较
  另列在 `eager_fallback`，不计分，也不与主结果合并。即时启动给同一消费者比较的两侧都加上启动开销，
  把比值拉向 1；一条 M 序列里混用两种方式会移动交叉点。SLT 和 BOLD 本来就没有 graph，两边都保留。
  NATIVE 的 gate/up kernel 也录成 graph 回放（vLLM 在非 eager 模式下本来就这样运行它），all-gatherv 即时执行。
  NATIVE 单独成组：它捕获失败时只有它自己退回即时执行，记入 `mixed_launch_mode_rows`，不影响 D/BNEW。
  SLT（AllToAll + 一次 cuBLASLt，经 ctypes 调用）和 BOLD 仍即时执行。
  BOLD 每次迭代 8 段逐对 NCCL 加 8 次 CUTLASS，无法录入 graph，所以桥接比较的是
  graph 回放的 D_Q1_G1 与即时执行的 BOLD，这又是一处不对称，只作定性参考。原生直接使用捕获的
  aligned IDs、expert IDs 与 padding extent。统一输出重排只在准入执行。
- **每次迭代分段计时。** 在设备时间线上用 CUDA event 把一次迭代分成几段，逐卡逐迭代保存：
  `pack`（仅快照）、`transport`（AllToAll、all-gatherv，或直读的发布 barrier）、`consumer`、
  `release`（直读的释放 barrier；其余 arm 接近零）。每个 arm 记录相同数量的 event。BOLD 的搬运和计算交错，不分段。
  分析输出 `phases.csv`，并给出"去掉直读释放 barrier"的敏感性结果（`release_sensitivity`）。
  释放 barrier 保护缓冲区复用，用双缓冲的部署不需要它；主判读仍用含两次 barrier 的总时间。
  **开机前定的算法：**在同一次迭代、同一张卡上，用这张卡的墙钟减去它自己的释放段，再取最慢卡，
  然后按进程取中位数、按配对算比值。barrier 在各卡上同时结束，先算完的卡把差额等在释放段里，
  所以"各卡最长释放段"是最快那张卡的等待时间，不能从最慢卡的墙钟里减。没有阶段数据的 direct 行
  不参与敏感性分析。`phases.csv` 里 `release_ms` 是各卡最长释放段，`release_added_ms` 是 barrier
  给墙钟实际增加的时间（墙钟减去上述去 barrier 的墙钟），图里画的是后者。
- Direct 的读取前发布 barrier 与读取后的释放 barrier 都在计时区，保护下一轮复用。
  所有臂同步后记录 max-rank 墙钟和 CUDA event，CUDA event 视作设备时间线经过时间，
  包括开始和结束记录之间的空档。另记每次迭代 CPU 发出全部工作所用的时间（enqueue）。
  开机前定的判据：enqueue 达到墙钟的 90% 即为 launch-bound，记入 `launch_bound_rows`，
  涉及这些行的格在解读时单独标注，判读规则本身不变。逐卡记录输入、有效投影、DP padding、零写和对齐槽位。
  回放范围到 gate/up 输出为止。静态路由、投影对齐预计算、正确性重排在计时外。

## 事前预测

两版预测都出自同一个冻结的预测器，分别计分：
- **开机前版**（`preboot_prediction/`）：情景参数，已随提交公开封存。它假设每次启动 40 µs、每次迭代固定 150 µs，
  对应的是即时启动的实现；消费者改为 graph 回放后，这些固定开销不再成立，所以它主要检验占位参数。
- **标定版**（`prediction_calibrated/`）：参数来自本机微基准，在 prepare 时封存，早于准入和计时。
  它检验的是"给定本机实测参数，模型结构能否预测边界"。

主 R1 只检查四条网格主对照。宽区间命中要求完整的实测严格交叉落在区间内并且方向相同，单列为
`hit_loose`。严格预测使用原来的一档容差。实测覆盖不全或方向未定不可判中。

主 R1 有一个已知的盲区：预测本身在交叉处有持平档，实测与预测逐格相同时，主 R1 给出的是"不判"。
因此开机前另登记两个次要指标（`r1_secondary`），对两版预测都计算：
- **逐格一致率**：已判定的格里，预测类别与实测类别相同的格数；另报双方都是明确胜负的格里方向相同的格数。
- **区间规则**：交叉区间取"一侧最后一个明确胜"到"另一侧第一个明确胜"，持平和未定档可以落在区间内。
  实测区间落在预测区间内记 `contained`，有重叠记 `overlap`，不相交记 `disjoint`；
  双方都无交叉且赢家相同记 `agree_none`；一方有交叉一方没有记 `miss_existence`；方向相反记 `miss_direction`；
  明确胜负交错出现或没有明确格时记 `unscorable`。
  **有缺测格时不判"无交叉"：**已测的格全是同一侧赢、但这条序列里还有没测完的格时，实测区间记为
  `incomplete`，这一项的结论也是 `incomplete`，因为交叉点可能就在缺测的格里。两侧都有明确胜的格时，
  区间照常计算，缺测格可以落在区间内。
主 R1 与次要指标都如实报告，不用次要指标替换主 R1 的结论。

P0 在 33 格全部选 SLT，R2 仅报离线选择和测后 regret，同选择规则的相等结果按原值报告；
标定版的选择另列为 `P0_CALIBRATED`。

本设计的封存用冻结的 `predict.py` 和 `scenario.json` 重算，覆盖三个宽度：231 行预测和 132 个比较类别
与 2026-11-21 的原封存完全相同。每个宽度各有四条预测；Qwen3.5 宽度的 q4 宽区间是 16–512，其余与 GLM 相同。
N=1024 的 M32、M64 两格以前用别的 kernel 测过（其中 M64 当过校准点），逐格一致率另报一份不含这两格的数。

scenario 参数是对着历史 CUTLASS J1 的 P1 M512 实测拟合的，本轮网格的 D/BNEW 改用 Triton 逐源消费者。
两者 tile 几何相同（64×128·g，split-K 1 或 4），重读次数 ρ 和字节账本不变，但 kernel 效率不同。
论文须写明：预测按 CUTLASS 标定，实测用 Triton；`D_Q4_G2` 的"P2 固定策略"只指 q4/g2 这组参数，
不再是 P2 的那个 kernel。

## arm 状态列

`design/arms.csv` 的 `gpu_status` 只说明该 arm 的计算 kernel 在本轮之前是否在 A100 上跑过：
`gpu_run` 只有 BOLD（CUTLASS 64×128）和 NATIVE（vLLM fused MoE Triton kernel）。
三个 Triton D arm、BNEW、SLT 和全部快照直读/staged arm 都是 `new`。各 arm 的主机侧调度代码
都是本轮新写的，一律以冒烟和准入为准。
