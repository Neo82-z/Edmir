# 当前执行口径：八卡 A100 上的单 GLM 投影

以此目录的 `design/queue.csv` 和论文仓库 `latex/review/FINAL_12H_PLAN_ZH_2026-11-23.txt`
（11-23 修订方案）为准；`2026-11-20-final-round/` 只作为只读的旧包和评分器来源。真实模型只有
GLM-4.7-Flash。FP16 网格是其 gate/up 几何的均衡代理；BF16 回放来自两个固定路由快照。
硬件单机 8×A100 SXM4 80 GB，NVSwitch 全 NV12。历史数据和冻结评分器使用原有契约。

## 前置顺序

所有设备命令都在目标节点运行。先备好已校验的模型、CPU 回执、源锁、
vLLM 0.20.1 环境和固定 CUTLASS 工作树。各设备作业回执写入新的实验目录。
启动后依次执行 `design/queue.csv` 的前置任务：

1. 用旧执行包中只读的 `scripts/hw_gate.py --load-test --out gate.json` 验证拓扑、负载、P2P 和功率。
2. `CUTLASS_ROOT=... ./build_lt.sh /abs/path/libedmir_glm.so` 编译 SLT 与桥接所用库。冒烟要用它，所以先编译。
3. 合成冒烟 `torchrun --standalone --nproc-per-node=8 replay.py --library /abs/path/libedmir_glm.so smoke --gate gate.json --out smoke.json`。
   - BF16 五臂使用真实 K/N，验证 symmetric rendezvous、Triton 对端读、AllToAll、原生
     pynccl all-gatherv 和 vLLM 投影调用。
   - FP16 网格七臂（D/BNEW 的 q1/q4、SLT、D_Q1_G1、BOLD）在 M=1 和 M=64 上各跑一次准入，
     覆盖 FP16 Triton、cuBLASLt heuristic 和 CUTLASS 桥接。
   - 某个 arm 抛异常时记为 runtime_error，其余 arm 照常跑完；有任何失败，命令以非零退出并列出失败项。
   - `prepare` 要求 BF16 五臂全部通过，并要求两个形状上的七个网格 arm 都有记录，
     且回执与当前 replay.py、gate_up.py、native.py、balanced.py 和库文件哈希一致。
     网格 arm 失败不阻止 prepare，由正式准入裁决（BNEW 失败会开启桥接）；能修就先修再重跑冒烟。
   - 先通过这里，再加载模型。
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
7. 用新采集器重新生成固定镜像的 CPU 回执。原 `run_real_moe.py` 在已有参数上加
   `--mode routes --snapshot-replay --native-tune <配置JSON>`。首个纯 prefill 和
   第八个纯 decode 各保留 rank 0–7 的原生 launch tensors 与 BF16 输入。若调优未通过，
   仅可登记为默认原生诊断，采集与分析须保留默认配置来源。

## 统一准备与运行

统一设置 `--run <new-run> --library <libedmir_glm.so> --snapshots <capture>/workers/snapshots`
作为 `replay.py` 的全局参数，放在子命令前。`verify`、`prepare`、`admit`、`admit-arm`、`time`、`time-all`
缺 `--snapshots` 时直接报错。`time` 在占用进程编号之前先核对快照目录和回执哈希，
参数写错不会消耗正式进程的唯一一次机会。

- `verify` 在 CPU 上检查两个快照、哈希和保存的原生对齐。
- `prepare --gate gate.json --smoke-receipt smoke.json --slt-receipt slt.json --microbench microbench.json --prediction preboot_prediction --native-tune <实际捕获配置>`
  一次登记 33 个均衡 seed、2 个快照以及全部 build。对于已确认的 default 捕获，
  省略最后一个参数。后续不得覆盖 RUN 或其中冻结的回执。
- 顺序执行父进程 `admit --unit GLM_PREFILL`、`admit --unit GLM_DECODE`、`admit`，
  然后 `lock-admission`。快照的每臂独立 torchrun，一个 arm 失败不会阻塞其他 arm。
  均衡默认每个 seed 一个进程，异常后由独立子进程完成剩余 arm。完整主准入记录后才锁定。
- 若任意均衡 BNEW seed 不合格，开启同一 11 格的 g1 桥接。直接臂使用 64×128 tile，
  BOLD 使用逐对 NCCL 搬运接既有 CUTLASS 64×128 kernel。两臂算法和传输共同变化，
  只作为弱传输桥接单列。匹配 B-a 的缺测格仍为空。桥接门在正式计时前锁定。
- `time-all` 按队列运行，或按进程 ID 用
  `torchrun --standalone --nproc-per-node=8 replay.py <全局参数> time --process <id>`。
  已启动的正式进程不重跑。失败、超时和剩余缺测都保留。必要的复跑须另建 run，
  完整保留首次运行，分析不跨 run 拼配对。
- `python analyze.py --run <run> --out <results>` 输出有效配对、交叉方向、离线选择
  和论文 JSON。队列共 78 个正式进程、390 个主臂计时。桥接只在触发后多测最多 132 个臂。
- BF16 direct 和 BNEW 使用同一逐源、跨本地专家的 tile 工作表。q1 每个非空源一次
  Triton 投影，q4 每源另有一次 FP32 归并，最终输出 BF16。原生直接使用捕获的
  aligned IDs、expert IDs 与 padding extent。统一输出重排只在准入执行。
- Direct 的读取前发布 barrier 与读取后的释放 barrier 都在计时区，保护下一轮复用。
  所有臂同步后记录 max-rank 墙钟和 CUDA event，CUDA event 视作设备时间线经过时间，
  包括开始和结束记录之间的空档。逐卡记录输入、有效投影、DP padding、零写和对齐槽位。
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
