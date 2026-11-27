# 2026-11-27 上机顺序

GPU 时间约 5.5 小时。先做便宜、能决定方向的，再封存，再确认。括号里是预计的 GPU 时间。
所有新代码已在 CPU 节点上用 `cpu_emulation.py` 跑过表和偏移的逻辑，kernel 都在 CPU 节点上为 SM80 编译过；
没有在 GPU 上运行过的只剩 kernel 本身的行为、CUDA graph 捕获和时间。

## 0. 搭建（约 15 分钟）

1. GPU 节点上（带 agent 转发登录）：`bootstrap.sh core`。它从 CPU 节点拉代码、桥接库、CPU 回执、镜像和 11-26 的采集
   （约 14 GB），核对哈希，加载镜像，铺好目录。
2. 本机：`node/push_code.sh shadeform@<GPU>`，把节点代码带到当前 main、逐文件核对，并把节点脚本放到 `/srv/edmir-run`。
3. 节点上：`./setup.sh`。依次做宿主机 gate、容器内 gate、预检、冒烟、三个模型的原生调优、微基准、标定版预测、SLT 调优。
   这些都绑定机器，必须在新节点上重跑。11-26 的 `RUN.json` 由 bootstrap 拷来，只用来核对快照身份。
4. 同时在另一个 tmux 窗口：`bootstrap.sh models`（约 160 GB，25 分钟，低优先级）。只有换层采集需要模型，
   它在后台传，前面的步骤不等它。**计时的步骤开始前确认它已结束或暂停。**

## 1. 探索

用 `run_exploratory.sh`。每一步先看准入（所有臂 `ok`），再看时间。

| 序 | 内容 | 命令要点 | 时间 |
|---|---|---|---|
| 1 | 探针：每种读法各测窄读和 16 字节读；尺寸扫描 | `probe`；`MODE=sizes TAG=-sizes probe` | 4 分钟 |
| 2 | 网格上看宽读的效果，GLM 的 M=1、16、64、256、1024 | `variants`，臂：`SLT,D_MRG16_Q4_G2,D_MRG16_Q4_G2_W,D_MRG32_Q4_G2_W,D_ALL32_Q1_G4,D_ALL32_Q1_G4_W,D_ALLR32_Q1_G4_W,PULL_SLT,PULLW_SLT,PULLR_SLT,BNEW_Q4_G2` | 5 分钟 |
| 3 | decode：宽读、带打包、去掉释放 barrier | `snapshots` 三个 decode，臂：`NATIVE,D_Q1_G2_R,D_Q4_G2_R,D_GATHER16_Q4_G2_R,D_GATHER16_Q4_G2_W_R,D_PACK16_Q4_G2_W_R,D_GATHER16_Q4_G2_W_NR_R,BNEW_BLAS_R` | 4 分钟 |
| 4 | prefill：库 GEMM 的三种取行方式，加上宽读的就地读和先搬 | `snapshots` 三个 prefill，臂：`NATIVE,D_Q1_G2_R,D_Q4_G2_R,BNEW_Q4_G2_R,BNEW_BLAS_R,BNEW_BLASC_R,PULL_BLAS_R,BNEW_GATHER64_Q1_G1_W_R,BNEW_GATHER128_Q1_G1_W_R,D_GATHER32_Q1_G4_W_R,D_GATHER64_Q1_G2_W_R` | 6 分钟 |
| 5 | 按第 2 步定下的两三个臂铺满 33 格，每格一个进程 | `variants` 全部格子 | 20 分钟 |
| 6 | 换层：每个模型再采两层，重复 decode 和 prefill 的比较 | `capture_layer.sh` 六次，再 `layers <模型> <层> <快照>` | 约 35 分钟 |

第 1 步决定方向：
- 如果 16 字节读把多 peer 的带宽从约 24 GB/s 提到接近单 peer 的 200 GB/s，大行数上的候选就是"先拉后算"
  （`PULLW_SLT`、`PULL_BLAS_R`），小行数上是带 `_W` 的就地读，第 2–5 步照表做。
- 如果带宽没有变化，说明瓶颈不在读的宽度，大行数上只剩 `BNEW_BLAS_R` 这一条 staged 的路，第 2、5 步里去掉 `PULL*`。

## 2. 封存（不占 GPU）

按第 1 步的结果定稿 `arms.json`：每条预期写明基线臂、候选臂、预期的赢家和覆盖的单元。
单进程比值在 1.10 以上或 0.91 以下的单元才写进预期，中间的列入 `no_prediction`。提交并推送，记下提交号。

## 3. 确认性实验（约 2 小时）

节点上 `./run_queue.sh`，放在 tmux 里跑；跑完用 `judge.py` 判定。

## 4. 收尾

- 结果回传本机：GPU 节点上传每条连接约 170 KB/s，要用几十条并行连接；小文件直接 rsync。
- 新的采集（换层）拷到 CPU 节点保存，核对哈希。
- 确认全部落盘后再关机。
