# 2026-11-27 上机顺序

GPU 时间约 5.5 小时。先做便宜、能决定方向的，再封存，再确认。每一步后面是预计的 GPU 时间。

## 0. 搭建（约 40 分钟，传输期间并行做 gate 和冒烟）

1. GPU 节点上（带 agent 转发登录）：`node/bootstrap.sh`。它从 CPU 节点拉资产和 11-26 的采集，核对哈希，加载镜像，铺好目录。
   小文件先到，镜像一到就可以开始 gate。
2. 本机：`node/push_code.sh shadeform@<GPU>`，把节点代码带到当前 main 并逐文件核对。
3. 节点上放好 `env.sh` 和 `node/` 里的脚本，然后依次：宿主机 gate、容器内 gate、预检、`smoke.sh`、
   三个模型的原生调优、`pre.sh`（微基准、标定版预测、SLT 调优）。
   这些都绑定机器，必须在新节点上重跑。11-26 的 `RUN.json` 由 bootstrap 拷来，只用来核对快照身份。

## 1. 探索（约 60 分钟）

用 `node/run_exploratory.sh`。

| 序 | 内容 | 命令要点 | 时间 |
|---|---|---|---|
| 1 | 探针尺寸扫描 | `MODE=sizes TAG=-sizes ... probe` | 2 分钟 |
| 2 | prefill 候选 O、N，连同已登记臂和原生 | `snapshots GLM_PREFILL QWEN35_PREFILL DSV2L_PREFILL`，臂：`NATIVE,D_Q1_G2_R,D_Q4_G2_R,BNEW_Q1_G2_R,BNEW_BLAS_R,BNEW_BLASC_R,BNEW_GATHER64_Q1_G1_R,BNEW_GATHER64_Q1_G2_R,BNEW_GATHER128_Q1_G1_R` | 5 分钟，首次要调通 |
| 3 | decode 上拆开打包的贡献，并实测去掉释放 barrier | `snapshots` 三个 decode，臂：`NATIVE,D_Q1_G2_R,D_Q4_G2_R,D_GATHER16_Q4_G2_R,D_PACK16_Q4_G2_R,D_GATHER16_Q4_G2_NR_R,D_PACK16_Q4_G2_NR_R,BNEW_BLAS_R` | 5 分钟 |
| 4 | 大 M：轮转 tile 和按源轮转的拉取 | `variants GLM_M0064 GLM_M0256 GLM_M1024 QWEN35_M0256`，臂：`SLT,PULL_SLT,PULLR_SLT,D_ALL32_Q1_G4,D_ALLR32_Q1_G4,D_ALLR_Q1_G2,BNEW_Q1_G2` | 5 分钟 |
| 5 | 中等 M 上把边界右推 | `variants` GLM、DSV2-Lite 的 M=32、64 和 Qwen3.5 的 M=128、256，臂：`SLT,D_MRG16_Q4_G2,D_MRG32_Q4_G2,D_MRG32_Q2_G2,D_MRG32_Q4_G4,D_ALL32_Q2_G4,D_MRG16_Q4_G2_NR,PULLR_SLT` | 6 分钟 |
| 6 | 换层：每个模型再采两层，重复 decode 和 prefill 的比较 | `node/capture_layer.sh` 六次，再 `layers <模型> <层> <快照>` | 约 35 分钟 |

第 2–5 步有结果后，有效的候选各补两个进程凑成三对。

## 2. 封存（不占 GPU）

按第 1 步的结果定稿 `arms.json`（是否加入 prefill 候选和 H5），提交并推送，记下提交号。

## 3. 确认性实验（约 2 小时）

节点上 `experiments/2026-11-27-confirm/run_queue.sh`，放在 tmux 里跑；跑完用 `judge.py` 判定。

## 4. 收尾

- 结果回传本机：GPU 节点上传每条连接约 170 KB/s，要用几十条并行连接；小文件直接 rsync。
- 新的采集（换层）拷到 CPU 节点保存，核对哈希。
- 确认全部落盘后再关机。
