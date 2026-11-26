# 节点侧脚本（2026-11-26 上机时实际使用的版本）

换一台 GPU 节点时按这个顺序重建，资产来自 CPU 节点的 `/srv/edmir-assets`：

1. 从 CPU 节点直传资产并核对哈希，加载镜像（见 `edmir-handoffs/2026-11-26-cpu-preflight/GPU-SSH-VLLM.md`；
   用 8 条 rsync 并行，约 113 MiB/s）。采集数据另在 `/srv/edmir-assets/run-2026-11-26/capture/`，
   拷到新节点的 `/srv/edmir-run/capture/` 并用 `capture.sha256` 核对。
2. 代码：`/srv/edmir-assets/code-preflight` 是 `56b3405`。拷成 `/srv/edmir-run/code`，再用本仓库当前 main
   覆盖 `code-files.txt` 列出的文件和 `experiments/2026-11-26-peer-read-probe/`，重写 `EDMIR_COMMIT`
   和 `EDMIR_CODE.sha256`，然后逐文件与 git 对照。
3. `env.sh`：容器模板（`$RUN`、`$T`、`$SN`）。`capture.sh`、`captures.sh`：三个模型的采集。
   `smoke.sh`：冒烟并汇总每个臂。`pre.sh`：微基准、标定版预测、SLT 调优。`admit.sh`、`time.sh`：正式准入和计时队列。
4. 硬件 gate、预检、冒烟、原生调优、微基准、SLT 调优都绑定机器，新节点上要重跑。

GPU 节点往外传数据时每条连接约 170 KB/s，往本机或 CPU 节点回传要用几十条并行连接。
