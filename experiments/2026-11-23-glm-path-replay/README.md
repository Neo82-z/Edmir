# GLM 直读与 staged 回放

当前执行入口见 [START_HERE.md](START_HERE.md)，方案见论文仓库 `latex/review/FINAL_12H_PLAN_ZH_2026-11-23.txt`。设计表和预测在本地生成，历史三模型
的预测器、六个冻结形状和数据保持封存。

本地验收运行 `python -B -m unittest discover -s tests -v`（需要 CPU PyTorch）。
本机没有 CUDA A100 环境，设备冒烟、SM80 二进制、cuBLASLt 调优、真实捕获及
全部正式时延仍需在目标节点验证。预先准备模型、tokenizer、镜像和工具仍沿用已有离线回执。
