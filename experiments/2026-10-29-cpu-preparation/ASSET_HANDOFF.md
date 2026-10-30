# 无 GPU 资产交接

2026-10-29。资产位于 RunPod **US-NE-1** 的 500 GB 网络卷 `zyw8svwb4a`，挂载后根目录为 `/workspace/edmir-assets`。约占用 170 GiB；同区新 Pod 在创建时挂载此卷即可复用，异区需复制。只下载和做 CPU 检查，未运行 GPU 推理。

## 已完成

| 模型 | 固定 revision | 权重 | 已校验分片 |
| --- | --- | --- | --- |
| zai-org/GLM-4.7-Flash | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 62.44 GB | 48 |
| Qwen/Qwen3.6-35B-A3B | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 71.90 GB | 26 |
| deepseek-ai/DeepSeek-V2-Lite-Chat | `85864749cd611b4353ce1decdb286193298f64c7` | 31.41 GB | 4 |

78 个分片均核对文件大小和上游 LFS SHA256。模型路径为 `hf-hub/models--组织--模型/snapshots/revision`，具体绝对路径见 `ASSET_ACCEPTANCE.json`。分词器与聊天模板可离线加载，检查时禁止 remote code。

- vLLM **0.20.1** GPU 软件缓存：`software/vllm-0.20.1-cp312-linux-x86_64/`，186 wheels，共 3.74 GB；`manifests/gpu-requirements.lock` 固定版本及哈希，离线依赖解析通过。没有在当前 CPU 环境安装 GPU 引擎。
- 目标为 Linux x86_64、Python 3.12、glibc >=2.35，建议 Ubuntu 24.04。PyTorch 2.11.0 配套 CUDA 13.0.2；后续先核对驱动。不要覆盖原 A100 CUDA 12.8 基线环境。
- `vllm-image-reference.json` 保存官方镜像 tag 和 digest；**没有缓存 Docker 镜像层**。
- Nsight Systems **2026.5.1**、Nsight Compute **2026.3.0** 安装包哈希通过，解包至 `tools/nvidia/opt/nvidia/`；CLI 版本检查通过。Nsight Systems 的 CPU 报告生成、SQLite 导出及完整性检查通过。没有验证 GPU trace 或性能计数器权限。
- `sources/` 保存 DeepInfer、DeepEP、DeepGEMM、nccl-tests、TileLang 固定 commit 的源码归档；`manifests/source-downloads.json` 记录哈希。没有构建，归档不包含 Git submodules。DeepInfer 为阅读参考，不能声明已支持 H100。

## 后续入口与未完成项

GPU 实例必须另行授权启动。挂卷后可审阅并执行 `bash /workspace/edmir-assets/install_cached_gpu_env.sh --install`，它创建独立 `gpu-venv`，从本地缓存按哈希安装，检查 CUDA 可见性；不启动云实例，也不运行模型。模型加载、路由采集、正确性、P2P/通信和 profiler 权限仍需在目标 GPU 上小规模验收后才能测性能。

GLM 在 Transformers 4.57.6 的本地加载过程中出现 `incorrect regex pattern` 警告。基本分词及模板检查通过，但这不证明 token ID 与官方引擎完全一致；保持原始 tokenizer 文件及规则，不擅自设置 `fix_mistral_regex=True`。后续须核对固定 vLLM 的 tokenizer 路径及编码一致性。警告完整保存在 `manifests/tokenizer-checks.json` 和 `logs/tokenizer-checks.log`。

A100×8 的原机制实验与 H100×1/4 的应用验证继续分开。没有改评分器、训练数据或冻结六形状。下载器和检查环境放在持久卷，当前 Ubuntu 20.04 CPU 下载机本身不作为后续 GPU 镜像。

本地回收 `receipts/manifests/`、日志和小型 profiler 报告；模型权重保留在网络卷。平台停机与费用见 `RUNPOD_STATUS.md`。持久卷在停算力后仍按 $35/月收费。
