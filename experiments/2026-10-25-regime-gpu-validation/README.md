# 下一轮 A100 采集包

2026-10-25。已完成本地准备；未租服务器，未运行 GPU。先复测错误，再采集能够区分模型表达的新点。主模型 `regime_max` 与未获采用的挑战模型 `residual_depth` 已冻结，结果回来后先评分再讨论修改。

## 机器与环境

单机 **4×A100 SXM4 80GB，四卡两两 NVLink/P2P 可达**；旧环境为全互联 NV12。该包不是单卡或 PCIe 版本。建议 CUDA 12.8 开发镜像，必须有 nvcc、GCC/G++、CMake、make、git、Python 3 和可用的 NVIDIA 驱动。旧工具链为 CUDA 12.8.93、GCC 13.3；新版工具链视为新环境，不能直接声称与旧耗时可比。无需安装 PyTorch，不需下载模型；输入、CPU 参考结果、CUTLASS 与 NCCL 源码已附带。

GPU 显存需求按原四卡实验配置准备；宿主机建议至少 32 GiB RAM、30 GiB 可用磁盘，编译默认四个并行任务。全套 183 次进程的纯采集按上次记录估算约 15 分钟，另外需要上传、编译、环境检查和回收数据时间。首次预留约 30–45 分钟；这是预算参考，实际编译时间尚未在新机器验证。

## 执行身份

保留原实测的 CUDA/runtime/compiler 源码与 `resource_queues_v1` 策略；仅修改采集计划、诊断预热参数和全候选对汇总。原始执行哈希逐文件记录在 [execution-source-identity.json](execution-source-identity.json)。**本轮不验证后来修复的 runtime。** 新机器的硬件、驱动、库算法与二进制仍需记录；不可将新标签改成旧 context，也不可自动混合两次采集。

包的执行内容 SHA256：`046cbfdac9750bf63d80b5800db9a67ea90dc67ecc95b98eebdaac39423b3f4d`。这是内部 package 的哈希；外层压缩文件另有 SHA256。

## 采集顺序

| 阶段 | 配置 / 独立进程 | 内容与目的 |
|---|---:|---|
| anchors | 31 / 93 | 复测全部九对误判、M256 的 q/g 对照、质量敏感边界和 host 身份 |
| crossings | 18 / 54 | K8192/N512 的 M192、320；K3072/N512 的 M256、320、576、640，检查增长段与 q/g 交互 |
| library | 8 / 24 | M64/N512，K768、1536，各四种 host，记录完整库算法 |
| warmup | 4 / 12 | 短 K 的四种 host，单独使用 200 次预热，诊断漂移 |

具体候选与执行顺序见 [bundle/plan.json](bundle/plan.json)。前三段共 57 个主配置，20 次预热、100 次计时迭代；末段四个诊断配置，200 次预热、100 次计时迭代。每段三次完整区组，每个候选每次启动新进程，区组内顺序固定随机化。单进程超时 120 秒，剩余预算不足时不启动下一进程；已完成的原始结果逐条落盘。预算只限制采集器，不包含此前依赖编译。

先完成 anchors，检查数值正确性、拓扑、漂移和身份，再决定是否按既定顺序继续。时间不足可以只回收第一段；不必为了凑全量而删减重复进程。crossings 的点受旧模型影响，属于针对性诊断，不包装成独立泛化测试。

## 在新服务器运行

解压后进入包目录。先验证外层所有文件，再验证内部计划和输入：

```sh
python3 -S verify_delivery.py
python3 -S server.py verify
python3 -S server.py inventory
python3 -S server.py build-deps --jobs 4
python3 -S server.py build --jobs 4
python3 -S server.py preflight
python3 -S server.py run --stage anchors --budget-seconds 900
python3 -S review.py --work work --out review-anchors-01.json
```

`verify_delivery.py` 在编译之前执行；编译会在 dependencies 和 work 下生成新文件。失败时保留日志，不在付费服务器展开大范围调试。GPU 数值校验由各次执行及采集器检查完成；当前本地检查只覆盖 CPU。

第一段检查后，继续剩余已冻结阶段的命令：

```sh
python3 -S server.py run --stage crossings --budget-seconds 900
python3 -S server.py run --stage library --budget-seconds 600
python3 -S server.py run --stage warmup --budget-seconds 300
python3 -S review.py --work work --out review-all-01.json
```

采集预算用完返回 75；同一环境续跑使用原阶段加 `--resume`，不得改候选、输入、预热或包内容。重新采集采用新 `--work` 目录；一个 work 下同阶段多次结果会被 review 拒绝自动混合。只有完成全部区组且封存存在，该阶段才报告 complete。

## 回收与判读

下机前保存整个 work、review JSON 和依赖构建记录。最简单的回收方式会包含 build 产物，体积比仅收集 CSV 大，但便于追溯：

```sh
tar -czf results-01.tar.gz work review-*.json
sha256sum results-01.tar.gz
```

没有 review 文件时，直接打包 work；中途中止的数据也必须保存。下载后校验压缩包 SHA256，再释放机器。不要仅保存汇总表。原始 stdout/stderr、全部 samples、context、算法、输入/源码/二进制身份和最终 run seal 都需要保留。

[review.py](review.py) 校验原始 stdout 与逐行记录、执行身份和最终封存，评分冻结预测，不训练。带质量标记的完整配置仍输出误差，稳定子集的选择损失另列；缺失候选会明确列出，不能叫作完整候选最优。所有候选对的实测差在各阶段汇总中保存。先完整评价冻结模型，再另建实验目录修改模型。

验收细则见 [acceptance.json](acceptance.json)。库算法切换、环境差异、稳定方向错误分别解释。不能因为模型预测接近平局就删除实测差，也不能用新标签重新标定后将改好的结果称为事先预测。

## 本地已完成的检查

内部包及输入哈希通过；执行协议 7 项检查通过，包括预算退出、上下文变更拒绝续跑、完整区组、诊断隔离、全部候选对与原始日志封存。冻结程序的 CPU 编译和布局检查为 51/51。15 组输入共抽查 720 个参考元素，使用可精确计算的整数点积独立核对。见 qa 目录。

`verify_inputs.py` 需要 NumPy，仅用于本地参考结果抽查；上述服务器路径只需 Python 标准库。未完成 CUDA 编译、GPU 数值校验或新硬件耗时测量。
