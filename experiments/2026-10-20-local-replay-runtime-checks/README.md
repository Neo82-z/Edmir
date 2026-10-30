# 历史数据回放与本地运行检查

日期：2026-10-20。未租用 GPU，未产生新的 GPU 性能数据。

这轮保持四个 GPU 形状、候选实现、随机运行顺序、输入数据和计时口径不变。
目标是复查预测器在历史数据上的表现，并在本地处理依赖遍历和进程超时问题。
没有重新拟合模型来追求更好的历史分数，也没有把本地 CPU 耗时换算为 A100 加速。

## 历史数据结果

以 `../2026-10-20-offline-predictor/data` 为数据源，按照原有
`../2026-10-20-splitk-local-revision/design.json` 重跑。
新的 `historical_replay.json` 与原 `results.json` 解析后的 JSON 完全相同，
包括模型参数、逐点选择、汇总指标和数据/源码身份。见 `replay_comparison.json`。

P=4 的 41 个形状使用原有按组留出协议。选择损失定义为
`所选实现的实测耗时 / 已测候选中的最低耗时 - 1`：

| 方法 | 平均选择损失 | 最坏选择损失 | 选择结果慢于 host 的形状数 |
| --- | ---: | ---: | ---: |
| 训练集选出的固定策略（q=4） | 5.43% | 18.18% | 3 |
| 原相对模型，加 grouped-N 适用范围限制 | 3.54% | 29.24% | 2 |
| 修订分解模型：wave 特征 + pairwise 目标 | 9.88% | 39.57% | 0 |

修订模型减少了慢于 host 的选择，但过度回退仍然明显，未通过原先的准入条件。
这批数据已经参与开发，不能视为新的独立验证。P=8 的旧诊断也原样复现。

四个重点形状的历史回放如下，均为 P=4，元组顺序为 **(M,K,N)**：

| 形状 | 原相对模型 | 修订模型 | 历史已测最优 | 修订模型耗时 / 最优耗时 |
| --- | --- | --- | --- | ---: |
| (64,1024,512) | split-K=2 | split-K=4 | split-K=2 | 0.3492 / 0.3426 ms |
| (64,4096,2048) | split-K=4 | host | host | 0.6453 / 0.6453 ms |
| (64,8192,512) | host | host | split-K=8 | 0.8427 / 0.6556 ms |
| (256,8192,512) | split-K=4 | host | split-K=4 | 1.2688 / 1.0741 ms |

这些耗时属于历史 `host_nccl_cutlass_sm80_per_source_v1` 口径。
历史数据没有本轮 prepared 执行器和 merged host 的实测结果，不能替它们打分。
当前优先检查大 K 下 fused 准入和 host 回退的判断，再检查 q 排序；
保持四个形状，不增加调参搜索范围。

## 遍历修改及本地性能

`EDM-IR/runtime/src/prepared_execution.cpp` 原先对每条依赖在已收集的前驱中线性查找。
在大量前驱汇入同一节点时，去重可能退化为平方级工作量。
现在按目的节点遍历，并用源节点标记去重，保留前驱首次出现的顺序，
使这部分工作量为 O(V+E)。V 是操作数，E 包含重复依赖。

prepared 模式也不再额外构造仅供 legacy 路径使用的前驱表。
执行阶段的 GPU API、参数和依赖等待顺序保持一致。
拓扑验证仍然保留，循环依赖在准备阶段被拒绝。
这里的复杂度结论仅指此次替换的去重逻辑；验证中的哈希容器依然存在。

用有效的 barrier/token 程序构造两个高入度节点、倒序前驱和重复依赖，
再加回边检查循环拒绝。Apple M4，Apple Clang 17，`-O2`，
旧、新版交替运行各三次，以下为准备阶段耗时中位数，包含程序验证：

| 操作数 | 含重复项的依赖数 | 修改前 | 修改后 |
| ---: | ---: | ---: | ---: |
| 1,000 | 3,992 | 1.460 ms | 0.776 ms |
| 10,000 | 39,992 | 57.664 ms | 8.459 ms |
| 40,000 | 159,992 | 813.863 ms | 41.197 ms |

原始样本在 `graph_scaling.json`。大图测试用于发现扩展时的遍历退化，
不是新增的 GPU 实验形状，也不代表当前小图或 GPU 推理有同等收益。

48 个生成程序分别对 legacy/prepared 路径执行 1,000 次，共 96,000 次
CPU API 替身执行检查；每轮调用数量保持一致。
另对正常及倒序设备映射比较完整的 API 参数和顺序。
计数模式不保存调用字符串，避免检查器自己不断积累日志。
这些检查不模拟 GPU 数值运算、真实异步执行或通信进度。

## 运行保护

`EDM-IR/tools/run_selector_gpu.py` 现在对所有外部命令设置有限等待时间：

| 阶段 | 超时 |
| --- | ---: |
| git、动态库检查、设备检查 | 60 秒/命令 |
| CMake 配置 | 300 秒 |
| 编译 | 1,800 秒 |
| CTest 整体 / 单项 | 600 秒 / 30 秒 |
| 每个 GPU 测量进程 | 180 秒 |

超时或键盘中断会终止整个 POSIX 进程组，终止后的等待也限制在 5 秒内。
输出写入文件，避免后代进程持有管道导致收尾无限等待。
如果驱动或操作系统处于不可中断状态，仍需人工处理；本地检查不能排除这种情况。

测量开始前保存 `progress.json`，同时记录 stdout/stderr。
进程失败、结果校验失败或设备身份变化会停止收集，并写入 `failure.json`。
只有通过校验的数据才写入 `measurements.jsonl`，并执行 flush/fsync。
恢复时仍校验包、构建、环境和旧结果，只启动尚未完成的任务。
JSON 状态文件通过临时文件替换，降低读取到半个状态文件的风险。
`failure.json` 保留上次失败信息；是否完成以 `progress.json` 和最终结果为准。

新增的三个 Python 测试实际启动本地子进程，覆盖：

- 超时保存输出，并终止忽略 SIGTERM 的后代进程。
- 非零退出和超时都留下失败信息，不生成测量行。
- 两项任务中第二项失败后立即退出，恢复时只启动第二项。

测试 GPU 环境及数值校验使用替身；这是收集器控制流程验证。
原有包身份、完整数值检查、计时样本和环境检查另有测试。

## 检查记录及版本

- Python 定向检查：11 项通过，见 `checks/python_targeted.txt`。
- C++：97 项现有检查通过；新增大图检查在修正测试构造中的 token 输出后通过。
  见 `checks/cpp_tests.txt`、`checks/cpp_graph_test.txt`。
- API 完整记录：`checks/cpp_api_traces.txt`；重复执行记录：`cpu_repetition_checks.json`。
- 旧源码 109 个文件保存在 `before_source/`，清单为 `before_source_manifest.json`。
  原 metadata 优化包已用这份源码重新核验，未修改原包。
- 新包：`gpu_pilot/`，SHA-256 为
  `20e4df1a58ab838541d4212f4d99035b6312cf7ce93ae348327c138e2df4d1c5`。
  仍为四个形状、48 个程序、216 次进程运行；全部输入和参考输出哈希不变。
  尚未在 GPU 上构建或执行。

从仓库根目录复查：

```sh
python3 -m unittest discover -s EDM-IR/tests -p 'selector_*test.py' -v

python3 EDM-IR/tools/evaluate_splitk_revision.py \
  --dataset experiments/2026-10-20-offline-predictor/data \
  --design experiments/2026-10-20-splitk-local-revision/design.json \
  --out /tmp/edmir-historical-replay-new.json

cmake -S EDM-IR/runtime -B /tmp/edmir-host-check -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/edmir-host-check -j 4
ctest --test-dir /tmp/edmir-host-check --output-on-failure --timeout 30
```

Python 包检查需要 NumPy。完整 CPU API 对比还需官方 CUDA/NCCL 头文件，
配置方法见上一轮 `../2026-10-20-host-metadata-optimization/README.md`，
把实验目录切换到本轮 `gpu_pilot/`。

进入服务器前保持新包固定。服务器上先做环境、构建和正确性检查，
任何一项失败即停；通过后才完成这四个形状的计时。
