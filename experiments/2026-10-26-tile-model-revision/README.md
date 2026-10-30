# Tile 几何、服务分层和开发划分的本地交付

2026-10-26。本目录交付可运行的结构表示和开发验证设计，没有训练新模型、没有采集 GPU 数据。正文方案见 [NEUSIGHT_ADAPTATION_PLAN.md](../../paper/NEUSIGHT_ADAPTATION_PLAN.md)。

## 已落实的三个方向

| 方向 | 实现 | 验收范围 |
|---|---|---|
| 执行几何 | `tile_execution_model.py` 复用执行账本，显式保留 M/N tile、q、K 分片、有效比例、轮次/尾部与 epilogue 顺序 | 两个旧特征碰撞可区分；边缘 tile 独立枚举和整轮边界通过；不宣称实测耗时不同 |
| 服务与依赖 | 服务缺项保留 null；同身份下完整节点耗时可做条件 DAG 汇总 | 缺项、非法值、环、身份不符拒绝；重排稳定；CPU/资源竞争未识别时程序墙钟预测仍为 null |
| 结构消融与划分 | A/B/C 方案、强 host、固定预算、四折整形状开发划分 | 所有候选和重复属于同一分区；候选区组完整性与重复检查通过；不是独立测试 |

最终输出在 [validated/plan.json](validated/plan.json) 和 [validated/execution-structures.json](validated/execution-structures.json)。`draft/` 保存最初开发运行，不能作为当前源码的验收记录。`validated` 表示本地结构检查通过，不表示 GPU、预测误差或编译器准入通过。

范围保持原有四个 P8 形状，每点三种 host 加十种 fused，共 52 个配置。三个区组对应 156 次计划运行，全部尚未测量；它们是原八点采集包的重点子集，不新增必测矩阵。cuBLASLt 算法身份及内部 tile 未知，记录为未知；库候选不会继承 CUTLASS 资源估计。

## 本地验证

```sh
# 从代码仓库根目录执行；输出目录必须是新的
python3 EDM-IR/tools/prepare_tile_model_revision.py --out /tmp/edmir-tile-review
python3 -m unittest discover -s EDM-IR/tests -p tile_execution_model_test.py -v
python3 -m unittest discover -s EDM-IR/tests -p resource_geometry_test.py -v
python3 -m unittest discover -s EDM-IR/tests -p execution_evidence_test.py -v
```

本次结果：新增 9 项、现有资源计数 6 项、执行账本/比较证据 18 项，共 33 项通过。计划生成 0.185403 秒（本机单次运行，不是性能保证）。整个流程只需 Python 标准库，不调用拟合、CUDA 或远程服务器。

旧几何审查绑定的五个输入/源码哈希保持一致；两个已封存 GPU tar.gz 的 SHA-256 均匹配各自记录。未改变归档八维评分、历史标签和已交付 PDF。新增代码不进入自动选择路径。

## 仍需要什么

1. 用既定 GPU 采集核对资源、阶段/操作耗时、新运行时和强 host 排序。不得从总耗时反算全部服务项。
2. 只在开发集确定 A/B/C 的最终时间公式及参数数量，同样计入标定和选择预算。当前计划明确标为未冻结，不能充当独立测试预注册。
3. 冻结公式、参数、形状和选择预算后再测未用过的形状；选择区组与验证区组分离。旧成绩与本轮结构检查不证明泛化或净收益。

合成单位节点时长得到的旧图 33 级和新图 8 级仅校验依赖结构；不能写成 4.125 倍加速或真实关键路径变化。
