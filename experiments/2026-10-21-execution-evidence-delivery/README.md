# 执行账本、错误机制与选择证据交付

本轮按三项要求修改代码，暂停继续拟合：

1. 每个候选生成可追溯的执行账本，保留真实 Program、依赖、tile、CTA、K 分片、
   工作集、逻辑读写和准备/计时边界。公式绑定已审阅的源码快照，源码变化要求重新核对。
2. 分别交付 M512、大 N 和 grouped-N 跨 M 的已确认事实、未确认解释，以及能区分
   这些解释的最小对照。沿用已有形状和执行包，不启动服务器。
3. 选择报告区分静态事实、历史观察、拟合参考和未知成本。缺少证据返回 `choice:null`；
   不把未知成本填成零，不用候选位置打破平局，也不默认回退 host 来宣称选对。

**完成的是执行表示与选择依据的修正，不是未知形状耗时预测已经通过验收。**
没有新训练、GPU 执行或独立性能标签。旧模型、第二轮审查和已交付 GPU 包保持原样。

## 阅读入口

- [完整说明与错误机制](../../EDM-IR/docs/EXECUTION_EVIDENCE.md)
- [可执行账本](../../EDM-IR/edmir/compiler/execution_ledger.py)
- [证据比较逻辑](../../EDM-IR/edmir/compiler/evidence_selection.py)
- `review/summary.json`：五个重点形状和全部旧形状的本地遍历结果。
- `review/mechanism_diagnosis.json`：逐问题的事实、假设、反证条件和最小测量。
- `verification.json`、`final-checks/`：本地验收、接口兼容检查与旧文件哈希核对。
- `execution-evidence-review.zip`：便于交接的独立源码、证据和说明包；文件身份见 `delivery.json`。

## 本轮重点形状

顺序统一为 `(P,M,K,N)`。前四项沿用原优先形状，第五项是已在处理的 M512 问题。

| 形状 | 历史已测候选最快者 | 其历史完整程序中位数 | grouped-N 历史覆盖 |
|---|---|---:|---|
| (4,64,1024,512) | q2 | 342.6 μs | 有 |
| (4,64,4096,2048) | host | 645.3 μs | 无 |
| (4,64,8192,512) | q8 | 655.6 μs | 有 |
| (4,256,8192,512) | q4 | 1074.1 μs | 无 |
| (4,512,4096,512) | q2 | 1005.9 μs | 无 |

这些是已有开发数据的观察，不是新预测或独立测试成绩。每个 JSON 保留已测候选和
未测候选名单、CSV 路径/哈希；生成器检查规范化数据及其 P4 原始 CSV 的哈希。
历史标签未与当前源码/硬件重新匹配，因此没有将它们喂入新的证据选择接口冒充当前证据。

M512 的 q2/q4 区别已能列出：64/128 个 CTA、64/32 次 K 主循环、每输出 tile
2/4 次有序 epilogue，以及相对 q1 的 1/3 MiB 额外输出读写。这是单 GEMM 计数，
不能由此直接推导整个程序 q4 慢 43.9 μs 的分项原因。

大 N 的历史 host/q2 差为 79.5 μs。单 GEMM 的 A 唯一工作集为 0.5 MiB，
g1 逻辑 A 读取为 8 MiB；真实链路流量、缓存与 CPU 提交各占多少仍未知。
账本明确保留这些区别，不再把全部变化压到一个 grid 或拟合系数。

## 本地使用

在源码仓库根目录：

```sh
python3 EDM-IR/tools/explain_execution.py \
  --m 512 --k 4096 --n 512 --environment planned-a100-p4 \
  --families host splitk2 splitk4 --out /tmp/m512-execution.json

python3 EDM-IR/tools/prepare_execution_review.py --out /tmp/execution-review-new

python3 -m unittest discover -s EDM-IR/tests -p execution_evidence_test.py -v
```

三个入口只需 Python 标准库。输出文件或目录必须是新路径。生成审阅材料有 60 秒
工作预算；形状、P 和候选数量有显式上限，不进行超参数搜索。
另有 `PYTHONPATH=EDM-IR python3 -m edmir.cli explain-execution` 内置入口。

原两个预测命令仍可加载旧模型，但输出结构已改变：
`fitted_model_diagnostic.model_preference` 是原拟合偏好；`decision.choice` 才是
证据支持的决定。已检查旧预测数值不变，推断不调用 fit。旧模型研究代码没有删除。

## 使用边界

最终本地验收为 40 项测试通过；41 个旧形状共 451 个候选账本生成与检查约
0.15 秒。独立进程复跑结果一致，353 个旧文件哈希未变。独立压缩包包含 133 个
受清单保护的文件，约 457 KiB，已在异路径解压并用无第三方包的 Python 复现审阅结果。
这些时间是本机账本程序开销，不是 GPU 算子时间。

本地检查证明了计数、身份和决策规则的一致性，不能证明设备驻留、流水重叠或真实流量。
目前证据接口仅作研究审阅，不接入编译器默认选择。将来接入独立进程测量后，
即使各候选的样本范围分离，也只代表给定测量条件下的结果，不是未来延迟保证。

本次从 Ascend 记录借鉴的是事务几何、缓冲区生命周期、关键路径与证据身份的分析方法；
没有迁移 NPU 的耗时参数，也未认定 A100 上发生相同瓶颈。
