# 执行账本与证据审阅包

先读 [机制与接口说明](EDM-IR/docs/EXECUTION_EVIDENCE.md)，再看
`review/summary.json` 和 `review/mechanism_diagnosis.json`。`review/` 的五个形状文件
包含逐候选账本、历史观察和未测候选。源码位于 `EDM-IR/`。

本包没有训练或运行 GPU，也不宣称 M512/大 N 的微秒级成因已经得到验证。
证据不足返回 choice=null；这不是新的预测准确率成绩。

在解压目录执行（Python 标准库即可）：

```sh
python3 verify_bundle.py
python3 EDM-IR/tools/explain_execution.py --m 512 --k 4096 --n 512 --environment planned-a100-p4 --out /tmp/m512-execution-new.json
python3 EDM-IR/tools/prepare_execution_review.py --out /tmp/execution-review-new
python3 -m unittest discover -s EDM-IR/tests -p execution_evidence_test.py -v
```

输出路径必须不存在。完整 40 项验收还包含既有 backend/schedule/resource 检查，
其中 backend 检查需要本机 C++ 工具链；它们不执行 CUDA kernel。
旧模型及两个推断命令保留作对照，使用时另需 NumPy；不需要重新训练。
其 model_preference 仅供参考，不能代替 decision.choice。

manifest.json 固定包内文件；verification.json 记录原仓库验收及 353 个旧文件未变。
本包是本次工作的独立审阅快照，未包含那 353 个文件的全部体积、旧 GPU 输入或完整历史仓库。
