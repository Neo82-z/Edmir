# Experiment：tinygrad-notes Primitive 复现

## 问题

我能否把 tinygrad 风格的 high-level tensor operations 解释为 primitive tensor transformations 的组合？

## Scope

用小规模 NumPy / Python 代码复现：

- reshape
- expand / broadcast
- permute
- sum / reduce
- elementwise add / mul
- GEMM as broadcast multiply + reduce
- gather as one-hot mask + reduce
- scatter_add as mask + reduce
- fake-device allreduce as reduce_scatter + allgather

## Non-Goals

- 不做 autograd。
- 不做 scheduler。
- 不做 UOp renderer。
- 不做 SASS backend。
- 不做 performance claim。

## Files

```text
code/
  mini_tensor.py
  puzzles.py
results/
  notes.md
```

## Observations

1. 
2. 
3. 

## 与研究主线的关系

这个实验用于建立 semantic layer：理解 collective communication 和 tensor operations 如何分解为 primitives，然后再 lowering 到 hardware-specific implementations。
