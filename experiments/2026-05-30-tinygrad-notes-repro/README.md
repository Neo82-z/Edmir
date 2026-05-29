# Experiment: tinygrad-notes Primitive Reproduction

## Question

Can I explain tinygrad-style high-level tensor operations as compositions of primitive tensor transformations?

## Scope

Implement small NumPy/Python reproductions for:

- reshape
- expand / broadcast
- permute
- sum / reduce
- elementwise add/mul
- GEMM as broadcast multiply + reduce
- gather as one-hot mask + reduce
- scatter_add as mask + reduce
- fake-device allreduce as reduce_scatter + allgather

## Non-Goals

- No autograd.
- No scheduler.
- No UOp renderer.
- No SASS backend.
- No performance claims.

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

## Connection To Main Thread

This builds the semantic layer for understanding how collective communication and tensor operations can be decomposed into primitives before being lowered to hardware-specific implementations.
