#!/usr/bin/env python3
"""Count how each kernel fetches global memory, by compiling it for SM80 without a GPU.

The compiler emits a 16-byte load (ld.global.v4.b32, or cp.async of 16 bytes inside a
pipelined loop) only when it can prove the address 16-byte aligned. An address taken from a
pointer table, or built from a row offset that was itself loaded, is not provably aligned,
and the kernel then fetches two bytes per instruction. This script prints, per kernel, the
global loads and stores in the PTX by width, and the depth of the cp.async pipeline.

  python3 ptx_loads.py

Run it in the locked image with the experiment package (experiments/2026-11-23-replay) on
PYTHONPATH. Needs no GPU and launches nothing.
"""
import collections
import re

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as compile_source, make_backend
from triton.runtime.jit import MockTensor, create_function_from_signature

TARGET = GPUTarget("cuda",80,32)
BACKEND = make_backend(TARGET)


def ptx(kernel, *args, **options):
    """The PTX of one specialization; tensor arguments are given as dtypes (assumed 16-byte aligned)."""
    binder = create_function_from_signature(kernel.signature,kernel.params,BACKEND)
    bound,specialization,parsed = binder(*map(MockTensor.wrap_dtype,args),**options)
    parsed,signature,constexprs,attrs = kernel._pack_args(BACKEND,options,bound,specialization,parsed)
    return compile_source(ASTSource(kernel,signature,constexprs,attrs),target=TARGET,options=parsed.__dict__).asm["ptx"]


def census(text):
    counts = collections.Counter()
    for match in re.finditer(r"^\s*(?:@!?%p\d+\s+)?(ld|st)\.global((?:\.c[ag])?(?:\.v[24])?\.[bfus]\d+)",text,re.M):
        counts[match[1]+match[2]] += 1
    for match in re.finditer(r"cp\.async\.cg\.shared\.global.*?(0x[0-9a-f]+|\d+),\s*%",text):
        counts[f"cp.async {int(match[1],0)} bytes"] += 1
    depth = max(map(int,re.findall(r"cp\.async\.wait_group\s+(\d+)",text)),default=None)
    return dict(sorted(counts.items())) | ({"pipeline depth":depth} if depth is not None else {})


# The two peer-read kernels of the registered microbenchmark (replay.py, microbench), verbatim.
@triton.jit
def copy(A, B, n: tl.constexpr, ctas: tl.constexpr, block: tl.constexpr):
    base = tl.program_id(0)*block+tl.arange(0,block)
    for offset in range(0,n,ctas*block):
        ix = offset+base
        value = tl.load(A+ix,ix<n,0,cache_modifier=".cg")
        tl.store(B+ix,value,ix<n)


@triton.jit
def gather7(PTRS, B, chunk: tl.constexpr, per_peer: tl.constexpr, block: tl.constexpr):
    cta = tl.program_id(0)
    peer = cta%7
    base = (cta//7)*block+tl.arange(0,block)
    src = tl.load(PTRS+peer).to(tl.pointer_type(tl.float16))
    for offset in range(0,chunk,per_peer*block):
        ix = offset+base
        value = tl.load(src+ix,ix<chunk,0,cache_modifier=".cg")
        tl.store(B+peer*chunk+ix,value,ix<chunk)


def main():
    from gate_up import project
    import variant_kernels as vk

    f16,f32,i32,i64 = torch.float16,torch.float32,torch.int32,torch.int64
    rows = [("microbenchmark, one peer (copy)",ptx(copy,f16,f16,32*1024**2,432,1024,num_warps=4)),
            ("microbenchmark, seven peers (gather7)",ptx(gather7,i64,f16,4*1024**2,64,1024,num_warps=4)),
            ("registered consumer, q=1, 64x256",ptx(project,f16,f16,f16,i32,64,2048,1536,1,64,256,32,num_warps=8))]
    for unit in (1,vk.WIDE):
        rows += [(f"one-launch consumer, q=4, 16x256, unit {unit}",
                  ptx(vk.project_rows,f16,i64,i32,f16,f32,i32,64,2048,1536,4,16,256,32,unit,num_warps=4)),
                 (f"one-launch consumer, q=1, 32x512, unit {unit}",
                  ptx(vk.project_rows,f16,i64,i32,f16,f16,i32,64,2048,1536,1,32,512,32,unit,num_warps=8)),
                 (f"row copy, unit {unit}",ptx(vk.copy_rows,f16,i64,f16,2048,1024,unit,num_warps=4))]
    rows.append(("block copy",ptx(vk.copy_block,f16,f16,1024*2048,432,1024,num_warps=4)))
    for name,text in rows:
        print(f"{name:44s} {census(text)}")


if __name__=="__main__":
    main()
