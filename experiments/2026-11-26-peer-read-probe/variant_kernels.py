"""Kernel of the exploratory direct-consumer variants (2026-11-26). Not registered.

gate_up.project reads the rows of one source through one base pointer. This kernel keeps its
arithmetic (FP32 accumulation, K block bk, q partitions of K, an expert's weight per tile)
and adds one indirection per row: where the row is held, and where its output row goes.
Symmetric memory maps every peer's buffer into the address space of the reading process, so
a row is addressed by its distance from one base pointer in elements.
"""
import triton
import triton.language as tl

from gate_up import merge4


@triton.jit
def project_rows(A0, ROWOFF, OUTROW, W, C, WORK, rows, k: tl.constexpr, n: tl.constexpr,
                 q: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    tile = tl.program_id(0)
    begin = tl.load(WORK + tile*3)
    stop = tl.load(WORK + tile*3+1)
    expert = tl.load(WORK + tile*3+2)
    row = begin + tl.arange(0,bm)
    valid = row < stop
    held = tl.load(ROWOFF + row,valid,0)
    out = tl.load(OUTROW + row,valid,0)
    col = tl.program_id(1)*bn + tl.arange(0,bn)
    partition = tl.program_id(2)
    offset = partition * (k//q)
    shared = tl.arange(0,bk)
    accum = tl.full((bm,bn),0,tl.float32)
    for chunk in range(k//q//bk):
        pos = offset + chunk*bk + shared
        a = tl.load(A0+held[:,None]+pos[None,:],valid[:,None],0)
        w = tl.load(W+(expert*n+col[None,:])*k+pos[:,None],col[None,:]<n,0)
        accum = tl.dot(a,w,accum)
    tl.store(C+(partition*rows+out[:,None])*n+col[None,:],accum,valid[:,None]&(col[None,:]<n))


@triton.jit(do_not_specialize=["elements"])
def merge_parts(P, C, elements, parts: tl.constexpr, block: tl.constexpr):
    """Sum the FP32 partial outputs of `parts` K partitions (gate_up.merge4 for any count)."""
    i = tl.program_id(0)*block + tl.arange(0,block)
    total = tl.load(P+i,i<elements,0)
    for part in range(1,parts):
        total += tl.load(P+i+part*elements,i<elements,0)
    tl.store(C+i,total,i<elements)


@triton.jit
def copy_rows(A0, ROWOFF, B, k: tl.constexpr, block: tl.constexpr):
    """Pull rows to local memory: one read of each element, from wherever the row is held."""
    row = tl.program_id(0)
    held = tl.load(ROWOFF + row)
    ix = tl.program_id(1)*block + tl.arange(0,block)
    value = tl.load(A0+held+ix,ix<k,0,cache_modifier=".cg")
    tl.store(B+row*k+ix,value,ix<k)


def launch_rows(base, held, out_rows, weight, out, partial, work, k, q, bm, g, bk=32):
    """One launch over every tile in work. With q>1 each K partition writes an FP32 partial
    output and a second kernel sums them; q=4 uses the registered merge4."""
    rows,n = out.shape
    bn = 128*g
    target = out if q==1 else partial
    project_rows[(work.shape[0],triton.cdiv(n,bn),q)](
        base,held,out_rows,weight,target,work,rows,k,n,q,bm,bn,bk,num_warps=8 if bm*bn>=64*256 else 4)
    if q==4:
        merge4[(triton.cdiv(rows*n,1024),)](partial,out,rows*n,1024)
    elif q>1:
        merge_parts[(triton.cdiv(rows*n,1024),)](partial,out,rows*n,q,1024)


def launch_copy(base, held, out):
    rows,k = out.shape
    copy_rows[(rows,triton.cdiv(k,1024))](base,held,out,k,1024,num_warps=4)


def element_offsets(views):
    """Distance of each mapped view from the first one, in elements of the views' dtype."""
    size = views[0].element_size()
    ptrs = [v.data_ptr() for v in views]
    if any((p-ptrs[0])%size for p in ptrs):
        raise ValueError("peer mappings are not element-aligned")
    return [(p-ptrs[0])//size for p in ptrs]
