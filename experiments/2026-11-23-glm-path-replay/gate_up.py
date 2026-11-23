"""Per-source tiled consumer used by both peer and local FP16/BF16 inputs."""
import triton
import triton.language as tl

from snapshots import TILE_M


@triton.jit
def project(A, W, C, WORK, rows, k: tl.constexpr, n: tl.constexpr,
            q: tl.constexpr, bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
    tile = tl.program_id(0)
    begin = tl.load(WORK + tile*3)
    stop = tl.load(WORK + tile*3+1)
    expert = tl.load(WORK + tile*3+2)
    row = begin + tl.arange(0,bm)
    col = tl.program_id(1)*bn + tl.arange(0,bn)
    partition = tl.program_id(2)
    offset = partition * (k//q)
    shared = tl.arange(0,bk)
    accum = tl.full((bm,bn),0,tl.float32)
    for chunk in range(k//q//bk):
        pos = offset + chunk*bk + shared
        a = tl.load(A+row[:,None]*k+pos[None,:],row[:,None]<stop,0)
        w = tl.load(W+(expert*n+col[None,:])*k+pos[:,None],col[None,:]<n,0)
        accum = tl.dot(a,w,accum)
    tl.store(C+(partition*rows+row[:,None])*n+col[None,:],accum,
             (row[:,None]<stop)&(col[None,:]<n))


@triton.jit(do_not_specialize=["elements"])
def merge4(P, C, elements, block: tl.constexpr):
    i = tl.program_id(0)*block + tl.arange(0,block)
    p0 = tl.load(P+i,i<elements,0)
    p1 = tl.load(P+i+elements,i<elements,0)
    p2 = tl.load(P+i+2*elements,i<elements,0)
    p3 = tl.load(P+i+3*elements,i<elements,0)
    tl.store(C+i,(p0+p1)+(p2+p3),i<elements)


def launch(x, weight, out, partial, table, q=1, g=2):
    m,k = x.shape
    n = weight.shape[-2]
    bn,bk = g*128,32
    target = out if q == 1 else partial
    project[(table.shape[0],triton.cdiv(n,bn),q)](
        x,weight,target,table,m,k,n,q,TILE_M,bn,bk,num_warps=8 if g==2 else 4)
    if q == 4:
        merge4[(triton.cdiv(m*n,1024),)](partial,out,m*n,1024)
