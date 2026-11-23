"""Small independent device probes for interpreting the sealed scenario."""
import math
import statistics

from make_design import ROOT, sha

SCHEMA = "edmir.glm-microbench.v2"
PEER_CTAS = (27,54,108,216,432)
# CTAs per peer when all seven peers are read at once: 28-448 CTAs in flight.
SEVEN_PEER_CTAS = (4,8,16,32,64)
KINDS = ("peer_copy","peer_read_7","hbm_copy","launch","peer_barrier","all_to_all")


def reader_slot(rank, owner):
    """Each owner's seven readers take disjoint eighths of its buffer."""
    return (rank-owner)%8-1


def check_receipt(micro, gate_sha256, uuids, m_values):
    if (micro.get("schema")!=SCHEMA or micro.get("hardware_sha256")!=gate_sha256
            or micro.get("gpu_uuids")!=uuids or micro.get("implementation_sha256")!=sha(ROOT/"microbench.py")):
        raise ValueError("microbenchmark receipt differs from the run")
    ranks = micro.get("rank_measurements",[])
    if len(ranks)!=8:
        raise ValueError("microbenchmark requires eight rank records")
    # A probe may have failed; it must then say why. Failures explain less, they never block.
    for readings in ranks:
        kinds = {r["kind"] for r in readings}
        a2a = sorted(r["m"] for r in readings if r["kind"]=="all_to_all")
        if (not set(KINDS)<=kinds or a2a!=sorted(m_values)
                or any(not (isinstance(r.get("error"),str) and r["error"]) and
                       not (isinstance(r.get("seconds"),float) and math.isfinite(r["seconds"]) and r["seconds"]>0)
                       for r in readings)):
            raise ValueError("microbenchmark receipt is missing a probe")
    return sorted({r["kind"] for readings in ranks for r in readings if "error" in r})


def benchmark(a):
    import torch
    import triton
    import triton.language as tl
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    from balanced import cells
    from replay import gpu_session,hardware,inventory,spread,write_json

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

    @triton.jit
    def tick(A):
        tl.store(A,1)

    def probe(readings, f, **fields):
        """Measure f, or record why it failed; ranks fail together on compile or launch errors."""
        try:
            times = f()
            readings.append(dict(**fields,seconds=statistics.median(times),samples=times))
        except Exception as exc:
            readings.append(dict(**fields,error=repr(exc)))

    def measure(f):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        times = []
        for _ in range(7):
            dist.barrier()
            start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            f()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end)/1000.)
        return times

    hardware(a.gate)
    uuids = inventory()
    m_values = sorted({shape[0] for shape in cells(a.design).values()})
    with gpu_session(uuids) as (rank,device,group):
        # 64 MiB exceeds per-device L2. Each rank reads from its next NVLink peer.
        count = 32*1024**2
        src = symmetric.empty((count,),dtype=torch.float16,device=device)
        src.fill_(rank+1)
        dest = torch.empty_like(src)
        handle = symmetric.rendezvous(src,dist.group.WORLD)
        peer = handle.get_buffer((rank+1)%8,src.shape,src.dtype,0)
        readings = []

        def peer_copy(ctas):
            times = measure(lambda:copy[(ctas,)](peer,dest,count,ctas,1024,num_warps=4))
            if not torch.equal(dest[:128],torch.full_like(dest[:128],(rank+1)%8+1)):
                raise ValueError("peer microbenchmark did not read the expected source")
            return times

        for ctas in PEER_CTAS:
            probe(readings,lambda:peer_copy(ctas),kind="peer_copy",peers=1,ctas=ctas,bytes=2*count)
        # All seven peers at once, as a direct consumer reads them. Every owner serves
        # seven disjoint 8 MiB eighths, 56 MiB in total, so its L2 cannot hold them.
        chunk = count//8
        owners = [(rank+1+j)%8 for j in range(7)]
        views = [handle.get_buffer(o,src.shape,src.dtype,0).narrow(0,reader_slot(rank,o)*chunk,chunk)
                 for o in owners]
        table = torch.tensor([v.data_ptr() for v in views],dtype=torch.int64,device=device)
        expected = torch.tensor([o+1 for o in owners],dtype=torch.float16,device=device)[:,None].expand(7,128)
        def seven(per_peer):
            times = measure(lambda:gather7[(7*per_peer,)](table,dest,chunk,per_peer,1024,num_warps=4))
            if not torch.equal(dest[:7*chunk].view(7,chunk)[:,:128],expected):
                raise ValueError("seven-peer microbenchmark did not read the expected sources")
            return times

        for per_peer in SEVEN_PEER_CTAS:
            probe(readings,lambda:seven(per_peer),kind="peer_read_7",peers=7,ctas=7*per_peer,bytes=2*7*chunk)
        for kind,fn,extra in (
            ("hbm_copy",lambda:dest.copy_(src),dict(bytes=4*count)),
            ("launch",lambda:tick[(1,)](dest),{}),
            ("peer_barrier",lambda:handle.barrier(0,60_000),{}),
        ):
            probe(readings,lambda:measure(fn),kind=kind,**extra)
        # The same equal-split AllToAll that BNEW and SLT issue for each grid cell.
        for m in m_values:
            send = torch.zeros((8*m,2048),dtype=torch.float16,device=device)
            recv = torch.empty_like(send)
            probe(readings,lambda:measure(lambda:dist.all_to_all_single(recv,send)),
                  kind="all_to_all",m=m,bytes=send.numel()*2,remote_bytes=7*m*2048*2)
        ranks = spread(readings,group)
        if rank==0:
            write_json(a.out,dict(schema=SCHEMA,origin="device_microbench",gpu_uuids=uuids,
                hardware_sha256=sha(a.gate),implementation_sha256=sha(ROOT/"microbench.py"),
                m_values=m_values,rank_measurements=ranks))
