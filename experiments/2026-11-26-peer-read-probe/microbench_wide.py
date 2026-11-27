#!/usr/bin/env python3
"""Exploratory: the seven-peer probe of the registered microbenchmark, repeated with 16-byte loads.

Not part of the registered design. The registered probe (microbench.py, peer_read_7) takes each
peer's address from a pointer table and so reads two bytes per instruction (see ptx_loads.py).
This script repeats it on the same buffers, chunks, CTA counts and timing loop with a kernel that
addresses a peer by its distance from one base pointer in units of 16 elements. With --merge it
also writes a copy of the registered microbench.json whose peer_read_7 readings are replaced by
these, as input for an offline, post-hoc run of calibrate_node.py; that copy is not a receipt of
the run and its prediction is not sealed.

  torchrun --standalone --nproc-per-node=8 microbench_wide.py --gate gate.json --out wide.json \
      [--merge microbench.json --merged microbench-wide.json]

Run it with the experiment package (experiments/2026-11-23-replay) on PYTHONPATH.
"""
import argparse
import json
import statistics
from pathlib import Path

SCHEMA = "edmir.moe-path-microbench-wide.exploratory.v1"
WIDE = 16  # elements


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--merge",type=Path,help="the registered microbench.json of this node")
    ap.add_argument("--merged",type=Path)
    a = ap.parse_args()
    import torch
    import triton
    import triton.language as tl
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    from make_design import sha
    from microbench import SEVEN_PEER_CTAS,reader_slot
    from replay import gpu_session,hardware,inventory,spread,write_json

    @triton.jit
    def gather7_wide(A0, OFFS, B, chunk: tl.constexpr, per_peer: tl.constexpr, block: tl.constexpr, unit: tl.constexpr):
        cta = tl.program_id(0)
        peer = cta%7
        base = (cta//7)*block+tl.arange(0,block)
        held = tl.load(OFFS+peer)*unit
        for offset in range(0,chunk,per_peer*block):
            ix = offset+base
            value = tl.load(A0+held+ix,ix<chunk,0,cache_modifier=".cg")
            tl.store(B+peer*chunk+ix,value,ix<chunk)

    def measure(f):   # the timing loop of microbench.py
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
    with gpu_session(uuids) as (rank,device,group):
        count = 32*1024**2
        src = symmetric.empty((count,),dtype=torch.float16,device=device)
        src.fill_(rank+1)
        dest = torch.empty_like(src)
        handle = symmetric.rendezvous(src,dist.group.WORLD)
        chunk = count//8
        owners = [(rank+1+j)%8 for j in range(7)]
        views = [handle.get_buffer(o,src.shape,src.dtype,0).narrow(0,reader_slot(rank,o)*chunk,chunk) for o in owners]
        distance = [v.data_ptr()-src.data_ptr() for v in views]
        if any(d%(2*WIDE) for d in distance):
            raise ValueError("peer mappings are not aligned to the wide unit")
        table = torch.tensor([d//(2*WIDE) for d in distance],dtype=torch.int64,device=device)
        expected = torch.tensor([o+1 for o in owners],dtype=torch.float16,device=device)[:,None].expand(7,128)
        readings = []
        for per_peer in SEVEN_PEER_CTAS:
            dest.zero_()
            times = measure(lambda:gather7_wide[(7*per_peer,)](src,table,dest,chunk,per_peer,1024,WIDE,num_warps=4))
            if not torch.equal(dest[:7*chunk].view(7,chunk)[:,:128],expected):
                raise ValueError("seven-peer probe did not read the expected sources")
            readings.append(dict(kind="peer_read_7",peers=7,ctas=7*per_peer,bytes=2*7*chunk,
                                 seconds=statistics.median(times),samples=times))
        ranks = spread(readings,group)
        if rank==0:
            write_json(a.out,dict(schema=SCHEMA,origin="exploratory_device_probe",registered=False,gpu_uuids=uuids,
                                  hardware_sha256=sha(a.gate),load="16 bytes",rank_measurements=ranks))
            for i,per_peer in enumerate(SEVEN_PEER_CTAS):
                rates = [r[i]["bytes"]/r[i]["seconds"]/1e9 for r in ranks]
                print(f"7 peers, {7*per_peer:4d} CTAs: {statistics.median(rates):6.1f} GB/s per rank ({min(rates):.1f}-{max(rates):.1f})")
            if a.merge:
                micro = json.loads(a.merge.read_text())
                for registered,wide in zip(micro["rank_measurements"],ranks):
                    registered[:] = [r for r in registered if r["kind"]!="peer_read_7"]+wide
                micro["origin"] = "device_microbench with peer_read_7 replaced by microbench_wide.py (exploratory, post hoc)"
                write_json(a.merged,micro)


if __name__=="__main__":
    main()
