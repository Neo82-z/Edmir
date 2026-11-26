#!/usr/bin/env python3
"""Exploratory probe of peer reads, run after the formal queue of 2026-11-26.

Not part of the registered design: no number from this file enters a score, a prediction or
a comparison of the paper's Section 8. It asks one question. The registered microbenchmark
found that a rank reading one peer reaches about 200 GB/s, while a rank reading all seven
peers at once reaches about 24 GB/s at every CTA count. Which part of that pattern costs
the bandwidth: the number of peers one kernel reads, the number of readers a source serves,
or all eight ranks reading at the same time?

Every pattern moves 8 MiB per (reader, owner) pair from disjoint eighths of each owner's
64 MiB buffer, with 448 CTAs in flight on the reading rank.

  torchrun --standalone --nproc-per-node=8 peer_read_probe.py --gate gate.json --out probe.json

Run it with the experiment package (experiments/2026-11-23-replay) on PYTHONPATH.
"""
import argparse
import statistics
from pathlib import Path

SCHEMA = "edmir.peer-read-probe.exploratory.v1"
CTAS = 448


def slot(rank, owner):
    """Each owner's seven readers take disjoint eighths of its buffer."""
    return (rank-owner)%8-1


def patterns(rank):
    """name -> list of kernels this rank issues per iteration; a kernel is a list of owners.
    An empty list means the rank stays idle."""
    ring = lambda k: [(rank+1+j)%8 for j in range(k)]
    others = [o for o in range(8) if o!=rank]
    return {
        # One kernel that reads k peers at once, on every rank.
        "all_ranks_1_peer": [ring(1)],
        "all_ranks_2_peers": [ring(2)],
        "all_ranks_4_peers": [ring(4)],
        "all_ranks_7_peers": [ring(7)],
        # The same kernels while the other seven ranks stay idle.
        "solo_1_peer": [ring(1)] if rank==0 else [],
        "solo_7_peers": [ring(7)] if rank==0 else [],
        # Seven readers on one source: every other rank reads rank 0.
        "seven_readers_one_source": [[0]] if rank else [],
        # Seven single-peer kernels in a row, as the direct consumer launches them.
        # In source order every rank reads the same source at the same time.
        "sequential_source_order": [[o] for o in others],
        # The same kernels rotated by rank, so each source serves one reader at a time.
        "sequential_rotated": [[o] for o in ring(7)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()

    import torch
    import triton
    import triton.language as tl
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    from make_design import sha
    from replay import gpu_session,hardware,inventory,spread,write_json

    @triton.jit
    def gather(PTRS, B, peers: tl.constexpr, chunk: tl.constexpr, per_peer: tl.constexpr, block: tl.constexpr):
        cta = tl.program_id(0)
        peer = cta%peers
        base = (cta//peers)*block+tl.arange(0,block)
        src = tl.load(PTRS+peer).to(tl.pointer_type(tl.float16))
        for offset in range(0,chunk,per_peer*block):
            ix = offset+base
            value = tl.load(src+ix,ix<chunk,0,cache_modifier=".cg")
            tl.store(B+peer*chunk+ix,value,ix<chunk)

    def measure(f):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        times = []
        for _ in range(9):
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
        chunk = count//8
        src = symmetric.empty((count,),dtype=torch.float16,device=device)
        src.fill_(rank+1)
        dest = torch.empty_like(src)
        handle = symmetric.rendezvous(src,dist.group.WORLD)
        view = lambda o: handle.get_buffer(o,src.shape,src.dtype,0).narrow(0,slot(rank,o)*chunk,chunk)
        readings = []
        for name,kernels in patterns(rank).items():
            launches,position = [],0
            for owners in kernels:
                table = torch.tensor([view(o).data_ptr() for o in owners],dtype=torch.int64,device=device)
                out = dest.narrow(0,position*chunk,len(owners)*chunk)
                launches.append((table,out,len(owners),CTAS//len(owners)))
                position += len(owners)

            def issue():
                for table,out,peers,per_peer in launches:
                    gather[(peers*per_peer,)](table,out,peers,chunk,per_peer,1024,num_warps=4)

            record = dict(pattern=name,kernels=len(kernels),pairs=position,bytes=2*chunk*position)
            try:
                dest.zero_()
                times = measure(issue)
                order = [o for owners in kernels for o in owners]
                expected = torch.tensor([o+1 for o in order],dtype=torch.float16,device=device)[:,None].expand(len(order),128)
                if order and not torch.equal(dest[:len(order)*chunk].view(len(order),chunk)[:,:128],expected):
                    raise ValueError("the probe did not read the expected sources")
                record.update(seconds=statistics.median(times),samples=times)
            except Exception as exc:
                record.update(error=repr(exc))
            readings.append(record)
        ranks = spread(readings,group)
        if rank==0:
            write_json(a.out,dict(schema=SCHEMA,origin="exploratory_device_probe",registered=False,
                gpu_uuids=uuids,hardware_sha256=sha(a.gate),implementation_sha256=sha(Path(__file__)),
                ctas=CTAS,bytes_per_pair=2*chunk,rank_measurements=ranks))
            print(f"{'pattern':28s} {'active':>6s} {'pairs':>5s} {'median ms':>10s} {'GB/s per rank':>14s} {'range GB/s':>16s}")
            for i,first in enumerate(ranks[0]):
                rows = [r[i] for r in ranks if r[i]["pairs"]]
                if any("error" in r for r in rows):
                    print(f"{first['pattern']:28s} error: {[r.get('error') for r in rows][:2]}")
                    continue
                rates = [r["bytes"]/r["seconds"]/1e9 for r in rows]
                print(f"{first['pattern']:28s} {len(rows):6d} {rows[0]['pairs']:5d} "
                      f"{statistics.median(r['seconds'] for r in rows)*1e3:10.3f} {statistics.median(rates):14.1f} "
                      f"{min(rates):7.1f}-{max(rates):.1f}")


if __name__=="__main__":
    main()
