#!/usr/bin/env python3
"""Exploratory: two variants of the direct consumer, timed next to the registered arms.

Not part of the registered design; nothing here enters a score or a sealed comparison.

The registered direct arm launches one kernel per source, and every destination walks the
sources in the same order. Two things follow: a small-M iteration is a chain of eight short
kernels that each wait on peer loads, and all eight destinations read the same source at the
same time. Every variant still reads each row where its source holds it and writes the same
output; the arithmetic (FP32 accumulation, K block 32, q partitions) is unchanged.

  D_Qq_G2_ROT       the registered eight kernels in an order rotated by rank (rank r reads
                    r+1, r+2, ..., and its own rows last): one reader per source at a time.
  D_ALL<bm>_Qq_Gg   one launch whose grid covers the tiles of all eight sources. A tile
                    holds rows of one source. bm is the M tile (16, 32 or 64; 64 if omitted)
                    and g the N tile in units of 128 columns (1, 2 or 4).
  D_MRG<bm>_Qq_Gg   one launch over the 8M rows as one matrix: a tile may hold rows of
                    several sources, so a small M shares one weight read across sources.
                    q may be 1, 2, 4, 8 or 16 here and in D_ALL; a suffix _K64 or _K128
                    changes the K block from 32.
  PULL_SLT          staging without a collective: after the publish barrier one kernel
                    copies each remote row once to local memory, then the tuned cuBLASLt
                    GEMM of the SLT arm runs on the copy.

The frozen package is reused unchanged: inputs, graph replay, barriers, the admission check
and the timing loop (20 warm-up and 100 timed iterations, slowest rank per iteration). The
arms of a cell are timed in one process, once in the listed order and once reversed.

  torchrun --standalone --nproc-per-node=8 consumer_variants.py --gate gate.json \
      --slt slt.json --library libedmir_replay.so --unit GLM_M0001 --out out.json

Run it with the experiment package (experiments/2026-11-23-replay) on PYTHONPATH.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "edmir.consumer-variants.exploratory.v1"
REP = "20261120"
ARMS = ("D_Q1_G2","D_Q1_G2_ROT","D_ALL_Q1_G2","D_MRG_Q1_G2","D_Q4_G2","D_Q4_G2_ROT","D_ALL_Q4_G2","D_MRG_Q4_G2",
        "BNEW_Q1_G2","BNEW_Q4_G2","SLT")
REGISTERED = ("D_Q1_G2","D_Q4_G2","BNEW_Q1_G2","BNEW_Q4_G2","SLT")
VARIANT = re.compile(r"D_(ALL|MRG)(16|32|64)?_Q(1|2|4|8|16)_G([124])(?:_K(32|64|128))?")


def known(arm):
    return (arm in REGISTERED or arm in ("D_Q1_G2_ROT","D_Q4_G2_ROT","PULL_SLT")
            or VARIANT.fullmatch(arm) is not None)
POLICY = dict(warmup=20,timed=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--slt",type=Path)
    ap.add_argument("--library",type=Path)
    ap.add_argument("--unit",required=True)
    ap.add_argument("--rep",default=REP,help="input seed of the registered design (20261120, 20261121 or 20261122)")
    ap.add_argument("--arms",default=",".join(ARMS))
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()

    import torch
    from balanced import BalancedPaths,cells,no_mark
    from make_design import ROOT,sha
    from replay import WORLD,gpu_session,hardware,inventory,spread,timed,write_json
    from snapshots import TILE_M
    from variant_kernels import element_offsets,launch_copy,launch_rows

    class Variants(BalancedPaths):
        def shared_buffers(self):
            if not hasattr(self,"held"):
                delta = element_offsets(self.remote)
                # Row r of the 8M rows is row r%M of source r//M; its output row is r.
                self.held = torch.tensor([delta[s]+i*self.k for s in range(WORLD) for i in range(self.m)],
                                         dtype=torch.int64,device=self.device)
                self.out_rows = torch.arange(WORLD*self.m,dtype=torch.int32,device=self.device)
                self.variant_work,self.variant_partial = {},{}

        def run(self, arm, mark=no_mark):
            if arm!="PULL_SLT":
                return super().run(arm,mark)
            self.ensure_lt()
            self.shared_buffers()
            self.handle.barrier(0,60_000)
            launch_copy(self.remote[0],self.held,self.recv)
            mark("transport")
            self.lt.execute(self.recv,self.weight,self.out,self.choice)
            mark("consumer")
            # Sources may rewrite their rows once every reader holds its copy.
            self.handle.barrier(0,60_000)
            return self.out

        def consumer(self, arm):
            if arm.endswith("_ROT"):
                q = 4 if "_Q4_" in arm else 1
                # Step j: rank r reads source r+1+j, a permutation of the sources at every step.
                order = [(self.rank+1+j)%WORLD for j in range(WORLD)]

                def issue():
                    for s in order:
                        self.launch(self.remote[s],self.weight,self.outputs[s],self.partial,self.table,q,2)
                return issue
            match = VARIANT.fullmatch(arm)
            if match is None:
                return super().consumer(arm)
            kind,bm,q,g,bk = match[1],int(match[2] or TILE_M),int(match[3]),int(match[4]),int(match[5] or 32)
            self.shared_buffers()
            if q>1 and q not in self.variant_partial:
                self.variant_partial[q] = self.out.new_empty((q,WORLD*self.m,self.n),dtype=torch.float32)
            partial = self.variant_partial.get(q)
            if (kind,bm) not in self.variant_work:
                m,total = self.m,WORLD*self.m
                work = ([(s*m+i,(s+1)*m,0) for s in range(WORLD) for i in range(0,m,bm)] if kind=="ALL"
                        else [(i,total,0) for i in range(0,total,bm)])
                self.variant_work[kind,bm] = torch.tensor(work,dtype=torch.int32,device=self.device)
            work = self.variant_work[kind,bm]

            def issue():
                launch_rows(self.remote[0],self.held,self.out_rows,self.weight,self.out,partial,
                            work,self.k,q,bm,g,bk)
            return issue

    hardware(a.gate)
    uuids = inventory()
    shape = cells(ROOT/"design")[a.unit]
    arms = a.arms.split(",")
    if not all(known(arm) for arm in arms):
        ap.error("unknown arm: "+", ".join(arm for arm in arms if not known(arm)))
    with gpu_session(uuids) as (rank,device,group):
        tuned = json.loads(a.slt.read_text())["cells"][a.unit][rank]["selected"] if a.slt else None
        if not all(spread(tuned is not None and a.library is not None,group)):
            arms = [arm for arm in arms if arm not in ("SLT","PULL_SLT")]
        paths = Variants(a.unit,a.rep,shape,rank,device,a.library,tuned,agree=lambda error:spread(error,group))
        admission = {}
        for arm in arms:
            try:
                local = dict(result=list(paths.admit(arm)))
            except Exception as exc:
                local = dict(error=repr(exc))
            values = spread(local,group)
            if any("error" in v for v in values):
                admission[arm] = dict(status="runtime_error",errors=[v.get("error") for v in values])
            else:
                errs = [v["result"][0] for v in values]
                err = None if None in errs else max(errs)
                ok = (err is not None and err<=.02 and not sum(v["result"][1] for v in values)
                      and all(v["result"][2] for v in values))
                admission[arm] = dict(status="ok" if ok else "numerical_failure",scaled_max_err=err,
                                      nonfinite=sum(v["result"][1] for v in values),
                                      fingerprint_ok=all(v["result"][2] for v in values))
        usable = [arm for arm in arms if admission[arm]["status"]=="ok"]
        paths.prepare_graphs(usable)
        rows = []
        for orientation,order in (("forward",usable),("reverse",usable[::-1])):
            for arm in order:
                ranks = timed(paths,POLICY,arm,group)
                host = list(zip(*(x["host"] for x in ranks),strict=True))
                enqueue = list(zip(*(x["enqueue"] for x in ranks),strict=True))
                phases = list(zip(*(x["phases"] for x in ranks),strict=True))
                names = ranks[0]["phase_names"]
                rows.append(dict(arm=arm,orientation=orientation,launch_mode=paths.graphs.launch_mode(arm),
                    wall_ms=statistics.median(map(max,host)),enqueue_ms=statistics.median(map(max,enqueue)),
                    # Slowest rank after each rank's own release phase is removed (see analyze.py).
                    wall_without_release_ms=statistics.median(
                        max(h-p[-1] for h,p in zip(hs,ps,strict=True)) for hs,ps in zip(host,phases,strict=True)),
                    phase_ms={n:statistics.median(max(r[j] for r in it) for it in phases) for j,n in enumerate(names)}))
        if rank==0:
            write_json(a.out,dict(schema=SCHEMA,origin="exploratory_device_probe",registered=False,unit=a.unit,
                shape=shape,rep=a.rep,gpu_uuids=uuids,hardware_sha256=sha(a.gate),
                implementation_sha256=sha(Path(__file__)),policy=POLICY,admission=admission,rows=rows))
            mean = lambda arm,key: math.exp(statistics.fmean(math.log(r[key]) for r in rows if r["arm"]==arm))
            print(f"{a.unit:13s} wall ms: "+"  ".join(f"{arm.replace('_G2','')}={mean(arm,'wall_ms'):.3f}" for arm in usable))
            print(f"{'':13s} no-release: "+"  ".join(f"{arm.replace('_G2','')}={mean(arm,'wall_without_release_ms'):.3f}"
                                                      for arm in usable if arm.startswith("D_")))
            bad = {arm:v for arm,v in admission.items() if v["status"]!="ok"}
            if bad:
                print("  not admitted:",json.dumps(bad)[:600])


if __name__=="__main__":
    main()
