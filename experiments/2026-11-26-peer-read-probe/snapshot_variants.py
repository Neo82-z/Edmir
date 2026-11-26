#!/usr/bin/env python3
"""Exploratory: a one-launch direct consumer on the captured routes, next to the registered arms.

Not part of the registered design; nothing here enters a score or a sealed comparison.

The registered direct arm on a snapshot packs the rows each destination needs into a peer
buffer, launches one kernel per source, and tiles the rows of each (source, expert) pair on
their own. The variant here changes three things and keeps the arithmetic:

  D_GATHER<bm>_Qq_Gg_R   each source keeps its activations in symmetric memory as they are
                         (no pack); one launch covers all sources; tiles are formed per
                         expert, so the rows that different sources route to one expert
                         share that expert's weight read. A row is fetched from the source
                         that holds it by its token index. bm is the M tile (16, 32 or 64)
                         and g the N tile in units of 128 columns; q may be 1, 2, 4, 8 or
                         16, and a suffix _K64 or _K128 before _R changes the K block.

Which token each destination needs from each source is the static routing table of the
snapshot. The registered protocol keeps that precomputation outside the timed region for
every arm, and the native arm receives its alignment tensors the same way.

The variant must reproduce the output of the registered direct arm with the same q, which
formal admission checked against the FP32 reference. Its output buffer is filled with NaN
before the run that is compared. Timing uses the frozen loop: 20 warm-up and 100 timed
iterations, slowest rank per iteration, the arms of a snapshot in one process, once in the
listed order and once reversed.

  torchrun --standalone --nproc-per-node=8 snapshot_variants.py --gate gate.json --run RUN_DIR \
      --snapshots <capture>/workers/snapshots --unit GLM_DECODE --out out.json

Run it with the experiment package (experiments/2026-11-23-replay) on PYTHONPATH.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "edmir.snapshot-variants.exploratory.v1"
REGISTERED = ("D_Q1_G2_R","D_Q4_G2_R","BNEW_Q1_G2_R","BNEW_Q4_G2_R","NATIVE")
ARMS = ("D_Q1_G2_R","D_GATHER16_Q1_G2_R","D_GATHER64_Q1_G2_R","D_Q4_G2_R","D_GATHER16_Q4_G2_R","D_GATHER64_Q4_G2_R",
        "BNEW_Q1_G2_R","BNEW_Q4_G2_R","NATIVE")
VARIANT = re.compile(r"D_GATHER(16|32|64)_Q(1|2|4|8|16)_G([124])(?:_K(32|64|128))?_R")
POLICY = dict(warmup=20,timed=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--run",type=Path,required=True,help="the formal run directory, for its RUN.json")
    ap.add_argument("--snapshots",type=Path,action="append",required=True)
    ap.add_argument("--unit",required=True)
    ap.add_argument("--arms",default=",".join(ARMS))
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    if any(arm not in REGISTERED and VARIANT.fullmatch(arm) is None for arm in arms):
        ap.error("unknown arm")

    import torch
    import torch.distributed._symmetric_memory as symmetric
    import torch.distributed as dist

    from balanced import no_mark
    from make_design import ROOT,sha
    from replay import WORLD,Paths,gpu_session,hardware,inventory,snapshot_receipts,spread,timed,write_json
    from snapshots import correctness,layout,load_rank,snapshot_spec
    from variant_kernels import element_offsets,launch_rows

    a.design = ROOT/"design"
    run = json.loads((a.run/"RUN.json").read_text())

    class Variants(Paths):
        def gather_setup(self):
            if hasattr(self,"held"):
                return
            k = self.hidden.shape[1]
            # Every source keeps its activations, unpacked, in a buffer its peers can read.
            self.resident = symmetric.empty((max(self.sizes),k),dtype=torch.bfloat16,device=self.device)
            self.resident[:len(self.hidden)].copy_(self.hidden)
            self.resident_handle = symmetric.rendezvous(self.resident,dist.group.WORLD)
            if self.resident_handle is None or self.resident_handle.world_size!=WORLD:
                raise RuntimeError("eight symmetric peer mappings are required")
            views = [self.resident_handle.get_buffer(s,self.resident.shape,torch.bfloat16,0) for s in range(WORLD)]
            delta = element_offsets(views)
            self.resident_base = views[0]
            # Output rows keep the registered order: by source, then local expert, then assignment.
            position,by_expert = 0,[[] for _ in self.routes[0][self.rank]]
            for src,source in enumerate(self.routes):
                for expert,assignments in enumerate(source[self.rank]):
                    for token,_ in assignments:
                        by_expert[expert].append((delta[src]+token*k,position))
                        position += 1
            if position!=self.output.shape[0]:
                raise ValueError("routing table does not match the registered output layout")
            rows = [row for expert in by_expert for row in expert]
            self.held = torch.tensor([h for h,_ in rows] or [0],dtype=torch.int64,device=self.device)
            self.out_rows = torch.tensor([o for _,o in rows] or [0],dtype=torch.int32,device=self.device)
            self.expert_counts = [len(expert) for expert in by_expert]
            self.gather_work,self.gather_partial = {},{}

        def consumer(self, arm):
            match = VARIANT.fullmatch(arm)
            if match is None:
                return super().consumer(arm)
            bm,q,g,bk = int(match[1]),int(match[2]),int(match[3]),int(match[4] or 32)
            self.gather_setup()
            if q>1 and q not in self.gather_partial:
                self.gather_partial[q] = self.output.new_empty((q,*self.output.shape),dtype=torch.float32)
            partial = self.gather_partial.get(q)
            if bm not in self.gather_work:
                work,offset = [],0
                for expert,count in enumerate(self.expert_counts):
                    work.extend((i,offset+count,expert) for i in range(offset,offset+count,bm))
                    offset += count
                self.gather_work[bm] = torch.tensor(work,dtype=torch.int32,device=self.device).view(-1,3)
            work = self.gather_work[bm]

            def issue():
                if len(work):
                    launch_rows(self.resident_base,self.held,self.out_rows,self.weight,self.output,
                                partial,work,self.hidden.shape[1],q,bm,g,bk)
            return issue

        def run(self, arm, mark=no_mark):
            if VARIANT.fullmatch(arm) is None:
                return super().run(arm,mark)
            self.gather_setup()
            self.resident_handle.barrier(0,60_000)   # publish: every source's rows are in place
            mark("transport")
            self.graphs.run(arm,self.consumer(arm))
            mark("consumer")
            self.resident_handle.barrier(0,60_000)   # release
            return self.output

    hardware(a.gate)
    uuids = inventory()
    with gpu_session(uuids) as (rank,device,group):
        spec = snapshot_spec(a.design,a.unit)
        root,receipts = snapshot_receipts(a,run,a.unit)
        payload = load_rank(root,a.unit,receipts[rank],spec)
        ids = spread(payload["ids"].tolist(),group)
        routes = layout(ids,[len(r["tokens"]) for r in receipts],spec["local"])
        aligned = tuple(payload[k] for k in ("sorted_token_ids","expert_ids","num_tokens_post_padded"))
        paths = Variants(payload,receipts,routes,rank,device,group,aligned,spec)
        admission,reference = {},{}
        for arm in arms:
            try:
                if arm in REGISTERED:
                    err,nans,fingerprint = paths.admit(arm)
                    if arm.startswith("D_"):
                        reference[arm[2:4]] = paths.run(arm).float().clone()
                        torch.cuda.synchronize()
                    local = dict(err=err,nans=nans,ok=bool(fingerprint))
                else:
                    # Any q is checked against a registered direct arm; q=4 against the q=4 arm.
                    expected = reference["Q4" if VARIANT.fullmatch(arm)[2]=="4" and "Q4" in reference else "Q1"]
                    paths.prepare_graphs([arm])
                    paths.output.fill_(float("nan"))
                    result = paths.run(arm)
                    torch.cuda.synchronize()
                    err,nans = correctness(result,expected)
                    local = dict(err=err,nans=nans,ok=True,identical=bool(torch.equal(result.float(),expected)))
            except Exception as exc:
                local = dict(error=repr(exc))
            values = spread(local,group)
            if any("error" in v for v in values):
                admission[arm] = dict(status="runtime_error",errors=[v.get("error") for v in values])
            else:
                errs = [v["err"] for v in values]
                err = None if None in errs else max(errs)
                ok = err is not None and err<=.02 and not sum(v["nans"] for v in values) and all(v["ok"] for v in values)
                admission[arm] = dict(status="ok" if ok else "numerical_failure",scaled_max_err=err,
                                      nonfinite=sum(v["nans"] for v in values),
                                      identical_to_registered=all(v.get("identical",True) for v in values))
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
                    phase_ms={n:statistics.median(max(r[j] for r in it) for it in phases) for j,n in enumerate(names)},
                    live_rows=[x["counts"]["projection_live_rows"] for x in ranks]))
        if rank==0:
            write_json(a.out,dict(schema=SCHEMA,origin="exploratory_device_probe",registered=False,unit=a.unit,
                spec=spec,gpu_uuids=uuids,hardware_sha256=sha(a.gate),implementation_sha256=sha(Path(__file__)),
                policy=POLICY,admission=admission,rows=rows))
            mean = lambda arm,key: math.exp(statistics.fmean(math.log(r[key]) for r in rows if r["arm"]==arm))
            short = lambda arm: arm.replace("_G2","").replace("_R","")
            print(f"{a.unit:15s} wall ms: "+"  ".join(f"{short(arm)}={mean(arm,'wall_ms'):.3f}" for arm in usable))
            print(f"{'':15s} no-release: "+"  ".join(f"{short(arm)}={mean(arm,'wall_without_release_ms'):.3f}"
                                                        for arm in usable if arm.startswith("D_")))
            first = {r["arm"]:r["phase_ms"] for r in rows if r["orientation"]=="forward"}
            print(f"{'':15s} phases: "+"  ".join(f"{short(arm)}={ {k:round(v,3) for k,v in first[arm].items()} }" for arm in usable))
            bad = {arm:v for arm,v in admission.items() if v["status"]!="ok"}
            if bad:
                print("  not admitted:",json.dumps(bad)[:800])


if __name__=="__main__":
    main()
