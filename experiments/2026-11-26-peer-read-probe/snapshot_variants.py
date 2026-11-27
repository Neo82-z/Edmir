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

  BNEW_GATHER<bm>_Qq_Gg_R the same kernel and per-expert tiles on rows that were staged first:
                         the registered pack and All-to-All deliver the routed rows, then
                         one launch consumes the local copy. Only the row source differs
                         from D_GATHER.

  D_PACK<bm>_Qq_Gg_R     D_GATHER with the registered pack kept: every source first packs the
                         rows each destination needs into its peer buffer, as the registered
                         direct arm does, and the one launch reads the packed rows. It
                         separates what the single launch gains from what dropping the pack
                         gains.
                         A suffix _NR before _R (direct arms only) leaves out the release
                         barrier, as a deployment with double buffering would; the inputs
                         are static here, so the readers never race a writer.
  BNEW_BLAS_R            staged rows, regrouped by expert in local memory, then one library
                         GEMM (torch.mm, BF16) per local expert, all replayed as one CUDA
                         graph. The result stays in per-expert order, as the native arm's
                         result stays in its own order; BNEW_BLASC_R also restores the
                         registered row order inside the timed region.

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
VARIANT = re.compile(r"(D_GATHER|D_PACK|BNEW_GATHER)(16|32|64|128)_Q(1|2|4|8|16)_G([124])(?:_K(32|64|128))?(_NR)?_R")
BLAS = ("BNEW_BLAS_R","BNEW_BLASC_R")


def known(arm):
    return arm in REGISTERED or arm in BLAS or VARIANT.fullmatch(arm) is not None
POLICY = dict(warmup=20,timed=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--run",type=Path,help="the formal run directory; its RUN.json binds the registered snapshots")
    ap.add_argument("--layer",type=int,help="an exploratory capture of another layer; not checked against a run")
    ap.add_argument("--snapshots",type=Path,action="append",required=True)
    ap.add_argument("--unit",required=True)
    ap.add_argument("--arms",default=",".join(ARMS))
    ap.add_argument("--orientation",choices=("both","forward","reverse"),default="both",
                    help="time the arms in the listed order, reversed, or both in this process")
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()
    arms = a.arms.split(",")
    if not all(known(arm) for arm in arms):
        ap.error("unknown arm: "+", ".join(arm for arm in arms if not known(arm)))
    if (a.run is None)==(a.layer is None):
        ap.error("give --run for a registered snapshot or --layer for an exploratory capture")

    import torch
    import torch.distributed._symmetric_memory as symmetric
    import torch.distributed as dist

    from balanced import no_mark
    from make_design import ROOT,sha
    from replay import WORLD,Paths,gpu_session,hardware,inventory,snapshot_receipts,spread,timed,write_json
    from snapshots import correctness,layout,load_rank,snapshot_spec
    from variant_kernels import element_offsets,launch_rows

    a.design = ROOT/"design"
    if a.layer is not None:
        # The receipts of an exploratory capture name their own layer; everything else must match the design.
        import snapshots
        from snapshots import load_receipts,locate
        registered = snapshots.snapshot_spec
        snapshots.snapshot_spec = lambda design,snapshot:dict(registered(design,snapshot),layer=a.layer)
        snapshot_spec = snapshots.snapshot_spec
    else:
        run = json.loads((a.run/"RUN.json").read_text())

    class Variants(Paths):
        def routing(self):
            """Rows of this destination in per-expert order: (source, token, registered output row,
            index inside the block that source packs or stages for this rank)."""
            if hasattr(self,"rows"):
                return
            position,by_expert = 0,[[] for _ in self.routes[0][self.rank]]
            for src,source in enumerate(self.routes):
                start = position
                for expert,assignments in enumerate(source[self.rank]):
                    for token,_ in assignments:
                        by_expert[expert].append((src,token,position,position-start))
                        position += 1
            if position!=self.output.shape[0]:
                raise ValueError("routing table does not match the registered output layout")
            self.rows = [row for expert in by_expert for row in expert]
            self.expert_counts = [len(expert) for expert in by_expert]
            self.k = self.hidden.shape[1]
            # Output rows keep the registered order: by source, then local expert, then assignment.
            self.out_rows = torch.tensor([p for _,_,p,_ in self.rows] or [0],dtype=torch.int32,device=self.device)
            self.row_source,self.row_work,self.row_partial = {},{},{}

        def source(self, kind):
            """(base, element offset of every row) for one way of holding the rows."""
            self.routing()
            if kind not in self.row_source:
                k = self.k
                if kind=="D_GATHER":
                    # Every source keeps its activations, unpacked, in a buffer its peers can read.
                    self.resident = symmetric.empty((max(self.sizes),k),dtype=torch.bfloat16,device=self.device)
                    self.resident[:len(self.hidden)].copy_(self.hidden)
                    self.resident_handle = symmetric.rendezvous(self.resident,dist.group.WORLD)
                    if self.resident_handle is None or self.resident_handle.world_size!=WORLD:
                        raise RuntimeError("eight symmetric peer mappings are required")
                    views = [self.resident_handle.get_buffer(s,self.resident.shape,torch.bfloat16,0) for s in range(WORLD)]
                    delta = element_offsets(views)
                    base,held = views[0],[delta[src]+token*k for src,token,_,_ in self.rows]
                elif kind=="D_PACK":
                    # The registered pack: source src holds this rank's rows from row offsets[src] of its peer buffer.
                    delta = element_offsets(self.peers)
                    base,held = self.peers[0],[delta[src]+(self.offsets[src]+i)*k for src,_,_,i in self.rows]
                else:
                    # Staged rows arrive in the registered output order: row p of the local copy is output row p.
                    base,held = self.staged,[p*k for _,_,p,_ in self.rows]
                self.row_source[kind] = (base,torch.tensor(held or [0],dtype=torch.int64,device=self.device))
            return self.row_source[kind]

        def blas_setup(self):
            self.routing()
            if not hasattr(self,"blas_index"):
                rows,n = self.output.shape
                self.blas_index = torch.tensor([p for _,_,p,_ in self.rows],dtype=torch.long,device=self.device)
                self.blas_rows = self.hidden.new_empty((rows,self.k))
                self.blas_out = self.hidden.new_empty((rows,n))
                self.weight_t = self.weight.transpose(1,2).contiguous()
                self.blas_slices,offset = [],0
                for expert,count in enumerate(self.expert_counts):
                    if count:
                        self.blas_slices.append((offset,count,expert))
                    offset += count

        def consumer(self, arm):
            if arm in BLAS:
                self.blas_setup()

                def gemms():
                    if len(self.blas_index):
                        torch.index_select(self.staged,0,self.blas_index,out=self.blas_rows)
                        for start,count,expert in self.blas_slices:
                            torch.mm(self.blas_rows[start:start+count],self.weight_t[expert],
                                     out=self.blas_out[start:start+count])
                        if arm=="BNEW_BLASC_R":
                            self.output.index_copy_(0,self.blas_index,self.blas_out)
                return gemms
            match = VARIANT.fullmatch(arm)
            if match is None:
                return super().consumer(arm)
            kind,bm,q,g,bk = match[1],int(match[2]),int(match[3]),int(match[4]),int(match[5] or 32)
            base,held = self.source(kind)
            if q>1 and q not in self.row_partial:
                self.row_partial[q] = self.output.new_empty((q,*self.output.shape),dtype=torch.float32)
            partial = self.row_partial.get(q)
            if bm not in self.row_work:
                work,offset = [],0
                for expert,count in enumerate(self.expert_counts):
                    work.extend((i,offset+count,expert) for i in range(offset,offset+count,bm))
                    offset += count
                self.row_work[bm] = torch.tensor(work,dtype=torch.int32,device=self.device).view(-1,3)
            work = self.row_work[bm]

            def issue():
                if len(work):
                    launch_rows(base,held,self.out_rows,self.weight,self.output,partial,work,self.k,q,bm,g,bk)
            return issue

        def run(self, arm, mark=no_mark):
            if arm in BLAS:
                self.materialize(self.hidden,True,mark)   # the registered pack and All-to-All
                self.graphs.run(arm,self.consumer(arm))
                mark("consumer")
                return self.output if arm=="BNEW_BLASC_R" else self.blas_out
            match = VARIANT.fullmatch(arm)
            if match is None:
                return super().run(arm,mark)
            kind = match[1]
            self.source(kind)
            if kind=="BNEW_GATHER":
                self.materialize(self.hidden,True,mark)   # the registered pack and All-to-All
                self.graphs.run(arm,self.consumer(arm))
                mark("consumer")
                return self.output
            if kind=="D_PACK":
                self.materialize(self.hidden,False,mark)  # the registered pack and publish barrier
                handle = self.handle
            else:
                handle = self.resident_handle
                handle.barrier(0,60_000)                  # publish: every source's rows are in place
                mark("transport")
            self.graphs.run(arm,self.consumer(arm))
            mark("consumer")
            if not match[6]:
                handle.barrier(0,60_000)                  # release
            return self.output

        def canonical(self, arm, result):
            """A result in the registered row order; only BNEW_BLAS_R leaves it in another order."""
            if arm!="BNEW_BLAS_R":
                return result
            ordered = torch.empty_like(self.output)
            if len(self.blas_index):
                ordered.index_copy_(0,self.blas_index,result)
            return ordered

    hardware(a.gate)
    uuids = inventory()
    with gpu_session(uuids) as (rank,device,group):
        spec = snapshot_spec(a.design,a.unit)
        if a.layer is not None:
            root = locate(a.snapshots,a.unit)
            receipts = load_receipts(root,a.design,a.unit)
        else:
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
                    match = VARIANT.fullmatch(arm)
                    expected = reference["Q4" if match is not None and match[3]=="4" and "Q4" in reference else "Q1"]
                    paths.prepare_graphs([arm])
                    paths.output.fill_(float("nan"))
                    if arm in BLAS:
                        paths.blas_out.fill_(float("nan"))
                    result = paths.canonical(arm,paths.run(arm))
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
            if a.orientation not in ("both",orientation):
                continue
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
                layer=spec["layer"],spec=spec,arms=arms,orientation=a.orientation,gpu_uuids=uuids,hardware_sha256=sha(a.gate),
                implementation_sha256=sha(Path(__file__)),policy=POLICY,admission=admission,rows=rows))
            mean = lambda arm,key: math.exp(statistics.fmean(math.log(r[key]) for r in rows if r["arm"]==arm))
            short = lambda arm: arm.replace("_G2","").replace("_R","")
            print(f"{a.unit:15s} wall ms: "+"  ".join(f"{short(arm)}={mean(arm,'wall_ms'):.3f}" for arm in usable))
            print(f"{'':15s} no-release: "+"  ".join(f"{short(arm)}={mean(arm,'wall_without_release_ms'):.3f}"
                                                        for arm in usable if arm.startswith("D_")))
            first = {r["arm"]:r["phase_ms"] for r in rows[:len(usable)]}
            print(f"{'':15s} phases: "+"  ".join(f"{short(arm)}={ {k:round(v,3) for k,v in first[arm].items()} }" for arm in usable))
            bad = {arm:v for arm,v in admission.items() if v["status"]!="ok"}
            if bad:
                print("  not admitted:",json.dumps(bad)[:800])


if __name__=="__main__":
    main()
