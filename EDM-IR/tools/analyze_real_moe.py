#!/usr/bin/env python3
"""Analyze collected artifacts locally; never trains or loads the selector."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.real_moe import read_journal, save, summarize_blocks, summarize_routes, timeline_overlap


def nsys_inventory(path):
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro",uri=True)) as db:
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:raise ValueError("no CUDA kernel activities in SQLite")
        cols={r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if not {"globalPid","deviceId","start","end"}<=cols:raise ValueError("unsupported Nsight schema")
        return [dict(globalPid=p,pid=(p//0x1000000)%0x1000000,deviceId=d,kernels=n) for p,d,n in db.execute(
            "select globalPid,deviceId,count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by globalPid,deviceId")]


def nsys_intervals(path,rank_map,kernel_classes):
    """Mappings are explicit reviewed evidence; no kernel-name guessing."""
    with closing(sqlite3.connect(f"file:{path.resolve()}?mode=ro",uri=True)) as db:
        cols={r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        name_col="demangledName" if "demangledName" in cols else "shortName"
        if name_col not in cols:raise ValueError("missing kernel names")
        strings=dict(db.execute("select id,value from StringIds"))
        result=[]
        for p,d,start,end,nameid in db.execute(f"select globalPid,deviceId,start,end,{name_col} from CUPTI_ACTIVITY_KIND_KERNEL"):
            key=f"{p}:{d}"
            if key not in rank_map:raise ValueError("unmapped process/GPU: "+key)
            row=rank_map[key]
            if not row.get("evidence"):raise ValueError("rank mapping needs worker/topology evidence")
            name=strings.get(nameid,str(nameid));spec=kernel_classes.get(name)
            kind="unknown"
            if spec is not None:
                if not spec.get("evidence"):raise ValueError("kernel classification needs evidence")
                kind=spec["kind"]
            result.append(dict(origin="nsys",rank=row["rank"],clock_domain=path.name,
                start_ns=start,end_ns=end,kind=kind,kernel_name=name))
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("performance","routes","nsys-inventory","timeline"),required=True)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--rank-map",type=Path);p.add_argument("--kernel-classes",type=Path)
    a=p.parse_args()
    if a.out.exists():raise SystemExit("refuse to overwrite analysis")
    if a.mode=="performance":
        blocks=[]
        for f in sorted(a.input.glob("r[0-2]-*.json")):
            if f.name.endswith(("-engine-metrics.json","-metrics-failure.json","-partial.json")):continue
            v=json.loads(f.read_text())
            if v.get("origin")!="real_vllm" or v.get("diagnostic_only") is not False:raise ValueError("non-performance data")
            blocks.append(v)
        result=summarize_blocks(blocks)
    elif a.mode=="routes":
        result={}
        for name in ("S1","S32","L1","L32"):
            files=sorted(a.input.glob(f"diag-{name}-rank*.jsonl.gz"))
            if len(files)!=8:
                result[name]=dict(status="incomplete",rank_files=len(files));continue
            try:
                result[name]=summarize_routes(r for f in files for r in read_journal(f))
            except Exception as exc:result[name]=dict(status="invalid",error=repr(exc),raw_files_preserved=True)
    elif a.mode=="nsys-inventory":result=dict(process_gpu_pairs=nsys_inventory(a.input),rank_assignment_verified=False)
    else:
        if a.rank_map is None or a.kernel_classes is None:raise ValueError("explicit rank and kernel mappings required")
        intervals=nsys_intervals(a.input,json.loads(a.rank_map.read_text()),json.loads(a.kernel_classes.read_text()))
        result=timeline_overlap(intervals)
    save(a.out,result);print(a.out)


if __name__=="__main__":main()
