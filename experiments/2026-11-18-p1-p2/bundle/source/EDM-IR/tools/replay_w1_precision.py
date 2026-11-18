#!/usr/bin/env python3
"""Full-output CPU arithmetic replay on the 12 prescribed W1 inputs only.

FP32 partial sums are exact for this dyadic generator, verified against each
archived reference. This does not test CUDA scheduling or give GPU latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import numpy as np

SHAPES={"grouped":[8,64,8192,1024],"m512":[8,512,8192,512],
        "large_n":[8,64,4096,2048],"m256":[8,256,8192,512]}
SEEDS=(20261020,20261101,20261102)


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=Path("/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01"))
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((args.assets/"bundle/inputs.json").read_text())
    started=time.monotonic();summaries=[]
    for tag,shape in SHAPES.items():
        for seed in SEEDS:
            begin=time.monotonic();meta=metadata[tag][str(seed)];root=args.assets/"inputs"/tag/str(seed)
            if meta["shape"]!=shape:raise ValueError("non-development shape")
            if sha(root/"inputs.bin")!=meta["input_sha256"] or sha(root/"reference.bin")!=meta["reference_sha256"]:
                raise ValueError("sealed input changed")
            p,m,k,n=shape
            a=np.memmap(root/"inputs.bin",mode="r",dtype="<f2",shape=(p,p,m,k))
            w=np.memmap(root/"inputs.bin",mode="r",dtype="<f2",offset=2*p*p*m*k,shape=(p,k,n))
            ref=np.memmap(root/"reference.bin",mode="r",dtype="<f4",shape=(p,p,m,n))
            values={f"q{q}_{mode}":dict(max_abs_error=0.,max_scaled_error=0.,failed_elements=0)
                    for q in (1,2,4,8,16) for mode in ("fp16_feedback","fp32_then_fp16")}
            for d in range(p):
                weight=w[d].astype(np.float32)
                for s in range(p):
                    if time.monotonic()-started>240:raise TimeoutError("bounded CPU replay exceeded four minutes; completed cases preserved")
                    activation=a[s,d].astype(np.float32)
                    for operand in (activation,weight):
                        if not (np.all(operand*32==np.rint(operand*32)) and np.max(np.abs(operand))<=.5):
                            raise ValueError("outside exact dyadic arithmetic scope")
                    partials=np.stack([activation[:,i:i+k//16]@weight[i:i+k//16] for i in range(0,k,k//16)])
                    expected=ref[s,d]
                    exact=np.sum(partials,axis=0,dtype=np.float32)
                    if np.any(exact!=expected):raise ValueError("CPU partition sums differ from archived reference")
                    for q in (1,2,4,8,16):
                        old=np.zeros_like(expected)
                        fp32=np.zeros_like(expected)
                        for i in range(0,16,16//q):
                            chunk=np.sum(partials[i:i+16//q],axis=0,dtype=np.float32)
                            old=(old+chunk).astype(np.float16).astype(np.float32)
                            fp32+=chunk
                        for mode,result in (("fp16_feedback",old),("fp32_then_fp16",fp32.astype(np.float16).astype(np.float32))):
                            absolute=np.abs(result-expected);scaled=absolute/np.maximum(1.,np.abs(expected))
                            st=values[f"q{q}_{mode}"]
                            st["max_abs_error"]=max(st["max_abs_error"],float(absolute.max()))
                            st["max_scaled_error"]=max(st["max_scaled_error"],float(scaled.max()))
                            st["failed_elements"]+=int(np.count_nonzero(scaled>.02))
            summary=dict(shape_tag=tag,shape=shape,seed=seed,checked_elements=p*p*m*n,results=values,
                input_sha256=meta["input_sha256"],reference_sha256=meta["reference_sha256"],
                source_sha256=sha(Path(__file__)),kind="CPU_arithmetic_not_GPU_validation",seconds=time.monotonic()-begin)
            (args.out/f"{tag}-{seed}.json").write_text(json.dumps(summary,indent=2)+"\n")
            summaries.append(summary)
            print(json.dumps(dict(shape=tag,seed=seed,seconds=summary["seconds"],
                                 old_q8_failures=values["q8_fp16_feedback"]["failed_elements"],
                                 fp32_q8_failures=values["q8_fp32_then_fp16"]["failed_elements"])),flush=True)
    (args.out/"summary.json").write_text(json.dumps(dict(cases=len(summaries),seconds=time.monotonic()-started,
        numpy_version=np.__version__,source_sha256=sha(Path(__file__)),
        limitations=["Exact dyadic synthetic inputs only, not real MoE activations or BF16",
                     "No GPU element arrays; compare summaries only", "No CUDA compilation, race or latency validation"]),indent=2)+"\n")


if __name__=="__main__":main()
