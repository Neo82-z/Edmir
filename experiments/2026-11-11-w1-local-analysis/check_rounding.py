#!/usr/bin/env python3
"""Bounded CPU arithmetic diagnostic, not a GPU kernel or latency simulation.

Uses one existing development input: M512, seed 20261101. Either eight evenly
spaced rows or the complete output; all N columns and all 64 pairs are included.
No adaptive coordinate search, no additional seeds or held-out shapes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feedback(partials, q):
    """FP32 dot products, but FP16 output read/modified at each K partition."""
    total = np.zeros_like(partials[0], dtype=np.float32)
    for start in range(0, 16, 16//q):
        chunk = np.sum(partials[start:start+16//q], axis=0, dtype=np.float32)
        total = (total + chunk).astype(np.float16).astype(np.float32)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("/Users/neo/Documents/工作/交付/edmir-w1-2026-11-01"))
    parser.add_argument("--out", type=Path, default=HERE/"rounding-diagnostic.json")
    parser.add_argument("--all-rows",action="store_true",help="Replay every output of this one seed")
    args = parser.parse_args()
    started = time.monotonic()
    tag, seed = "m512", 20261101
    meta = json.loads((args.assets/"bundle/inputs.json").read_text())[tag][str(seed)]
    root = args.assets/"inputs"/tag/str(seed)
    hashes = {name:sha(root/name) for name in ("inputs.bin","reference.bin")}
    if hashes != {"inputs.bin":meta["input_sha256"],"reference.bin":meta["reference_sha256"]}:
        raise ValueError("input/reference bytes differ from sealed development data")
    p,m,k,n = meta["shape"]
    if (p,m,k,n)!=(8,512,8192,512):raise ValueError("unexpected diagnostic shape")
    a = np.memmap(root/"inputs.bin",mode="r",dtype="<f2",shape=(p,p,m,k))
    w = np.memmap(root/"inputs.bin",mode="r",dtype="<f2",offset=2*p*p*m*k,shape=(p,k,n))
    ref = np.memmap(root/"reference.bin",mode="r",dtype="<f4",shape=(p,p,m,n))
    row_ids = np.arange(m) if args.all_rows else np.linspace(0,m-1,8,dtype=int)
    stats = {q:dict(max_abs_error=0.,max_scaled_error=0.,failed_elements=0,first_failures=[]) for q in (1,2,4,8,16)}
    reference_difference, checked = 0., 0
    for d in range(p):
        weight = w[d].astype(np.float32)
        for s in range(p):
            activation = a[s,d,row_ids,:].astype(np.float32)
            # Values lie on 1/32; verify the dyadic precondition for exact FP32
            # products/sums here, rather than assuming generic inputs are exact.
            for value in (activation,weight):
                if not (np.all(value*32==np.rint(value*32)) and np.max(np.abs(value))<=.5):
                    raise ValueError("diagnostic requires the sealed dyadic input grid")
            partials = np.stack([activation[:,i:i+k//16] @ weight[i:i+k//16,:]
                                 for i in range(0,k,k//16)])
            expected = ref[s,d,row_ids,:]
            exact = np.sum(partials,axis=0,dtype=np.float32)
            difference = float(np.max(np.abs(exact-expected)))
            reference_difference = max(reference_difference,difference)
            if difference != 0.:raise ValueError("CPU partition sums do not equal the saved FP32 reference")
            checked += expected.size
            for q, summary in stats.items():
                result = feedback(partials,q)
                absolute = np.abs(result-expected)
                scaled = absolute/np.maximum(np.abs(expected),1.)
                summary["max_abs_error"] = max(summary["max_abs_error"],float(absolute.max()))
                summary["max_scaled_error"] = max(summary["max_scaled_error"],float(scaled.max()))
                summary["failed_elements"] += int(np.count_nonzero(scaled>.02))
                remaining = 4-len(summary["first_failures"])
                if remaining>0:
                    for i,j in np.argwhere(scaled>.02)[:remaining]:
                        summary["first_failures"].append(dict(source=s,destination=d,row=int(row_ids[i]),column=int(j),
                            reference=float(expected[i,j]),simulated=float(result[i,j]),scaled_error=float(scaled[i,j])))
    output = dict(kind="CPU_arithmetic_diagnostic_not_GPU_measurement",shape_PMKN=meta["shape"],seed=seed,
                  diagnostic_source_sha256=sha(Path(__file__)),
                  fixed_row_indices=row_ids.tolist(),source_destination_pairs=64,checked_elements=checked,
                  fraction_of_full_output=checked/(p*p*m*n),input_hashes=hashes,
                  reference_max_abs_difference=reference_difference,
                  recurrence="D_i=FP16(FP32_dot(A_i,B_i)+FP32(D_(i-1))); D_-1=0",
                  precision_contract="max(abs(error)/max(1,abs(reference))) <= 0.02; not atol+rtol allclose",
                  q_results=stats,elapsed_seconds=time.monotonic()-started,numpy_version=np.__version__,
                  limitations=["No saved GPU output elements to compare coordinate-by-coordinate",
                               "Does not validate semaphore ordering, races or real GPU FP32 intermediates",
                               "One development seed and one shape; full output" if args.all_rows else "Only 1/64 of M512 output elements for one development seed",
                               "No latency/occupancy/communication prediction"])
    args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in output.items() if k not in ("input_hashes","limitations","fixed_row_indices")},indent=2))


if __name__ == "__main__":
    main()
