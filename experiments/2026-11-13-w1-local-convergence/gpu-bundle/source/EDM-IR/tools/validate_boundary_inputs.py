"""Independent integer-dot spot checks of the sealed FP32 references.

Also simulate serial FP16 partition accumulation on sampled output elements.
This is not GPU correctness validation or a complete emulation of CUTLASS.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_boundary_gpu import verify_bundle
from independent_selector_validation import digest


def check(root):
    p=verify_bundle(root);records=[]
    for work in p['workloads']:
        P,M,K,N=work['shape'];directory=root/'data'/work['id']
        data=np.memmap(directory/'inputs.bin',dtype='<f2',mode='r')
        reference=np.memmap(directory/'reference.bin',dtype='<f4',mode='r')
        ai=P*P*M*K;rows=[0,M//2,M-1];columns=[0,N//2,N-1]
        errors={q:{'max_abs':0.,'max_rel':0.} for q in (1,2,4,8)};checked=0
        for s in range(P):
            for d in range(P):
                idx=s*P+d;a=data[idx*M*K:(idx+1)*M*K].reshape(M,K)[rows].astype(np.float64)
                b=data[ai+d*K*N:ai+(d+1)*K*N].reshape(K,N)[:,columns].astype(np.float64)
                A=(a*32).astype(np.int64);B=(b*32).astype(np.int64)
                if not np.array_equal(A/32,a) or not np.array_equal(B/32,b):raise ValueError('input values are not exact dyadics')
                exact=(A@B).astype(np.float64)/1024
                expected=reference[idx*M*N:(idx+1)*M*N].reshape(M,N)[np.ix_(rows,columns)]
                if not np.array_equal(exact,expected):raise ValueError('FP32 reference disagrees with independent integer dot')
                checked+=exact.size
                for q in errors:
                    running=np.zeros_like(exact,dtype=np.float16)
                    for part in range(q):
                        lo,hi=part*K//q,(part+1)*K//q
                        partial=(A[:,lo:hi]@B[lo:hi]).astype(np.float32)/1024
                        running=(partial+running.astype(np.float32)).astype(np.float16)
                    err=np.abs(running.astype(np.float64)-exact)
                    errors[q]['max_abs']=max(errors[q]['max_abs'],float(err.max()))
                    errors[q]['max_rel']=max(errors[q]['max_rel'],float((err/np.maximum(np.abs(exact),1.)).max()))
        if any(v['max_abs']>.02 and v['max_rel']>.02 for v in errors.values()):raise ValueError('sampled split accumulation fails planned tolerance')
        records.append({'shape':work['shape'],'independent_integer_dot_elements':checked,'sampled_serial_fp16_errors':errors})
    return {'package_sha256':digest(p),'scope':'CPU integer reference and sampled FP16 serial accumulation only; no GPU validation',
            'records':records,'checked_reference_elements':sum(r['independent_integer_dot_elements'] for r in records)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('fresh result path required')
    a.out.write_text(json.dumps(check(a.package),indent=2)+'\n')
