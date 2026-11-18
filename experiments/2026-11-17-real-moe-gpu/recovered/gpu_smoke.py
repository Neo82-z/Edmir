"""Short correctness/compatibility smoke, not a performance benchmark."""
import datetime, importlib.metadata, json, os
from pathlib import Path
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']); torch.cuda.set_device(rank)
out=Path('/work/preflight'); out.mkdir(exist_ok=True)
r={'rank':rank,'status':'failed','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    assert torch.cuda.device_count()==8
    props=torch.cuda.get_device_properties(rank)
    assert 'A100' in props.name and 'SXM' in props.name and props.total_memory>=75*1024**3
    assert torch.cuda.get_device_capability(rank)==(8,0)
    r.update(name=props.name,uuid=str(props.uuid),memory_bytes=props.total_memory,torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version())
    peers=[torch.cuda.can_device_access_peer(rank,j) if j!=rank else True for j in range(8)]
    assert all(peers),peers
    r['p2p_access_all']=peers
    for dtype in (torch.float16,torch.bfloat16):
        a=torch.ones((128,128),device='cuda',dtype=dtype); b=a@a
        assert torch.equal(b,torch.full_like(b,128))
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
    x=torch.tensor([rank+1.],device='cuda',dtype=torch.bfloat16)
    dist.all_reduce(x); assert x.item()==36
    src=torch.full((16,),rank+1.,device='cuda',dtype=torch.bfloat16)
    gather=torch.empty((128,),device='cuda',dtype=torch.bfloat16)
    dist.all_gather_into_tensor(gather,src)
    assert torch.equal(gather,torch.arange(1,9,device='cuda',dtype=torch.bfloat16).repeat_interleave(16))
    output=torch.empty((16,),device='cuda',dtype=torch.bfloat16)
    dist.reduce_scatter_tensor(output,torch.full((128,),rank+1.,device='cuda',dtype=torch.bfloat16))
    assert torch.equal(output,torch.full_like(output,36))
    torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
    r.update(status='passed',fp16_bf16_matmul=True,allreduce=True,allgather=True,reduce_scatter=True)
except BaseException as exc:
    r['error']=repr(exc);raise
finally:
    (out/f'gpu-smoke-rank{rank}.json').write_text(json.dumps(r,indent=2)+'\n')
