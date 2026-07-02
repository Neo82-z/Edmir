import torch
from torch.profiler import profile, record_function, ProfilerActivity



def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


    //   https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html


torch.cuda.is_available()
torch.cuda.get_device_name()
torch.cuda.Stream()
torch.cuda.Event()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push()
torch.cuda.nvtx.range_pop()

torch.randn(..., device="cuda")
torch.empty(..., device="cuda")
torch.empty(..., pin_memory=True)
torch.matmul(a, b)
tensor.copy_(cpu_tensor, non_blocking=True)


// we shold test a gemm.

A = torch.randn(128, 256, device='cuda')
B = torch.randn(256, 512, device='cuda')
c = torch.mm(A,B)


batch_A = torch.randn(10, 128, 256, device="cuda")
batch_B = torch.randn(10, 256, 512, device="cuda")
batch_C = torch.bmm(batch_A, batch_B)

