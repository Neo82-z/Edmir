"""FP16 balanced inputs, matched transport, strong staged, and bridge arms."""
import ctypes as ct
import hashlib
import statistics

from make_design import read
from snapshots import tiles

WORLD = 8
WORKSPACE = 32*1024**2


def cells(design):
    return {r["cell_id"]: tuple(int(r[f]) for f in ("m_per_source_destination","k","n"))
            for r in read(design/"cells.csv")}


def seed_for(unit, rep, rank, use):
    text = f"edmir-balanced-generator-v1:{unit}:{rep}:{rank}:{use}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"little")


def inputs(torch, unit, rep, rank, shape):
    m,k,n = shape
    values = []
    for sizes,use in (((WORLD*m,k),"activation"),((n,k),"weight")):
        gen = torch.Generator(device="cpu").manual_seed(seed_for(unit,rep,rank,use))
        values.append((torch.randn(sizes,generator=gen,dtype=torch.float32)*.125).to(torch.float16))
    return tuple(values)


def tensor_digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(ct.string_at(tensor.data_ptr(),tensor.numel()*tensor.element_size()))
    return h.hexdigest()


def combine_input_digests(digests):
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def input_digest(torch, unit, rep, shape):
    return combine_input_digests(tensor_digest(inputs(torch,unit,rep,rank,shape)) for rank in range(WORLD))


def load_library(path):
    """Declare every C signature at load time: 64-bit device pointers must never pass as C int."""
    library = ct.CDLL(str(path.resolve()))
    p,i,z = ct.c_void_p,ct.c_int,ct.c_size_t
    for name,args in (
        ("create",(i,i,i,ct.POINTER(p))),("destroy",(p,)),("count",(p,)),
        ("algo",(p,i,p,z)),("algo_size",()),("run",(p,p,p,p,p,p,p)),
    ):
        fn = getattr(library,"edmir_lt_"+name)
        fn.argtypes = args
        fn.restype = None if name=="destroy" else i
    library.edmir_bridge_cutlass.argtypes = (i,i,i,p,p,p,p)
    library.edmir_bridge_cutlass.restype = i
    return library


class Lt:
    def __init__(self, library, m, n, k, torch, device):
        self.library = library
        p = ct.c_void_p
        self.plan = p()
        self.check(library.edmir_lt_create(m,n,k,ct.byref(self.plan)))
        self.work = torch.empty(WORKSPACE,dtype=torch.uint8,device=device)
        self.size = library.edmir_lt_algo_size()
        self.torch = torch

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"native staged backend status {code}")

    def close(self):
        if self.plan:
            self.library.edmir_lt_destroy(self.plan)
            self.plan = ct.c_void_p()

    def candidates(self):
        for i in range(self.library.edmir_lt_count(self.plan)):
            buf = ct.create_string_buffer(self.size)
            if self.library.edmir_lt_algo(self.plan,i,buf,self.size)==0:
                yield buf.raw.hex()

    def execute(self, a, b, out, algorithm):
        self.check(self.library.edmir_lt_run(self.plan,a.data_ptr(),b.data_ptr(),out.data_ptr(),
                   self.work.data_ptr(),algorithm,self.torch.cuda.current_stream().cuda_stream))


class ConsumerGraphs:
    """Replay each consumer arm's kernel launches as one CUDA graph; communication stays eager.

    Every rank captures the same consumer arms together. If capture fails on any rank, every
    rank runs all of them eagerly, so both sides of a comparison share one launch mode.
    Replay issues the same kernels with the same arguments, so outputs are unchanged."""

    def __init__(self, torch, agree):
        self.torch,self.agree = torch,agree
        self.graphs,self.mode,self.errors = {},None,None

    def prepare(self, consumers):
        todo = {arm:fn for arm,fn in consumers.items() if arm not in self.graphs}
        if not todo or self.mode=="eager":
            return
        torch,captured,error = self.torch,{},None
        try:
            if self.agree is None:
                raise RuntimeError("no rank agreement channel for graph capture")
            for fn in todo.values():
                fn()  # compile and load every kernel before capture
            torch.cuda.synchronize()
            for arm,fn in todo.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,capture_error_mode="thread_local"):
                    fn()
                captured[arm] = graph
            torch.cuda.synchronize()
        except Exception as exc:
            error = repr(exc)
        errors = self.agree(error) if self.agree is not None else [error]
        if any(errors):
            self.mode,self.errors,self.graphs = "eager",errors,{}
        else:
            self.mode = "graph"
            self.graphs.update(captured)

    def run(self, arm, fn):
        graph = self.graphs.get(arm)
        if graph is None:
            fn()
        else:
            graph.replay()

    def launch_mode(self, arm):
        return "graph" if arm in self.graphs else "eager"


def consumer_arm(arm):
    return arm.startswith("D_") or arm.startswith("BNEW_")


class BalancedPaths:
    def __init__(self, unit, rep, shape, rank, device, library, tuned, agree=None):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symmetric
        from gate_up import launch

        self.torch,self.dist,self.launch = torch,dist,launch
        self.m,self.k,self.n = shape
        self.rank,self.device = rank,device
        # Only SLT and the BOLD bridge need the shared library; D and BNEW run without it.
        self.library_path,self._library,self.lt,self.choice = library,None,None,None
        vec,weight = inputs(torch,unit,rep,rank,shape)
        self.input_sha256 = tensor_digest((vec,weight))
        self.weight = weight.to(device)
        self.weight_t = self.weight.T.contiguous()
        self.send = symmetric.empty(tuple(vec.shape),dtype=torch.float16,device=device)
        self.send.copy_(vec)
        self.handle = symmetric.rendezvous(self.send,dist.group.WORLD)
        if self.handle is None or self.handle.world_size != WORLD:
            raise RuntimeError("balanced symmetric memory rendezvous failed")
        self.remote = [self.handle.get_buffer(s,self.send.shape,torch.float16,0)
                       .narrow(0,rank*self.m,self.m) for s in range(WORLD)]
        self.recv = torch.empty_like(self.send)
        self.local = self.recv.split(self.m)
        self.out = self.send.new_empty((WORLD*self.m,self.n))
        self.outputs = self.out.split(self.m)
        self.partial = self.out.new_empty((4,self.m,self.n),dtype=torch.float32)
        self.table = torch.tensor(tiles([self.m]),dtype=torch.int32,device=device)
        self.tuned = tuned
        self.graphs = ConsumerGraphs(torch,agree)

    @property
    def library(self):
        if self._library is None:
            self._library = load_library(self.library_path)
        return self._library

    def close(self):
        if self.lt is not None:
            self.lt.close()

    def ensure_lt(self):
        if self.lt is None:
            self.lt = Lt(self.library,WORLD*self.m,self.n,self.k,self.torch,self.device)
        if self.choice is None:
            if not self.tuned:
                raise ValueError("SLT has no frozen algorithm for this rank and shape")
            self.choice = ct.create_string_buffer(bytes.fromhex(self.tuned))

    def tune(self):
        from snapshots import correctness

        self.lt = Lt(self.library,WORLD*self.m,self.n,self.k,self.torch,self.device)
        self.dist.all_to_all_single(self.recv,self.send)
        reference = self.recv.float() @ self.weight.float().T
        scores = []
        for algorithm in self.lt.candidates():
            buf = ct.create_string_buffer(bytes.fromhex(algorithm))
            record = dict(algorithm=algorithm)
            try:
                for _ in range(3):
                    self.lt.execute(self.recv,self.weight,self.out,buf)
                err,nan = correctness(self.out,reference,dtype=self.torch.float16)
                if nan or err>.02:
                    raise ValueError(f"cuBLASLt heuristic numerical failure: {err}, {nan}")
                times = []
                for _ in range(10):
                    begin,end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    self.lt.execute(self.recv,self.weight,self.out,buf)
                    end.record()
                    end.synchronize()
                    times.append(begin.elapsed_time(end))
                record.update(event_ms=statistics.median(times),scaled_max_err=err)
            except (RuntimeError,ValueError) as exc:
                record["error"] = repr(exc)
            scores.append(record)
        eligible = [r for r in scores if "event_ms" in r]
        return dict(algorithms=scores,selected=min(eligible,key=lambda x:x["event_ms"])["algorithm"] if eligible else None)

    def pairwise(self, cutlass):
        dist = self.dist
        for step in range(WORLD):
            src,dst = (self.rank+step)%WORLD,(self.rank-step)%WORLD
            if step:
                requests = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend,self.send.narrow(0,dst*self.m,self.m),dst),
                    dist.P2POp(dist.irecv,self.local[src],src)])
                for op in requests:
                    op.wait()
            else:
                self.local[src].copy_(self.send.narrow(0,src*self.m,self.m))
            if cutlass:
                Lt.check(self.library.edmir_bridge_cutlass(self.m,self.n,self.k,self.local[src].data_ptr(),
                    self.weight_t.data_ptr(),self.outputs[src].data_ptr(),
                    self.torch.cuda.current_stream().cuda_stream))

    def run(self, arm):
        if arm=="BOLD_Q1_G1":
            self.pairwise(cutlass=True)
        elif arm=="SLT":
            self.ensure_lt()
            self.dist.all_to_all_single(self.recv,self.send)
            self.lt.execute(self.recv,self.weight,self.out,self.choice)
        else:
            staged = arm.startswith("BNEW_")
            if staged:
                self.dist.all_to_all_single(self.recv,self.send)
            else:
                self.handle.barrier(0,60_000)
            self.graphs.run(arm,self.consumer(arm))
            if not staged:
                self.handle.barrier(0,60_000)
        return self.out

    def consumer(self, arm):
        """The per-source launches of a D or BNEW arm; only their input rows differ."""
        sources = self.local if arm.startswith("BNEW_") else self.remote
        g = 1 if arm.endswith("G1") else 2
        q = 4 if "_Q4_" in arm else 1

        def issue():
            for x,y in zip(sources,self.outputs,strict=True):
                self.launch(x,self.weight,y,self.partial,self.table,q,g)
        return issue

    def prepare_graphs(self, arms):
        self.graphs.prepare({arm:self.consumer(arm) for arm in arms if consumer_arm(arm)})

    def admit(self, arm):
        from snapshots import correctness

        # Admission runs the same launch mode that timing will use.
        self.prepare_graphs([arm])
        self.dist.all_to_all_single(self.recv,self.send)
        reference = self.recv.float() @ self.weight.float().T
        err,nans = correctness(self.run(arm),reference,dtype=self.torch.float16)
        original = self.send.clone()
        self.send[:,0] = self.rank
        self.send[:,1] = self.torch.arange(len(self.send),device=self.device)//128
        self.send[:,2] = self.torch.arange(len(self.send),device=self.device)%128
        if arm=="BOLD_Q1_G1":
            self.pairwise(cutlass=False)
            observed = self.recv[:,:3]
        elif not arm.startswith("D_"):
            self.dist.all_to_all_single(self.recv,self.send)
            observed = self.recv[:,:3]
        else:
            self.handle.barrier(0,60_000)
            observed = self.torch.cat([x[:,:3] for x in self.remote])
            self.handle.barrier(0,60_000)
        lo = self.rank*self.m
        expected = self.torch.tensor([(src,t//128,t%128) for src in range(WORLD)
                                     for t in range(lo,lo+self.m)],dtype=self.torch.float16,device=self.device)
        result = bool(self.torch.equal(observed,expected))
        self.dist.barrier()
        self.send.copy_(original)
        self.torch.cuda.synchronize()
        return err,nans,result

    def counts(self, arm):
        return dict(input_rows=8*self.m,live_assignments=8*self.m,projection_rows=8*self.m,
                    projection_live_rows=8*self.m,padding_rows=0,
                    projection_launches=1 if arm=="SLT" else 8,merge_launches=8 if "_Q4_" in arm else 0,
                    launch_mode=self.graphs.launch_mode(arm))
