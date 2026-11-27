#!/usr/bin/env python3
"""Run the exploratory scripts on a machine without a GPU, to check their tables before GPU time.

Eight threads stand in for the eight ranks. The collectives, the symmetric-memory rendezvous
and the barriers are replaced by thread barriers; a peer buffer is the other thread's tensor,
so the distance between two buffers is a real address difference, as on the device. Every
kernel launch is replaced by FP32 arithmetic that follows the same tables: a row is read from
the address base + offset, a tile takes the rows and the expert its table entry names, and the
result goes to the output row the table names. A read outside every known buffer raises.

What this checks: routing tables, row offsets, tile tables, output order, the admission and
timing plumbing, and the output files. What it cannot check: the kernels themselves, CUDA
graphs, the native arm (left out), and any time. Times in the output files are meaningless.

  cpu_emulation.py snapshots --gate gate.json --run RUN_DIR --snapshots DIR --unit GLM_DECODE --arms ... --out x.json
  cpu_emulation.py variants --gate gate.json --slt slt.json --unit GLM_M0004 --arms ... --out x.json

The arguments after the first are those of snapshot_variants.py or consumer_variants.py.
Run it in the locked image with the experiment package and this folder on PYTHONPATH.
"""
import contextlib
import ctypes
import sys
import threading
import time

WORLD = 8
me = threading.local()


class World:
    """Every rank deposits a value; all eight see all values before any rank moves on."""

    def __init__(self):
        self.barrier,self.slots = threading.Barrier(WORLD,timeout=600),[None]*WORLD

    def exchange(self, value=None, then=None):
        self.slots[me.rank] = value
        self.barrier.wait()
        result = list(self.slots)
        if then is not None:
            then(result)
        self.barrier.wait()
        return result


def install(world):
    import numpy
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symmetric

    import balanced
    import gate_up
    import replay
    import variant_kernels

    known = set()  # (first address, one past the last) of every buffer a kernel may read

    def note(tensor):
        known.add((tensor.data_ptr(),tensor.data_ptr()+tensor.numel()*tensor.element_size()))
        return tensor

    def rows_at(base, offsets, k):
        """Rows of k elements read at base + offset elements, as a kernel would address them."""
        size = base.element_size()
        out = torch.empty((len(offsets),k),dtype=base.dtype)
        raw = out.view(torch.int16).numpy()
        for i,offset in enumerate(offsets):
            address = base.data_ptr()+int(offset)*size
            if not any(lo<=address and address+k*size<=hi for lo,hi in tuple(known)):
                raise ValueError(f"row {i} is read outside every buffer (offset {int(offset)})")
            raw[i] = numpy.frombuffer((ctypes.c_char*(k*size)).from_address(address),dtype=numpy.int16)
        return out

    def project(x, weight, q):
        k = x.shape[1]
        parts = [x[:,p*(k//q):(p+1)*(k//q)].float() @ weight[:,p*(k//q):(p+1)*(k//q)].float().T for p in range(q)]
        return sum(parts[1:],parts[0])

    def launch(x, weight, out, partial, table, q=1, g=2):
        n,k = weight.shape[-2:]
        note(x)
        for begin,stop,expert in table.tolist():
            stop = min(stop,begin+gate_up.TILE_M)
            rows = rows_at(x,[r*k for r in range(begin,stop)],k)
            out[begin:stop] = project(rows,weight.reshape(-1,n,k)[expert],q).to(out.dtype)

    def launch_rows(base, held, out_rows, weight, out, partial, work, k, q, bm, g, bk=32, unit=1):
        n = out.shape[1]
        if (q>1)!=(partial is not None) or (q>1 and partial.shape!=(q,*out.shape)):
            raise ValueError("partial buffer does not match q")
        note(base)
        written = torch.zeros(out.shape[0],dtype=torch.int32)
        for begin,stop,expert in work.tolist():
            index = list(range(begin,min(stop,begin+bm)))
            rows = rows_at(base,[int(held[r])*unit for r in index],k)
            target = out_rows[index].long()
            out[target] = project(rows,weight.reshape(-1,n,k)[expert],q).to(out.dtype)
            written[target] += 1
        if not bool((written==1).all()):
            raise ValueError(f"tiles write {int((written!=1).sum())} output rows zero or several times")

    def launch_copy(base, held, out, unit=1):
        note(base)
        out.copy_(rows_at(base,[int(h)*unit for h in held[:out.shape[0]]],out.shape[1]))

    def launch_block_copy(source, out, ctas=432):
        if source.shape!=out.shape:
            raise ValueError("block copy needs equal shapes")
        out.copy_(source)

    class Handle:
        world_size = WORLD

        def __init__(self, tensors):
            self.tensors = tensors

        def get_buffer(self, rank, shape, dtype, offset):
            tensor = self.tensors[rank]
            if tuple(tensor.shape)!=tuple(shape) or tensor.dtype!=dtype or offset:
                raise ValueError("peer view differs from the peer's buffer")
            return tensor

        def barrier(self, channel, timeout):
            world.exchange()

    def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None):
        def deliver(slots):
            parts = []
            for tensor,sizes in slots:
                sizes = sizes or [len(tensor)//WORLD]*WORLD
                parts.append(tensor.narrow(0,sum(sizes[:me.rank]),sizes[me.rank]))
            if output_split_sizes is not None and [len(p) for p in parts]!=list(output_split_sizes):
                raise ValueError("receive sizes differ from what the sources send")
            if parts:
                output[:sum(map(len,parts))] = torch.cat(parts)
        world.exchange((input,input_split_sizes),deliver)

    def broadcast(tensor, src, group=None):
        world.exchange(tensor,lambda slots: None if me.rank==src else tensor.copy_(slots[src]))

    class Event:
        def __init__(self, enable_timing=False):
            self.at = 0.

        def record(self):
            self.at = time.perf_counter()

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return (other.at-self.at)*1e3

    class Graphs(balanced.ConsumerGraphs):
        """Replay calls the launch function again; nothing is captured."""

        def prepare(self, launches, group="consumer"):
            todo = {arm:fn for arm,fn in launches.items() if arm not in self.graphs}
            if not todo:
                return
            for fn in todo.values():
                fn()
            if any(self.agree(None)):
                raise RuntimeError("ranks disagree in emulation")
            self.modes[group] = "graph"
            self.graphs.update({arm:type("Replay",(),dict(replay=staticmethod(fn))) for arm,fn in todo.items()})
            self.group_of.update(dict.fromkeys(todo,group))

    class Lt:
        def __init__(self, library, m, n, k, torch_, device):
            pass

        def close(self):
            pass

        def execute(self, a, b, out, algorithm):
            out.copy_((a.float() @ b.float().T).to(out.dtype))

    @contextlib.contextmanager
    def gpu_session(expected_uuids):
        yield me.rank,torch.device("cpu"),"cpu-group"

    mm = torch.mm

    def mm_fp32(a, b, out=None):
        if a.dtype in (torch.bfloat16,torch.float16):
            return out.copy_((a.float() @ b.float()).to(out.dtype))
        return mm(a,b,out=out)

    symmetric.empty = lambda shape,dtype,device: note(torch.zeros(tuple(shape),dtype=dtype))
    symmetric.rendezvous = lambda tensor,group: Handle(world.exchange(tensor))
    symmetric.get_backend = lambda device: "CUDA"
    dist.all_to_all_single,dist.broadcast = all_to_all_single,broadcast
    dist.barrier = lambda *a,**k: world.exchange()
    torch.cuda.synchronize = lambda *a,**k: None
    torch.cuda.Event = Event
    torch.mm = mm_fp32
    for module in (replay,balanced):
        module.ConsumerGraphs = Graphs
    replay.gpu_session = gpu_session
    replay.hardware = lambda path: None
    replay.inventory = lambda: [f"GPU-emulated-{i}" for i in range(WORLD)]
    replay.spread = lambda value,group: world.exchange(value)
    balanced.Lt,balanced.load_library = Lt,lambda path: None
    gate_up.launch = launch
    variant_kernels.launch_rows,variant_kernels.launch_copy = launch_rows,launch_copy
    variant_kernels.launch_block_copy = launch_block_copy
    torch.set_num_threads(1)


def main():
    kind = sys.argv.pop(1)
    world = World()
    install(world)
    import consumer_variants
    import snapshot_variants
    script = dict(snapshots=snapshot_variants,variants=consumer_variants)[kind]
    script.POLICY = dict(warmup=0,timed=1)
    failures = []

    def rank_main(rank):
        me.rank = rank
        try:
            script.main()
        except BaseException as exc:
            failures.append((rank,repr(exc)))
            world.barrier.abort()
            raise

    threads = [threading.Thread(target=rank_main,args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"emulation failed: {failures[:2]}")


if __name__=="__main__":
    main()
