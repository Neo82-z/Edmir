"""Audit the frozen fused program's queue constraints without changing them."""
from pathlib import Path
from collections import Counter
import hashlib, json, sys

here = Path(__file__).resolve().parent
source = here.parent/'bundle/source/EDM-IR'
sys.path.insert(0,str(source))
from edmir.compiler.model import A2AGemmProblem
from edmir.compiler.schedule import build_fused_program
from edmir.compiler.dependence import derive_dependencies

p = build_fused_program(A2AGemmProblem(4,512,4096,512),chunk_rows=512)
d = derive_dependencies(p)
d.require_valid()
paths = {}
predecessors = d.predecessors()
for op in d.topological_order:
    paths[op] = max((paths[v] for v in predecessors[op]), key=len, default=[]) + [op]
chain = max(paths.values(),key=len)
ops = p.op_map()
output_buffers = [name for op in p.ops for name in op.writes]
read_buffers = {name for op in p.ops for name in op.reads}
assert len(output_buffers)==len(set(output_buffers))==16
assert not set(output_buffers) & read_buffers
assert all(x.kind=='queue' for x in d.dependencies)
relative = ('edmir/compiler/model.py','edmir/compiler/dependence.py',
            'edmir/compiler/schedule.py','runtime/src/cuda_backend.cpp','runtime/src/prepared_execution.cpp')
obj = {
    'source':str(source),
    'source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in relative},
    'shape':[4,512,4096,512], 'ops':len(p.ops), 'edges':len(d.dependencies),
    'edge_kinds':dict(Counter(x.kind for x in d.dependencies)),
    'longest_unit_weight_op_chain':chain,
    'chain_source_destination_pairs':[[ops[o].peer_rank,ops[o].rank] for o in chain],
    'declared_resources':{o.id:list(o.resources()) for o in p.ops},
    'outputs_disjoint_and_never_read':True,
    'counterfactual_compute_stream_only_chain_length':4,
    'counterfactual_scope':'Illustrative graph bound only. Removing queue edges is not implemented or validated; peer-window lifetime, completion ordering and contention need review.',
    'finding':'The current IR serializes fused work through both endpoint communication resources even though the kernel reads peer memory. This is an execution-policy constraint, not a proven RAW/WAR/WAW data hazard for these immutable inputs.',
    'runtime_scope':'The event lowering and shared communication streams can add timing constraints. The unit-weight IR chain is not a latency prediction or an exact reconstruction of every runtime wait.',
    'compiler_admission':False, 'runtime_changed':False,
}
(here/'queue-dependency-audit.json').write_text(json.dumps(obj,indent=2)+'\n')
print('ops',len(p.ops),'queue_edges',len(d.dependencies),'longest_chain',len(chain))
