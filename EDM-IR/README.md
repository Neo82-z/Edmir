# EDM-IR

EDM-IR deliberately keeps compilation and analysis different:

```text
Program(Buffer, Op) --derive dependencies--> EventGraph --analyze--> ECR / EDMR
profiler trace ----------------------------> EventGraph --analyze--> ECR / EDMR
```

The compiler IR is an ordered, executable Tile/communication program. It does
not store dependency edges. An operation declares only its queue plus the
physical buffers it reads and writes; queue order and RAW/WAR/WAW hazards are
derived in one place. The resulting graph is consumed by both the legality
checker and the deterministic cost simulator.

The current artifact does not generate or execute CUDA, Triton, CuTe, or NCCL
Device API code.

## Minimal compiler model

The core types are:

```python
Buffer(name, rank, role, size_bytes)
Op(id, kind, rank, queue, reads, writes, tile, rows, peer_rank)
Program(problem, strategy, chunk_rows, buffer_count, buffers, ops)
```

[`examples/minimal_program.py`](examples/minimal_program.py) constructs a
complete two-rank program directly and prints every automatically derived
dependency.

The first version has only three executable op kinds:

- `transfer`: move one All2All tile from `peer_rank` to `rank` on both ranks'
  communication resources;
- `gemm`: consume a received staging tile on the destination compute queue;
- `barrier`: create a readiness token for the sequential baseline.

`ProgramBuilder` constructs those ops directly. `derive_dependencies()` then
applies five rules:

1. operations sharing a hardware resource keep program order;
2. the last writer precedes every reader (RAW/data readiness);
3. prior readers precede a buffer overwrite (WAR/buffer reuse);
4. consecutive writers are ordered (WAW);
5. a transfer occupies both source and destination communication resources.

All derived buffer edges carry release/acquire semantics. There is no second
edge list for schedule generation, verification, and simulation to disagree on.

## CPU-only quick start

```bash
python -m unittest discover -s tests -p '*_test.py' -v

python -m edmir compile examples/a2a_gemm_problem.json \
  --profile examples/synthetic_profile.json \
  --topology examples/full_p2p_topology.json \
  --chunk-rows 16,32,64 --buffers 1,2,4 \
  --out /tmp/edmir-program.json

python -m edmir verify /tmp/edmir-program.json
python -m edmir inspect /tmp/edmir-program.json --dependencies
python -m edmir simulate /tmp/edmir-program.json \
  --profile examples/synthetic_profile.json
```

The output schema is `edmir.program.v1`. Compiler-generated `edmir.plan.v1`
files from the first prototype remain readable and are rebuilt as Programs;
their serialized edge lists are intentionally discarded.

The bundled profile and topology are synthetic fixtures, not experimental
results. Use `python -m edmir probe --out topology.json` on the target machine
and replace profile coefficients with measured values before drawing any
performance conclusion.

## Code map

```text
compiler/model.py       Buffer, Op, Program, problem/profile/topology schemas
compiler/schedule.py    direct ProgramBuilder and finite candidate construction
compiler/dependence.py  the only queue and buffer dependency implementation
compiler/verify.py      All2All->GEMM semantic checks
compiler/simulate.py    ready-time/resource interpreter and EventGraph emission
compiler/compile.py     candidate enumeration, selection, and fallback
analysis.py             critical path and counterfactual exposure on EventGraph
```

## v0.1 boundary

- static world size and shapes;
- row-major equal-sized All2All blocks;
- cyclic peer order;
- one destination-local GEMM per received row tile;
- one communication, compute, and control resource model per rank;
- sequential fallback when P2P connectivity or predicted gain is absent.

Dynamic MoE routing, ragged tokens, Grouped GEMM, multi-node networking, true
collective rendezvous, and GPU code generation remain outside this version.
