# Progress Log

This log records public-safe milestones for the MLSys 2027 paper direction.
It should not contain private draft text, raw unpublished results, or local LaTeX artifacts.

## 2026-06-29: EDM-IR Direction Recovered

Today the paper direction shifted from an overlap-centered framing to an IR-centered framing:

```text
When Does Overlap Help?
  -> EDM-IR: Exposed Data-Movement Analysis for Multi-GPU LLM Serving
```

The key realization:

```text
DBO is not the paper's foundation.
DBO is one graph transformation that EDM-IR should be able to analyze.
```

The actual foundation is:

```text
raw serving trace
  -> EDM uops
  -> dependency graph
  -> critical path
  -> exposed communication / ECR
  -> overlap-gain classification
```

Local private LaTeX draft status:

- abstract and introduction reframed around EDM-IR;
- model section extended with EDM uops and EDM Graph;
- methodology section reframed around trace-to-IR construction;
- experiments restructured into synthetic EDM-IR, dense TP trace, controlled overlap probe, and DBO case study;
- PDF successfully compiled locally.

Public repo boundary:

```text
LaTeX source and PDF remain local/private.
Only research notes and artifact plans are committed here.
```

Next main technical task:

```text
Build a tinygrad-inspired EDM-IR generator/analyzer subset.
```

The first target is intentionally small:

```text
synthetic_hidden_comm.json
synthetic_exposed_comm.json
synthetic_overlap_harmful.json

-> print uops
-> build graph
-> compute critical path
-> compute total/exposed communication
-> classify overlap as helpful / neutral / harmful
```

Important boundary:

```text
Do not build a full DBO scheduler first.
Do not rewrite vLLM first.
Do not add DeepEP / remote KV / RDMA first.
```

The first artifact should prove that ECR is computed from uops and edges rather than manually inferred from profiler screenshots.

