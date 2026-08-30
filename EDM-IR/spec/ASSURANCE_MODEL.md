# EDMIR assurance model

This specification fixes the assurance boundary before the next compiler and
inference integration changes. It is intentionally narrower than a verified
inference runtime and wider than a kernel unit test.

## Boundary

The current executable operation begins with immutable routed activation tiles
and local expert weights. It ends when every destination rank has produced the
expert GEMM output for every source tile:

```text
router trace -> routed source tiles -> dispatch/remote read -> expert GEMM -> output tiles
```

The router, continuous-batching policy, paged KV cache, attention kernels,
NCCL transport internals, and CUTLASS instruction generation are external.
They enter through versioned problem, environment, and trace contracts. EDMIR
assures its use of those contracts; it does not reimplement or verify their
internals. Producer-generated or mutable source activations are a separate
protocol boundary. Until publication events are Program-visible and backed by
runtime primitives, the verifier rejects that mode rather than treating a
remote pointer as ready by implication.

For source rank `s`, destination/expert rank `d`, and chunk `c`, the reference
semantics are:

```text
Y[s,d,c] = X[s,d,c] @ W[d]
```

Concatenating `Y[s,d,c]` in the declared source/chunk order gives the
destination-local expert output. Host staging, direct LSA, split-K, and
grouped-N are alternative schedules for this same relation.

## Three models

EDMIR keeps three kinds of reasoning separate.

1. **Semantic model.** Exact tensor identity, ownership, coverage, dependency,
   and lifecycle obligations. A violation is a compiler or runtime error.
2. **Performance ledger.** FLOPs, local/remote bytes, CTA supply, replay,
   synchronization, and reduction terms. It explains and prunes candidates;
   it is not an absolute latency proof.
3. **Admission evidence.** Target-bound numerical and timing observations.
   A candidate is deployable only when its evidence matches the exact problem,
   code, protocol, raw source, and environment and clears a conservative host
   margin.

The admission implication is:

```text
Admit(candidate, problem, environment)
  => Legal(candidate, problem)
  && SemanticEquivalent(candidate, host)
  && EnvironmentMatches(evidence, environment)
  && SourceMatches(evidence, observation)
  && NumericalCheckPassed(evidence)
  && CandidateUpperBound < HostLowerBound * (1 - safety_margin)
```

Failure or missing evidence selects host. No analytic or learned latency model
can bypass this implication.

## Invariants

The identifiers below remain stable so tests, model-checker properties, and
paper claims can cite the same obligations.

| ID | Invariant |
|---|---|
| SEM-1 | Every declared `(source, destination, chunk)` tile is produced exactly once. |
| SEM-2 | Every output tile reads the matching source tile and destination-local weight. |
| SEM-3 | Host, pipelined, and fused schedules implement the same reference relation. |
| MEM-1 | A consumer cannot read a staging slot before its producer publishes it. |
| MEM-2 | A staging slot cannot be overwritten until all prior readers release it. |
| MEM-3 | Fused schedules do not materialize a hidden program-visible staging buffer. |
| PUB-1 | A remote consumer acquires only an epoch that its producer has published. |
| PUB-2 | A remote consumer consumes an epoch only after its matching acquire. |
| PUB-3 | An epoch retires only after every declared consumer completes. |
| PUB-4 | A producer overwrites an epoch only after every declared consumer completes. |
| RED-1 | Split-K partitions are disjoint and their union is exactly `[0, K)`. |
| RED-2 | Reduction completion requires every declared split partial. |
| REUSE-1 | Grouped-N changes remote-tile reuse and visit order, not output ownership. |
| ENV-1 | Evidence is invalid after any problem, environment, code, protocol, or raw-source mismatch. |
| ADM-1 | Missing correctness or insufficient samples deterministically select host. |
| ADM-2 | Overlapping timing intervals deterministically select host. |

## Assurance methods

| Property class | Method | Claim strength |
|---|---|---|
| Tensor mapping and coverage | Static verifier plus executable reference | Checked for every compiled Program |
| Algebraic schedule equivalence | Deterministic and randomized property tests | Tested over the declared matrix |
| Buffer publication, reuse, and split completion | Derived hazards plus bounded TLA+/TLC model | Model-checked for the recorded abstract bounds; runtime correspondence remains tested, not proved |
| Floating-point behavior | CPU reference, error ledger, adversarial inputs | Numerically validated, not bitwise assumed |
| Performance safety | Target-bound confidence intervals and fallback | Statistical admission, not formal proof |
| End-to-end inference effect | Controlled trace replay and serving experiment | Empirical causal claim |

The project must use the phrases **proved**, **model-checked**, **statically
verified**, **numerically validated**, and **measured** only for the matching
row above.

## Falsifiability and provenance

- Calibration and held-out groups are immutable once their protocol hash is
  published.
- Oracle, selected, and fallback results are reported separately.
- A held-out failure remains a failure; changing a rule creates a new protocol.
- Raw observations are append-only and addressed by SHA-256.
- Mutation tests must demonstrate that the verifier detects missing
  dependencies, wrong tensor mappings, corrupted outputs, and evidence
  mismatches.
- If measurements reject the performance interpretation, the interpretation
  changes. Candidate families or thresholds are not silently tuned around the
  held-out result.

## Bounded formal artifact

The first TLA+ model covers only the host staging-slot lifecycle:

```text
Free -> Producing -> Ready -> Consuming -> Released -> Producing
```

The retained TLC run checks `MEM-1`, `MEM-2`, `RED-2`, structural typing,
unique residency, deadlock freedom, and eventual completion.  The expanded
safe bound uses three slots, four tiles, and three splits: TLC generates
254,566 states and exhausts 63,295 distinct states without a violation.  Three
mutated configurations admit early read, early reuse, or incomplete release;
each produces the expected counterexample.  The model, configurations, raw
output, hashes, and reproduction runner live under `spec/tla/`.

`RemotePublication.tla` separately models one reusable producer buffer across
versioned epochs and multiple remote consumers.  The retained safe runs cover
two epochs/two consumers and three epochs/three consumers.  TLC exhausts 23 and
88 distinct states without a violation.  Four mutated configurations permit
acquire-before-publish, consume-before-acquire, incomplete retirement, or
premature overwrite; each produces the expected counterexample.

These are bounded model-checking claims about abstract compiler-owned
protocols.  They are not proofs of arbitrary bounds, CUDA execution, NCCL, or
the CUTLASS semaphore implementation.  `validate_slot_correspondence` projects
each Program-visible host staging epoch onto the first lifecycle.  The versioned
`remote_input_contract` is the bridge for the second: immutable prelaunch input
is executable, while `explicit_epoch` remains fail-closed until publish,
acquire, completion, and reuse events are Program-visible.
