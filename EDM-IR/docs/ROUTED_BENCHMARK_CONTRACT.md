# Routed benchmark contract

Status: D7 accepted and frozen on 2026-09-08.  A semantic or measurement
change requires a new protocol version; old rows cannot be relabeled as
held-out evidence.

## Boundary and canonical layout

The primary boundary begins with post-top-k route records, source-rank
activations, gate weights, expert placement, and preloaded expert weights.  It
ends with expert-GEMM results plus an inverse-route index in a canonical
combine-input representation.

Input records are source-major and ordered by:

1. source rank;
2. global token identity;
3. top-k route slot.

Canonical outputs are ordered by:

1. destination rank;
2. expert identity;
3. source rank;
4. global token identity;
5. route slot.

Empty experts are represented by zero-length spans.  A backend may use a
private layout, but routing-dependent permutation, metadata movement, and
conversion to the canonical output are timed.  Every route is consumed once;
capacity truncation and token dropping are illegal.

## Analytical traffic ledger

The ledger is descriptive and must not be called measured link traffic.  For
each trace and schedule it records:

- per-rank unique and replayed remote TX/RX bytes;
- maximum-rank and maximum directional-cut loads;
- nonempty source--destination--expert buckets and bucket-size p50/p95;
- per-expert rows, routing skew, and normalized entropy;
- N-tile replay and per-rank CTA supply.

The reference implementation is
`edmir.compiler.routed_contract.build_routed_traffic_ledger`.

## Fragmentation sentinel

The sentinel varies only source--expert bucket decomposition.  Its low,
center, and high fragmentation traces must preserve:

- world size, phase, dtype, K, and N;
- total rows and per-source/per-destination row marginals;
- expert placement and per-expert row totals;
- unique and replayed per-rank traffic;
- replay and CTA supply.

Only the number and distribution of nonempty remote buckets may change.  The
sentinel measures the causal materialized host path, direct LSA, and one
preregistered split-K family.  It does not train the selector or enlarge the
finite-search oracle.  A linear per-bucket latency term is not assumed in
advance; the experiment decides whether fragmentation warrants an explicit
model term.

`edmir.compiler.routed_contract.validate_fragmentation_sentinel` rejects a
sentinel set that changes a frozen confounder.

## Current exclusions

Multi-node transport, InfiniBand, SHARP, end-to-end serving, and cross-dtype
pooling are outside this contract.  BF16 is the primary routed stratum; current
FP16 measurements remain mechanism evidence only.
