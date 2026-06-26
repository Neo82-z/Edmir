# Overlap As A Critical-Path Question

This note records the current shift in the paper idea:

> The paper should not start from "someone else tried DBO."
> It should start from our own trace-level observation that large communication does not necessarily imply that overlap will improve serving latency.

## Core Observation

In multi-GPU LLM serving, communication can be large in the trace and still be the wrong optimization target.

Possible cases:

- communication is hidden behind useful compute;
- communication is outside the request critical path;
- queueing, runtime overhead, or memory pressure dominates latency;
- overlap hides communication but introduces new exposed overhead.

Therefore, the central question is:

```text
When does overlap remove exposed communication from the serving critical path?
```

## Overlap Is Not A Switch

Overlap changes the execution dependency graph.

For a serving phase:

```text
G_phase --overlap_policy--> G_phase'
```

The optimization is useful only if the new critical path is shorter:

```text
critical_path(G_phase') < critical_path(G_phase)
```

For DBO-like overlap:

```text
Delta_T_phase ~=
  Delta_T_exposed_comm
- T_split
- T_stream
- T_metadata
- T_new_contention
```

Interpretation:

- `Delta_T_exposed_comm`: communication removed from the critical path.
- `T_split`: microbatch split and scheduling overhead.
- `T_stream`: stream/event/synchronization overhead.
- `T_metadata`: attention, KV, position, and request metadata overhead.
- `T_new_contention`: new contention among compute, communication, memory, and hardware execution resources.

## Why DBO-Like Work Is Hard

DBO is difficult because it cuts across multiple layers:

- request and token microbatching;
- attention metadata split and reconstruction;
- KV cache addressability and readiness;
- collective scheduling;
- compute / communication stream ordering;
- CPU runtime scheduling;
- hardware resource contention;
- compatibility across TP / DP / PP / EP regimes.

This means DBO cannot be evaluated only by asking whether communication and compute overlap visually in a timeline.
The real question is whether the overlap shortens the request critical path.

## How To Use Open-Source PRs

Open-source PRs and issues can be used as engineering context:

- they show that the problem is real;
- they identify practical implementation hazards;
- they help define what a usable DBO path must handle.

They should not be used as paper evidence.

Paper evidence must come from:

- our own traces;
- our own ablations;
- our own prediction errors;
- our own controlled comparisons.

## Experimental Shape

Compare:

```text
no overlap
naive overlap
critical-path-aware overlap
```

Report:

- TTFT / ITL / P50 / P90 / P99;
- total communication time;
- exposed communication time;
- split / stream / metadata overhead;
- contention indicators;
- prediction from the overlap gain model.

Success is not "overlap always improves performance."
Success is explaining the regimes:

- overlap-helpful;
- overlap-neutral;
- overlap-harmful.

