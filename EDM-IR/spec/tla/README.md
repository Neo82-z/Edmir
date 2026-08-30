# Bounded concurrency models

These deliberately small models cover compiler protocol boundaries, not the
whole inference stack.

## Host staging slots

`SlotLifecycle.tla` is the first model-checked EDMIR concurrency boundary.  It
models only compiler-owned staging publication, consumption, split completion,
and slot reuse:

```text
Free -> Producing -> Ready -> Consuming -> Released -> Producing
```

It checks:

- `MEM-1` / `ReadAfterPublish`: consumption cannot begin before publication;
- `MEM-2` / `NoPrematureReuse`: an incomplete tile cannot lose its resident
  slot;
- `RED-2` / `CompleteReduction`: release requires every declared split;
- structural typing, unique residency, deadlock freedom, and bounded eventual
  completion.

The three unsafe configurations are intentional mutation witnesses.  They
permit an active slot to be overwritten, a producing slot to be read, or a tile
to be released before every split finishes.  They must violate
`NoPrematureReuse`, `ReadAfterPublish`, and `CompleteReduction`, respectively.
A passing safe configuration without those counterexamples is insufficient
evidence that the properties are testing the intended mistakes.  The expanded
safe configuration increases slot, tile, and split counts to expose more
interleavings than the quick configuration.

This model does **not** verify CUTLASS's internal serial split-K semaphore,
NCCL's transport implementation, CUDA's scheduler, floating-point accuracy, or
the full inference runtime.  Those remain library contracts or separate
numerical/empirical obligations.

## Reproduce

Use the official `tla2tools.jar` release with Java 11 or newer:

```bash
export TLA2TOOLS_JAR=/path/to/tla2tools.jar
python3 tools/run_tlc.py --workers 1
```

The runner checks two safe bounds and three expected counterexamples.  It writes
raw TLC output plus `results/summary.json`; the ordinary Python test suite
rejects the retained report if the model or a configuration changes without a
new TLC run.

`compiler/slot_protocol.py` is the implementation bridge.  It projects each
host staging-buffer epoch onto `BeginProduce`, `Publish`, `StartConsume`, one
abstract split completion, and `Release`, and requires the corresponding RAW
and buffer-reuse dependency edges.  Fused Programs are explicitly reported as
not applicable because their CUTLASS semaphore is not Program-visible.

## Mutable remote publication

`RemotePublication.tla` models a reusable producer buffer whose contents are
identified by epochs.  Each declared remote consumer must acquire the matching
published epoch, consume it, and complete before retirement or overwrite.  It
checks `PUB-1..4` as `AcquireAfterPublish`, `ConsumeAfterAcquire`,
`CompleteConsumption`, and `NoOverwriteBeforeConsume`, plus bounded eventual
completion.

The four unsafe configurations independently enable early acquire, early
consume, incomplete retirement, or premature overwrite.  Each must produce the
corresponding counterexample.  Reproduce and retain this suite with:

```bash
export TLA2TOOLS_JAR=/path/to/tla2tools.jar
python3 tools/run_publication_tlc.py --workers 1
```

The runner writes `publication_results/summary.json` and raw outputs.  The
compiler-side `remote_input_contract` admits only `immutable_external` today.
`explicit_epoch` describes the future contract but remains non-executable until
publish, acquire, completion, and reuse are represented by Program and runtime
events.
