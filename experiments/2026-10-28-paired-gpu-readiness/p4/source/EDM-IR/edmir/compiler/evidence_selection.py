"""Candidate comparison with explicit evidence and unresolved outcomes.

No training, model probabilities, architecture-to-latency conversion, or
automatic compiler admission. Measured ranges describe supplied process
medians; they are not confidence intervals or guarantees about future runs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .execution_ledger import SCHEMA as LEDGER_SCHEMA, ALL_FAMILIES, digest

SCHEMA = "edmir.evidence-selection.v2"


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class TimingEvidence:
    """Caller-supplied evidence; source IDs trace claims, not authenticate them.

    Only independent process/block medians belong here, never individual
    iterations from one process. All compared candidates must share blocks
    and context. The caller/collector must establish independence and source
    truth; this layer checks consistency and makes a conditional comparison.
    """
    identity_sha256: str
    measurement_context: str
    process_medians_us: tuple[tuple[str, float], ...]
    sources: tuple[str, ...]
    correctness_passed: bool
    profiler: bool = False

    def validate(self, ledger):
        if ledger["identity"].get("algorithm_id") is None:
            raise ValueError("record the runtime library algorithm before attaching timing evidence")
        if self.identity_sha256 != ledger["identity_sha256"]:
            raise ValueError("timing evidence does not match shape/implementation/environment/source/timer")
        if not isinstance(self.measurement_context, str) or not self.measurement_context.strip():
            raise ValueError("measurement context required")
        if self.correctness_passed is not True or self.profiler is not False:
            raise ValueError("only correctness-checked, unprofiled timing evidence is comparable")
        samples = self.process_medians_us
        if not isinstance(samples, tuple) or not 3 <= len(samples) <= 32:
            raise ValueError("3..32 independent process/block medians required")
        if any(not isinstance(s, tuple) or len(s) != 2 for s in samples):
            raise ValueError("each observation needs a block ID and its median")
        if (any(not isinstance(b, str) or not b.strip() or not _positive(t) for b, t in samples)
                or len({b for b, _ in samples}) != len(samples)):
            raise ValueError("unique block IDs and finite positive medians required")
        if (not isinstance(self.sources, tuple) or not self.sources
                or any(not isinstance(s, str) or not s.strip() for s in self.sources)):
            raise ValueError("measurement sources required")


def assess_candidates(ledgers: Mapping, *, evidence: Mapping | None = None,
                      model_predictions_us: Mapping | None = None, minimum_margin_us=0.0):
    """Compare both host and fused symmetrically; absent evidence is never zero cost.

    A winner requires its largest supplied process median plus the declared
    margin to be below every rival's smallest median. Missing candidates remain
    unresolved. Point predictions are diagnostic only, regardless of margin.
    """
    if not isinstance(ledgers, Mapping) or not 1 <= len(ledgers) <= len(ALL_FAMILIES):
        raise ValueError("nonempty bounded candidate ledgers required")
    if type(minimum_margin_us) not in (int, float) or not math.isfinite(minimum_margin_us) or minimum_margin_us < 0:
        raise ValueError("finite nonnegative comparison margin required")
    names = sorted(ledgers)
    contexts = []
    for name in names:
        ledger = ledgers[name]
        if ledger.get("schema") != LEDGER_SCHEMA:
            raise ValueError("unknown ledger schema")
        ident = ledger["identity"]
        if ident["family"] != name or digest(ident) != ledger["identity_sha256"]:
            raise ValueError("ledger identity is inconsistent")
        if (digest(ledger["execution_graph"]) != ident["execution_graph_digest"] or
                digest(ledger["provenance"]) != ident["source_digest"]):
            raise ValueError("ledger execution graph or source identity is inconsistent")
        context = ident["comparison_context"]
        if (any(context[k] != ident[k] for k in ("shape", "environment_id", "timing_scope")) or
                context["timer_source_sha256"] != ledger["provenance"]["project"]["timer"]["sha256"]):
            raise ValueError("ledger comparison context is inconsistent")
        contexts.append(context)
    if any(c != contexts[0] for c in contexts):
        raise ValueError("do not compare different workloads, environments or measurement contracts")
    evidence = {} if evidence is None else evidence
    predictions = {} if model_predictions_us is None else model_predictions_us
    if not isinstance(evidence, Mapping) or not isinstance(predictions, Mapping):
        raise TypeError("evidence and predictions must be candidate mappings")
    if set(evidence)-set(names) or set(predictions)-set(names):
        raise ValueError("evidence/prediction names must belong to the requested candidate set")
    if any(not _positive(t) for t in predictions.values()):
        raise ValueError("model predictions must be finite positive microseconds")
    intervals, measured_contexts = {}, []
    for name, item in sorted(evidence.items()):
        if not isinstance(item, TimingEvidence):
            raise TypeError("explicit TimingEvidence required, not fitted or archived point labels")
        item.validate(ledgers[name])
        samples = [v for _, v in item.process_medians_us]
        intervals[name] = {"lower_us": min(samples), "upper_us": max(samples),
                           "basis": "supplied_process_median_range", "confidence_interval": False,
                           "provenance": asdict(item)}
        measured_contexts.append((item.measurement_context, frozenset(b for b, _ in item.process_medians_us)))
    if any(c != measured_contexts[0] for c in measured_contexts):
        raise ValueError("compared measurements must use the same context and matched process blocks")
    beaten_by = {n: [] for n in names}
    for name in names:
        if name not in intervals:
            continue
        beaten_by[name] = [r for r in names if r != name and r in intervals
                           and intervals[r]["upper_us"] + minimum_margin_us < intervals[name]["lower_us"]]
    retained = [n for n in names if not beaten_by[n]]
    complete = set(intervals) == set(names)
    winner = retained[0] if complete and len(retained) == 1 and len(names) > 1 else None
    status = ("insufficient_comparison" if len(names) == 1 else
              "separated_under_supplied_evidence" if winner else
              "unresolved_overlap" if complete else "insufficient_evidence")
    return {
        "schema": SCHEMA, "status": status, "choice": winner,
        "retained_candidates": retained, "separated_from_best": {n: r for n, r in beaten_by.items() if r},
        "missing_evidence": [n for n in names if n not in intervals],
        "intervals_us": intervals, "minimum_margin_us": minimum_margin_us,
        "reason": ("one candidate has separated process-median ranges against every requested rival"
                   if winner else "no evidence-supported unique winner; ties and unknown costs remain candidates"),
        "model_reference": {"predictions_us": dict(sorted(predictions.items())),
                            "used_for_selection": False,
                            "meaning": "conditional fitted estimates, not observed times or optimality probabilities"},
        "limitations": ["conditional on the supplied source and independence claims",
                        "observed process ranges do not bound future latency",
                        "only the explicitly requested candidate set is compared"],
        "eligible_for_compiler_admission": False,
    }
