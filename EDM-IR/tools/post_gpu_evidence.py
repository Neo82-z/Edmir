"""Import sealed old/new measurements without pooling repetitions or contexts."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from edmir.compiler.mechanism_prediction import HOSTS, POLICY

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO/"experiments/2026-10-25-runpod-hour-01"
OLD = REPO/"experiments/2026-10-25-final-local-revision/results"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_context(context):
    return {k:v for k,v in context.items() if k != "stage"}


def import_stage(configs, context, raw_path, expected_hash, collection, stage):
    if sha(raw_path) != expected_hash:
        raise ValueError(f"raw measurement hash mismatch: {raw_path}")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    grouped = defaultdict(list)
    for row in raw:
        grouped[(tuple(row["shape"]), row["candidate_id"])].append(row)
    output = []
    for row in configs:
        if row.get("warmup", 20) != 20 or not row.get("primary_timing_label", True):
            continue
        key = (tuple(row["shape"]), row["candidate_id"])
        trials = sorted(grouped.pop(key), key=lambda r:r["repetition"])
        if len(trials) != 3 or [t["repetition"] for t in trials] != [0,1,2]:
            raise ValueError("three distinct process blocks required")
        identity_fields = ("family", "implementation_id", "metadata_mode", "warmup", "iterations",
                           "timing_scope", "input_sha256", "reference_sha256", "runtime")
        first = trials[0]
        if any(any(t[k] != first[k] for k in identity_fields) for t in trials):
            raise ValueError("within-configuration identity drift")
        for t in trials:
            if (not t["correctness_passed"] or t["nonfinite_outputs"] or t["iterations"] != 100
                    or len(t["samples_ms"]) != 100 or t["runtime"] != context["runtime"]
                    or t["measurement_policy"] != {"profiler":False,"nvtx":False,"discarded_samples":0}):
                raise ValueError("measurement protocol mismatch")
        medians = [statistics.median(t["samples_ms"])*1000 for t in trials]
        if any(abs(a-b)>1e-7 for a,b in zip(medians,row["process_medians_us"])):
            raise ValueError("sealed review disagrees with raw timing")
        if abs(statistics.median(medians)-row["observed_us"])>1e-7:
            raise ValueError("sealed aggregate label mismatch")
        if sorted({t["gemm_algorithm_id"] for t in trials}) != sorted(row["algorithm_ids"]):
            raise ValueError("algorithm identity mismatch")
        family = first["family"]
        output.append({**row, **{k:first[k] for k in identity_fields if k != "runtime"},
            "collection":collection, "stage":stage, "context_id":digest(normalize_context(context)),
            "config_id":digest([collection, stage, key]), "primary_scope":True,
            "eligible_for_calibration":not row["quality_flags"],
            "dependency_policy":None if family in HOSTS else POLICY,
            "raw_source":str(raw_path.relative_to(REPO))})
    if grouped and stage != "warmup":
        raise ValueError("unreviewed raw configurations")
    return output


def load_evidence():
    seal = read(CAMPAIGN/"analysis-seal.json")
    names = ("review-primary-sealed.json", "review-recheck-sealed.json", "review-baseline-extension-local.json")
    for name in names:
        if sha(CAMPAIGN/name) != seal["sources_and_outputs"][name]:
            raise ValueError("campaign review seal mismatch")
    oldpath = OLD/"configurations.json"
    if sha(oldpath) != read(OLD/"analysis-seal.json")["outputs"]["configurations.json"]:
        raise ValueError("old configuration seal mismatch")
    datasets = {"old":[r for r in read(oldpath) if r["primary_scope"]]}
    for label,name,folder in (("primary",names[0],"work"), ("repeat",names[1],"work-recheck")):
        review=read(CAMPAIGN/name)
        if review["cross_stage_context_count"] != 1:
            raise ValueError("mixed contexts in primary/repeat review")
        rows=[]
        for stage in review["stages"]:
            if stage["stage"] == "warmup":
                continue
            if not stage["complete"] or not stage["finalized_seal_present"]:
                raise ValueError("incomplete sealed stage")
            raw=CAMPAIGN/"recovered"/folder/(stage["stage"]+"-01")/"measurements.jsonl"
            rows.extend(import_stage(stage["configurations"],review["contexts"][stage["stage"]],
                                     raw,stage["raw_sha256"],label,stage["stage"]))
        datasets[label]=rows
    extension=read(CAMPAIGN/names[2])
    if not extension["complete"]:
        raise ValueError("incomplete extension")
    datasets["extension"]=import_stage(extension["configurations"],extension["context"],
        CAMPAIGN/"recovered/baseline-extension-01/measurements.jsonl",extension["raw_sha256"],
        "extension","baseline_extension")
    contexts={label:sorted({r["context_id"] for r in rows}) for label,rows in datasets.items()}
    if any(len(c)!=1 for c in contexts.values()) or contexts["old"] == contexts["primary"]:
        raise ValueError("old/new execution identities must remain separate")
    if contexts["primary"] != contexts["repeat"] or contexts["primary"] != contexts["extension"]:
        raise ValueError("repeat/extension execution identity drift")
    return datasets, {"inputs_sha256":{str(p.relative_to(REPO)):sha(p) for p in
        [oldpath,OLD/"analysis-seal.json",CAMPAIGN/"analysis-seal.json",*[CAMPAIGN/n for n in names]]},
        "contexts":contexts, "configurations":{k:len(v) for k,v in datasets.items()},
        "raw_hashes_and_process_labels_verified":True, "repeat_or_extension_used_for_fit":False}


def coverage(datasets):
    output=[]
    families=(*HOSTS, *(f"splitk{q}{suffix}" for q in (2,4,8) for suffix in ("","_groupedn2")))
    for collection,rows in datasets.items():
        for shape in sorted({tuple(r["shape"]) for r in rows}):
            by_family={r["family"]:r for r in rows if tuple(r["shape"])==shape}
            output.append({"collection":collection,"shape":shape,"candidates":{
                f:({"status":"flagged" if by_family[f]["quality_flags"] else "measured",
                    "observed_us":by_family[f]["observed_us"],"algorithm_ids":by_family[f]["algorithm_ids"],
                    "quality_flags":by_family[f]["quality_flags"]} if f in by_family else {"status":"unmeasured"})
                for f in families}})
    return output
