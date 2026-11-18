#!/usr/bin/env python3
"""Read-only, standard-library verification of the Git-published P1/P2 data.

Checks published hashes and recomputes statistics from recorded samples.
Does not import the predictor, fit a model, contact a server, or run a GPU.
This does not replace analyze.py's full recovery/build-context audit.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics as st

HERE = Path(__file__).resolve().parent
HOSTS = {"host", "host_merged_cutlass", "host_merged_cublaslt"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError((a, b))


def choose(candidates, times):
    return min(candidates, key=lambda c: (times[c], c not in HOSTS, c))


def verify_files():
    manifest = read(HERE / "RELEASE_MANIFEST.json")["files"]
    for name, entry in manifest.items():
        file = (HERE / name).resolve()
        assert file.is_relative_to(HERE), name
        assert file.stat().st_size == entry["bytes"], name
        assert sha(file) == entry["sha256"], name
    recovery = read(HERE / "recovered/recovery-manifest.json")["files"]
    published = 0
    for name, entry in recovery.items():
        if "recovered/" + name in manifest:
            assert manifest["recovered/" + name] == entry, name
            published += 1
    seal = read(HERE / "analysis-preregistration.json")
    for name, expected in seal["files"].items():
        assert sha(HERE / name) == expected, name
    pred = read(HERE / "preregistered-predictions.json")
    assert sha(HERE / "frozen-primary-model.json") == pred["primary_model_sha256"]
    return len(manifest), len(recovery), published


def load_records(folder, jobs, predictions, fixed_order, failed):
    records, counts = [], {"checks": 0, "timing_rows": 0, "samples": 0, "failures": 0}
    for i, job in enumerate(jobs):
        row = read(folder / f"collect-{i:03d}.json")
        assert row["job"] == job
        assert row["context"] == sha(folder / "context.json")
        assert sha(folder / row["stdout"]) == row["stdout_sha256"]
        assert (folder / row["stderr"]).is_file()
        tag, phase = job["shape"], job["phase"]
        if phase == "select":
            ranked = [c for c in predictions[tag]["frozen_ranking"] if (tag, c) not in failed]
            expected = sorted(set(ranked[:3]) | {c for c in fixed_order if (tag, c) not in failed})
        else:
            expected = [c for c in job["order"] if phase == "check" or (tag, c) not in failed]
        assert row["order"] == expected
        assert [r["family"] for r in row["rows"]] == expected
        times = {}
        for r in row["rows"]:
            assert r["input_seed"] == job["seed"]
            if not r["correctness_passed"]:
                failed.add((tag, r["family"]))
                counts["failures"] += 1
            if phase == "check":
                counts["checks"] += 1
                assert "samples_ms" not in r
            else:
                assert r["correctness_passed"]
                assert r["warmup"] == 20 and r["iterations"] == 100
                samples = r["samples_ms"]
                assert len(samples) == 100 and all(math.isfinite(v) and v > 0 for v in samples)
                close(st.median(samples), r["median_ms"])
                times[r["family"]] = st.median(samples) * 1000
                counts["timing_rows"] += 1
                counts["samples"] += len(samples)
        records.append({"group": i, "job": job, "times_us": times})
    return records, counts


def verify_statistics():
    plan = read(HERE / "bundle/plan.json")
    assert sha(HERE / "preregistered-predictions.json") == plan["predictions_sha256"]
    pred = read(HERE / "preregistered-predictions.json")["predictions"]
    result = read(HERE / "analysis.json")
    failed = set()
    records, counts = load_records(HERE / "recovered/results", plan["jobs"], pred, plan["fixed_order"], failed)
    paired, extra = load_records(HERE / "recovered/results-p1-balanced",
                                read(HERE / "balanced-p1-plan.json")["jobs"], pred, [], set())
    for key in counts:
        counts[key] += extra[key]
    assert counts == {"checks": 246, "timing_rows": 294, "samples": 29400, "failures": 14}, counts
    assert sorted(map(list, failed)) == result["audit"]["failed_candidates"]
    assert len(records) == 51 and len(paired) == 6
    for key, pair in result["p1"]["pairs"].items():
        a, b = key.split("__")
        ratios = []
        for measured, saved in zip(paired, pair["blocks"]):
            close(measured["times_us"][a], saved["a_us"])
            close(measured["times_us"][b], saved["b_us"])
            ratios.append(measured["times_us"][a] / measured["times_us"][b])
        close(st.median(ratios), pair["median_speedup"])
        assert sum(v > 1 for v in ratios) == pair["b_faster_blocks"] == 6
    aggregates, process_losses = defaultdict(list), defaultdict(list)
    shape_count = candidate_count = 0
    for tag in sorted(pred):
        shape_count += 1
        ver = [r for r in records if r["job"]["shape"] == tag and r["job"]["phase"] == "verify"]
        sel = [r for r in records if r["job"]["shape"] == tag and r["job"]["phase"] == "select"]
        assert len(ver) == 3 and len(sel) == 1
        assert len({r["job"]["seed"] for r in ver}) == 1
        assert sel[0]["group"] not in [r["group"] for r in ver]
        families = set.intersection(*(set(r["times_us"]) for r in ver)) - {c for t, c in failed if t == tag}
        candidate_count += len(families)
        aggregate = {c: st.median(r["times_us"][c] for r in ver) for c in families}
        saved = result["p2"][tag]
        assert set(aggregate) == set(saved["aggregate_us"])
        for c in families:
            close(aggregate[c], saved["aggregate_us"][c])
        oracle = choose(families, aggregate)
        assert oracle == saved["oracle"]
        ranking = [c for c in pred[tag]["frozen_ranking"] if c in families]
        assert oracle in ranking
        fixed = [c for c in plan["fixed_order"] if c in families]
        choices = {}
        for k in (1, 2, 3):
            for label, ordering in [("model", ranking), ("fixed_order", fixed)]:
                short = ordering[:k]
                choices[f"{label}_top{k}"] = (short, choose(short, sel[0]["times_us"]))
        choices["fixed_q4_g2"] = (["splitk4_groupedn2"], "splitk4_groupedn2")
        hosts = [c for c in ["host_merged_cublaslt", "host_merged_cutlass"] if c in families]
        choices["independent_host_choice"] = (hosts, choose(hosts, sel[0]["times_us"]))
        best_host = choose(families & HOSTS, aggregate)
        choices["strong_host_oracle"] = ([best_host], best_host)
        assert set(choices) == set(saved["evaluations"])
        for method, (short, chosen) in choices.items():
            ev = saved["evaluations"][method]
            assert ev["candidate"] == chosen and ev["shortlist"] == short
            loss = 100 * (aggregate[chosen] / aggregate[oracle] - 1)
            close(loss, ev["aggregate_regret_pct"])
            aggregates[method].append(loss)
            losses = [100 * (r["times_us"][chosen] / min(r["times_us"][c] for c in families) - 1) for r in ver]
            for measured, expected in zip(losses, ev["block_regrets_pct"]):
                close(measured, expected)
            process_losses[method].append(losses)
    block_summary = read(HERE / "block-regret-summary.json")
    for method, values in aggregates.items():
        summary = result["summary"]["methods"][method]
        close(st.mean(values), summary["mean_regret_pct"])
        close(max(values), summary["max_regret_pct"])
        close(st.mean(st.mean(v) for v in process_losses[method]), block_summary[method]["mean_of_shape_mean_block_regret_pct"])
        close(max(v for rows in process_losses[method] for v in rows), block_summary[method]["worst_process_block_regret_pct"])
    assert shape_count == 6 and candidate_count == 72
    return dict(counts, p1_blocks=6, p2_shapes=6, admitted_shape_candidates=72,
                verified_policies=len(aggregates), selection_verification_separate=True)


def main():
    published, recovered, matched = verify_files()
    stats = verify_statistics()
    print(json.dumps(dict(status="passed", published_files_verified=published,
                          complete_recovery_manifest_entries=recovered,
                          published_recovery_entries_verified=matched,
                          complete_recovery_verified=False, statistics=stats,
                          fitting=False, gpu_used=False), indent=2))


if __name__ == "__main__":
    main()
