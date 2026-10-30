#!/usr/bin/env python3
"""CPU-only audit of pinned benchmark accounting; no GPU or timing measurements.

Extract selected pure functions by AST, without importing the inference framework.
Synthetic timestamps demonstrate accounting semantics, not device performance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace


DEEPINFER_COMMIT = "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01"
RECIPE_COMMIT = "935817c66f5d6e64838f419692545f06787e485f"
DEEPINFER_FILES = (
    "README.md",
    "examples/README.md",
    "examples/benchmark.py",
    "examples/_config.py",
    "examples/sweep/plot_pareto.py",
    "examples/sweep/run_sweep.sh",
    "deepinfer/timer.py",
    "deepinfer/worker/worker.py",
    "deepinfer/model_executor/eplb/__init__.py",
    "deepinfer/model_executor/models/decoder_layer.py",
    "deepinfer/model_executor/models/v4_modules/mlp.py",
    "csrc/engine/input_metadata/prepare_tensor_input_base.cpp",
    "deepinfer/model_executor/layers/tilelang_ops/prepare_v4_decode_metadata.py",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pinned_sources(root: Path, commit: str, paths: tuple[str, ...]) -> dict:
    require(git(root, "rev-parse", "HEAD").decode().strip() == commit,
            f"{root}: expected pinned HEAD {commit}")
    hashes = {}
    for relative in paths:
        working = (root / relative).read_bytes()
        committed = git(root, "show", f"{commit}:{relative}")
        require(working == committed, f"Modified audited source: {relative}")
        hashes[relative] = sha(working)
    return {"commit": commit, "files_sha256": hashes}


def extract(root: Path, relative: str, functions: tuple[str, ...], *,
            constants: tuple[str, ...] = (), scope: dict | None = None) -> dict:
    """Execute only the requested definitions and literal constants at the pinned revision."""
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    for name in functions:
        matches = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(matches) == 1, f"Expected one definition for {name}")
        require(not matches[0].decorator_list, f"Unexpected decorator on {name}")
        selected.append(matches[0])
    namespace = dict(scope or {})
    for name in constants:
        matches = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
        require(len(matches) == 1, f"Expected one assignment for {name}")
        namespace[name] = ast.literal_eval(matches[0].value)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def accounting_example(root: Path) -> dict:
    # 60 intervals: 1 ms work each, with an additional 5 ms every tenth interval.
    timestamp = 0.0
    decode = [(timestamp, False)]
    prefill = [(0, timestamp, False)]
    durations = []
    for step in range(1, 61):
        flushed = step % 10 == 0
        duration = 0.001 + (0.005 if flushed else 0.0)
        durations.append(duration)
        timestamp += duration
        decode.append((timestamp, flushed))
        prefill.append((4096, timestamp, flushed))
    namespace = extract(
        root, "examples/benchmark.py", ("measure_prefill", "measure_decode"),
        scope={"pairwise": itertools.pairwise, "stable_decode": decode,
               "stable_prefill": prefill, "cp_size": 1,
               "args": SimpleNamespace(decode_num_requests=64),
               "envs": SimpleNamespace(DEEPINFER_USE_NEXTN_DECODE=5)},
    )
    clean_decode = namespace["measure_decode"]()
    clean_prefill = namespace["measure_prefill"]()
    raw_sps = 60 / sum(durations)
    require(clean_decode["decode_window_steps"] == 54, "Unexpected flush exclusion")
    require(math.isclose(clean_decode["decode_avg_sps"], 1000), "Unexpected clean SPS")
    require(math.isclose(raw_sps, 2000 / 3), "Unexpected inclusive SPS")
    require(clean_prefill["prefill_window_steps"] == 54, "Prefill accounting differs")
    require(clean_prefill["prefill_window_tokens"] == 54 * 4096,
            "Removed interval must remove its tokens as well as its duration")
    all_clean = list(namespace["stable_decode"])
    namespace["stable_decode"] = [(ts, False) for ts, _ in all_clean]
    require(math.isclose(namespace["measure_decode"]()["decode_avg_sps"], raw_sps),
            "No-filter control should equal inclusive SPS")
    namespace["stable_decode"] = [(ts, True) for ts, _ in all_clean]
    require(namespace["measure_decode"]() == {}, "All-filtered control should be empty")
    return {
        "evidence_kind": "synthetic_timestamps_executed_through_upstream_functions",
        "hardware_measurement": False,
        "input": {"intervals": 60, "base_ms": 1, "extra_ms_every_tenth": 5},
        "inclusive_steps_per_s": raw_sps,
        "inclusive_max_step_ms": max(durations) * 1000,
        "inclusive_nominal_tokens_per_s_per_gpu": raw_sps * 64 * 6,
        "upstream_clean_decode": clean_decode,
        "upstream_clean_prefill": clean_prefill,
        "clean_to_inclusive_throughput_ratio": clean_decode["decode_avg_sps"] / raw_sps,
        "excluded_interval_fraction": 6 / 60,
        "excluded_wall_time_fraction": sum(durations[9::10]) / sum(durations),
        "controls": ["no flags equals inclusive rate", "all flags returns empty metrics"],
    }


def plot_example(root: Path) -> dict:
    namespace = extract(
        root, "examples/sweep/plot_pareto.py", ("load_points", "decode_tau"),
        constants=("SWEEP_KEY", "FIXED_FIELDS"), scope={"Path": Path, "json": json},
    )

    def record(batch: int, sps: float, **changes) -> dict:
        return dict(
            {"model_name": "synthetic", "ep_size": 4, "cp": 1, "context_length": 32768,
             "nextn": 5, "mtp_varlen": False, "num_extra_experts": 32,
             "prompt_mode": "text", "decode_num_requests": batch, "decode_avg_sps": sps,
             "decode_tokens_per_s_per_gpu": sps * batch * 6}, **changes,
        )

    with tempfile.TemporaryDirectory(prefix="deepinfer-accounting-") as directory:
        path = Path(directory) / "synthetic.jsonl"

        def load(rows: list[dict]) -> list[dict]:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            return namespace["load_points"](path, "decode")

        duplicate = load([record(64, 100), record(64, 90)])
        require(len(duplicate) == 1 and duplicate[0]["decode_avg_sps"] == 90,
                "Expected last-run selection")
        mixed = load([record(64, 100), record(128, 90, prompt_mode="image")])
        require(len(mixed) == 2, "Expected unchecked prompt-mode mismatch")
        erased = load([record(64, 100, nextn=3), record(64, 90)])
        require(erased[0]["nextn"] == 5, "Earlier configuration was not overwritten")
        try:
            load([record(64, 100), record(128, 90, ep_size=8)])
        except AssertionError:
            guarded_ep_mismatch = True
        else:
            raise ValueError("EP mismatch should be rejected")
        tau = namespace["decode_tau"]([record(64, 100)], 0.75)
        require(tau == 4.75, "Unexpected token accounting")
    return {
        "evidence_kind": "synthetic_records_executed_through_upstream_functions",
        "hardware_measurement": False,
        "two_repetitions_retained": len(duplicate),
        "retained_sps": duplicate[0]["decode_avg_sps"],
        "mixed_prompt_modes_accepted": [r["prompt_mode"] for r in mixed],
        "earlier_nextn_mismatch_erased_before_validation": True,
        "different_ep_across_distinct_batches_rejected": guarded_ep_mismatch,
        "nominal_tau_at_nextn5_acceptance075": tau,
        "full_acceptance_to_nominal075_ratio": 6 / tau,
        "note": "Acceptance is assumed, not measured; last-run selection can raise or lower a curve.",
    }


def local_baseline(root: Path) -> dict:
    pilot = root / "experiments/2026-10-20-splitk-local-revision/gpu_pilot"
    envelope = json.loads((pilot / "package.json").read_text())
    package = envelope["package"]
    checked = 0
    for relative, expected in package["sources"].items():
        require(sha((root / relative).read_bytes()) == expected,
                f"Frozen pilot source changed: {relative}")
        checked += 1
    counts = {}
    for family in ("host", "direct"):
        directory = pilot / "generated" / f"selector_w000_{family}"
        program_path = directory / "program.json"
        program = json.loads(program_path.read_text())
        accesses = sum(len(op["reads"]) + len(op["writes"]) for op in program["ops"])
        declared = re.search(r"std::array<AccessDesc,\s*(\d+)>", (directory / "edmir_program.hpp").read_text())
        require(declared is not None and accesses == int(declared.group(1)),
                "JSON access count differs from generated C++")
        calls = 2 * sum(op["kind"] != "barrier" for op in program["ops"])
        counts[family] = {
            "program_sha256": sha(program_path.read_bytes()), "ops": len(program["ops"]),
            "access_records": accesses, "accesses_calls_per_execute": calls,
            "records_visited_per_execute": accesses * calls,
        }
    require(counts["host"]["records_visited_per_execute"] == 7424, "Host program changed")
    require(counts["direct"]["records_visited_per_execute"] == 1536, "Fused program changed")
    return {
        "evidence_kind": "static_counts_from_current_runtime_and_generated_programs",
        "hardware_measurement": False,
        "sources_matching_frozen_pilot": checked,
        "pilot_package_sha256": envelope["package_sha256"],
        "per_execute": counts,
        "note": "Counts exclude prepare(); they are not latency estimates or a measured bottleneck.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepinfer", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("audit_results.json"))
    args = parser.parse_args()
    require(__debug__, "Run without Python -O; upstream consistency checks use assert")
    sources = {
        "DeepInfer": pinned_sources(args.deepinfer, DEEPINFER_COMMIT, DEEPINFER_FILES),
        "deepseek-recipe": pinned_sources(args.recipe, RECIPE_COMMIT, ("README.md", "server-py/README.md")),
    }
    result = {
        "schema": "deepinfer_measurement_audit.v1", "gpu_experiments_run": False,
        "sources": sources,
        "accounting_example": accounting_example(args.deepinfer),
        "plot_example": plot_example(args.deepinfer),
        "our_runtime": local_baseline(args.repo),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: pinned-source accounting, plot controls, and local static counts; {args.out}")


if __name__ == "__main__":
    main()
