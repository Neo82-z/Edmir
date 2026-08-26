from __future__ import annotations

import math
import json
import re
from dataclasses import replace
from pathlib import Path
from statistics import median

from .model import CalibrationProfile, CommSample, GemmSample, PairwiseCommCurve


PAIR_LOG_PATTERN = re.compile(r"pair_(\d+)_(\d+)_default_sendrecv\.txt$")


def parse_nccl_tests_log(path: str | Path) -> tuple[CommSample, ...]:
    """Read out-of-place message size/time points from one nccl-tests log."""

    source = Path(path)
    time_column: int | None = None
    observations: dict[int, list[float]] = {}
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            fields = line.lstrip("# ").split()
            if len(fields) >= 2 and fields[:2] == ["size", "count"] and "time" in fields:
                time_column = fields.index("time")
            continue
        if not line or not line[0].isdigit() or time_column is None:
            continue
        fields = line.split()
        if len(fields) <= time_column:
            continue
        try:
            size_bytes = int(fields[0])
            latency_us = float(fields[time_column])
        except ValueError:
            continue
        if size_bytes > 0 and latency_us >= 0 and math.isfinite(latency_us):
            observations.setdefault(size_bytes, []).append(latency_us)

    if not observations:
        raise ValueError(f"no nccl-tests size/time rows found in {source}")
    return tuple(
        CommSample(size_bytes=size, latency_us=median(observations[size]))
        for size in sorted(observations)
    )


def fit_pairwise_nccl_profile(
    base_profile: CalibrationProfile,
    log_directory: str | Path,
) -> CalibrationProfile:
    """Attach symmetric pairwise curves from default-mode sendrecv logs.

    The nccl-tests runner emits one unordered pair log. Until directional
    measurements are collected, the same measured curve is used in both
    directions. GEMM and fallback analytic parameters remain inherited from
    the base profile.
    """

    directory = Path(log_directory)
    curves: list[PairwiseCommCurve] = []
    matched_logs: list[Path] = []
    for path in sorted(directory.glob("pair_*_*_default_sendrecv.txt")):
        match = PAIR_LOG_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        src_rank, dst_rank = (int(value) for value in match.groups())
        if src_rank == dst_rank:
            raise ValueError(f"pair log must name distinct ranks: {path.name}")
        samples = parse_nccl_tests_log(path)
        curves.extend(
            (
                PairwiseCommCurve(src_rank, dst_rank, samples),
                PairwiseCommCurve(dst_rank, src_rank, samples),
            )
        )
        matched_logs.append(path)

    if not matched_logs:
        raise ValueError(
            f"no default sendrecv logs matching pair_<src>_<dst>_default_sendrecv.txt in {directory}"
        )
    profile = replace(
        base_profile,
        pairwise_comm_curves=tuple(sorted(curves, key=lambda curve: (curve.src_rank, curve.dst_rank))),
        source=(
            f"pairwise nccl-tests from {directory}; "
            f"non-communication parameters inherited from {base_profile.source}"
        ),
    )
    profile.validate()
    return profile


def parse_gemm_benchmark_log(
    path: str | Path,
    *,
    dtype: str = "float16",
) -> tuple[GemmSample, ...]:
    """Parse JSON-lines GEMM measurements and aggregate devices by median.

    Lines that are not JSON objects with ``m``, ``n``, ``k`` and
    ``median_us`` are ignored, which also accepts logs with human-readable
    preambles and a trailing ``SUMMARY_JSON=...`` line.
    """

    observations: dict[tuple[str, int, int, int], list[float]] = {}
    source = Path(path)
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
            sample_dtype = str(value.get("dtype", dtype))
            shape = (sample_dtype, int(value["m"]), int(value["n"]), int(value["k"]))
            latency_us = float(value["median_us"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if min(shape[1:]) > 0 and latency_us >= 0 and math.isfinite(latency_us):
            observations.setdefault(shape, []).append(latency_us)
    if not observations:
        raise ValueError(f"no JSON-lines GEMM measurements found in {source}")
    return tuple(
        GemmSample(
            dtype=shape[0],
            m=shape[1],
            n=shape[2],
            k=shape[3],
            latency_us=median(latencies),
        )
        for shape, latencies in sorted(observations.items())
    )


def fit_gemm_profile(
    base_profile: CalibrationProfile,
    benchmark_log: str | Path,
    *,
    dtype: str = "float16",
    gemm_provider: str = "cublas",
    gemm_kernel: str = "library_default",
) -> CalibrationProfile:
    """Replace GEMM samples of one dtype with measured shape medians."""

    measured = parse_gemm_benchmark_log(benchmark_log, dtype=dtype)
    keep = tuple(sample for sample in base_profile.gemm_samples if sample.dtype != dtype)
    profile = replace(
        base_profile,
        gemm_samples=tuple(sorted((*keep, *measured), key=lambda s: (s.dtype, s.n, s.k, s.m))),
        gemm_provider=gemm_provider,
        gemm_kernel=gemm_kernel,
        source=f"{base_profile.source}; GEMM shapes from {benchmark_log}",
    )
    profile.validate()
    return profile


def parse_local_copy_benchmark_log(path: str | Path) -> tuple[CommSample, ...]:
    """Parse JSON-lines D2D copy measurements and aggregate devices."""

    observations: dict[int, list[float]] = {}
    source = Path(path)
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
            size_bytes = int(value["size_bytes"])
            latency_us = float(value["median_us"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if size_bytes > 0 and latency_us >= 0 and math.isfinite(latency_us):
            observations.setdefault(size_bytes, []).append(latency_us)
    if not observations:
        raise ValueError(f"no JSON-lines local-copy measurements found in {source}")
    return tuple(
        CommSample(size_bytes=size, latency_us=median(observations[size]))
        for size in sorted(observations)
    )


def fit_local_copy_profile(
    base_profile: CalibrationProfile,
    benchmark_log: str | Path,
) -> CalibrationProfile:
    samples = parse_local_copy_benchmark_log(benchmark_log)
    profile = replace(
        base_profile,
        local_copy_curve=samples,
        source=f"{base_profile.source}; local D2D copies from {benchmark_log}",
    )
    profile.validate()
    return profile


__all__ = [
    "fit_gemm_profile",
    "fit_local_copy_profile",
    "fit_pairwise_nccl_profile",
    "parse_gemm_benchmark_log",
    "parse_local_copy_benchmark_log",
    "parse_nccl_tests_log",
]
