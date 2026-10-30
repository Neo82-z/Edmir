#!/usr/bin/env python3
"""Optional independent NumPy/SVD check of split-K normal-equation arithmetic."""
import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edmir.compiler.measurement_dataset import load_identified_selector_dataset, sha256
from edmir.compiler.splitk_revision import SplitKFit


def manual_features(row, basis):
    q, k = row.split_k_slices, row.k
    grid = math.ceil(row.rows_per_peer/64)*math.ceil(row.n/128)
    if basis == "original":
        return [math.log2(q), math.log2(q)*math.log2(k/4096),
                math.log2(q)*math.log2(grid/64), (q-1)*4096/k]
    return [math.log2(math.ceil(grid*q/108)/(q*math.ceil(grid/108))),
            math.log2(q), (q-1)/16, 32*(q-1)/k]


def audit(dataset):
    records = []
    groups = [sorted([r for r in s.values() if not r.is_host and r.n_tiles_per_cta == 1],
                     key=lambda r: r.split_k_slices) for s in dataset.validate().values()]
    for basis in ("original", "waves"):
        arrays = [np.array([manual_features(r, basis) for r in g]) for g in groups]
        scales = np.maximum(np.sqrt(np.mean([np.mean(a[1:]**2, axis=0) for a in arrays], axis=0)), 1e-9)
        for loss in ("reference", "pairwise"):
            xs, ys = [], []
            for rows, values in zip(groups, arrays):
                pairs = [(i, 0) for i in range(1, len(rows))] if loss == "reference" else list(combinations(range(len(rows)), 2))
                weight = 1/math.sqrt(len(pairs))
                for a, b in pairs:
                    xs.append((values[a]-values[b])/scales*weight)
                    ys.append(math.log(rows[a].measured_ms/rows[b].measured_ms)*weight)
            x, y = np.array(xs), np.array(ys)
            # Independent augmented least squares; do not call the production solver.
            coefficients, _, _, _ = np.linalg.lstsq(np.vstack((x, np.eye(4))), np.r_[y, np.zeros(4)], rcond=None)
            fitted = SplitKFit.fit(dataset, basis=basis, loss=loss)
            error = float(np.max(np.abs(coefficients-np.array(fitted.coefficients[1:]))))
            scale_error = float(np.max(np.abs(scales-np.array(fitted.scales))))
            if error > 1e-10 or scale_error > 1e-10:
                raise AssertionError("production fit differs from independent SVD reconstruction")
            singular = np.linalg.svd(x, compute_uv=False)
            records.append({"basis": basis, "loss": loss, "max_coefficient_error": error,
                            "max_scale_error": scale_error, "design_rank": int(np.linalg.matrix_rank(x)),
                            "singular_values": singular.tolist(), "condition_number": float(singular[0]/singular[-1])})
    return {"schema": "edmir.splitk-numerical-audit.v1", "numpy_version": np.__version__,
            "status": "arithmetic agreement; not prediction validation", "records": records,
            "audit_source_sha256": sha256(Path(__file__)),
            "model_source_sha256": sha256(Path(__file__).parents[1]/"edmir/compiler/splitk_revision.py")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error("output exists")
    data, _, _, _ = load_identified_selector_dataset(args.dataset)
    args.out.write_text(json.dumps(audit(data), indent=2, allow_nan=False)+"\n")
    print(args.out)
