"""Shared helpers for predict.py and analyze.py; the same tie band and interval rule apply to both."""

import csv
import hashlib
from pathlib import Path

CLEAR = ("candidate_wins", "baseline_wins")


def classify(ratio, upper):
    """ratio = T_baseline / T_candidate."""
    if ratio > upper:
        return "candidate_wins"
    if ratio < 1.0 / upper:
        return "baseline_wins"
    return "tie"


def intervals(seq):
    """seq: (m, class) ascending in m.

    strict: adjacent M bins with opposite clear wins.
    loose:  opposite clear wins separated only by tie/undetermined bins.
    """
    found = []
    for (m0, c0), (m1, c1) in zip(seq, seq[1:]):
        if c0 in CLEAR and c1 in CLEAR and c0 != c1:
            found.append(("strict", m0, m1, f"{c0}->{c1}"))
    last = None
    for i, (m, c) in enumerate(seq):
        if c in CLEAR:
            if last is not None and seq[last][1] != c and i - last > 1:
                if all(g in ("tie", "undetermined") for _, g in seq[last + 1:i]):
                    found.append(("loose", seq[last][0], m, f"{seq[last][1]}->{c}"))
            last = i
    return found


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
