#!/usr/bin/env bash
# Recompute the first study's analysis from the archived run directory and compare it with the
# published analysis output. Needs python3 and xz; no GPU.
#
#   ./reproduce.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

tar -xJf "$here/run.tar.xz" -C "$work"
# Every extracted file must be the one listed in the manifest.
(cd "$work" && shasum -a 256 -c "$here/MANIFEST.sha256" --quiet)
echo "manifest: $(wc -l < "$here/MANIFEST.sha256" | tr -d ' ') files match"

python3 "$here/../2026-11-23-replay/analyze.py" --run "$work/run" --out "$work/results"
for f in paper_results.json effects.csv phases.csv timing_diagnostics.csv; do
    cmp "$work/results/$f" "$here/results/$f"
    echo "identical: $f"
done
