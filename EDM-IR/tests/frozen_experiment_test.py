from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments/2026-08-29-a100-8gpu-transfer-plan/manifest.json"
RUNNER = ROOT / "EDM-IR/tools/run_frozen_fused_experiment.py"


class FrozenExperimentTest(unittest.TestCase):
    def test_manifest_has_disjoint_calibration_and_heldout_shapes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        by_role: dict[str, set[tuple[int, int, int]]] = {
            "calibration": set(),
            "heldout": set(),
        }
        for sweep in manifest["sweeps"]:
            for k in sweep["k_values"]:
                by_role[sweep["role"]].add(
                    (sweep["rows_per_peer"], sweep["n"], k)
                )
        self.assertFalse(by_role["calibration"] & by_role["heldout"])
        self.assertEqual(len(by_role["calibration"]), 4)
        self.assertEqual(len(by_role["heldout"]), 19)
        self.assertEqual(11 * sum(map(len, by_role.values())), 253)

    def test_runner_is_dry_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            topology = temporary / "topology.json"
            p2p = [[True for _ in range(8)] for _ in range(8)]
            links = [
                ["X" if source == destination else "NV12" for destination in range(8)]
                for source in range(8)
            ]
            topology.write_text(
                json.dumps(
                    {
                        "world_size": 8,
                        "devices": ["NVIDIA A100-SXM4-80GB"] * 8,
                        "p2p": p2p,
                        "link_labels": links,
                    }
                ),
                encoding="utf-8",
            )
            output = temporary / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--manifest",
                    str(MANIFEST),
                    "--runtime",
                    str(temporary / "runtime"),
                    "--topology",
                    str(topology),
                    "--cutlass",
                    str(temporary / "cutlass"),
                    "--nccl-home",
                    str(temporary / "nccl"),
                    "--out",
                    str(output),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            record = json.loads(
                (output / "run_record.json").read_text(encoding="utf-8")
            )
            self.assertFalse(record["execute"])
            self.assertEqual(
                len(record["commands"]),
                len(json.loads(MANIFEST.read_text())["sweeps"]),
            )
            self.assertTrue(
                all("--order-seed" in entry["command"] for entry in record["commands"])
            )


if __name__ == "__main__":
    unittest.main()
