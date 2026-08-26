from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.topology import _parse_link_matrix, _parse_p2p_matrix  # noqa: E402


P2P = """
        GPU0 GPU1 GPU2 GPU3
GPU0    X    OK   OK   OK
GPU1    OK   X    OK   OK
GPU2    OK   OK   X    OK
GPU3    OK   OK   OK   X
"""

TOPOLOGY = """
        GPU0 GPU1 GPU2 GPU3 CPU Affinity
GPU0    X    NV12 NV12 NV12 0-63
GPU1    NV12 X    NV12 NV12 0-63
GPU2    NV12 NV12 X    NV12 0-63
GPU3    NV12 NV12 NV12 X    64-127
"""


class TopologyParserTest(unittest.TestCase):
    def test_parse_p2p_matrix_accepts_nvlink_capability_output(self) -> None:
        matrix = _parse_p2p_matrix(P2P, 4)
        self.assertTrue(all(all(row) for row in matrix))

    def test_parse_link_matrix_preserves_transport_labels(self) -> None:
        matrix = _parse_link_matrix(TOPOLOGY, 4)
        self.assertEqual(matrix[0], ("X", "NV12", "NV12", "NV12"))
        self.assertEqual(matrix[3][0], "NV12")

    def test_parse_matrix_rejects_missing_gpu_row(self) -> None:
        with self.assertRaises(ValueError):
            _parse_p2p_matrix("GPU0 X OK\n", 2)


if __name__ == "__main__":
    unittest.main()
