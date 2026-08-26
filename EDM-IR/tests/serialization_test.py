from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir import EDMBuilder, MovementDesc  # noqa: E402
from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    Program,
    build_fused_program,
    build_pipelined_program,
)
from edmir.ops import COMM_ALL_GATHER, STREAM_ORDER  # noqa: E402
from edmir.serialization import graph_from_dict, graph_to_dict  # noqa: E402


class SerializationTest(unittest.TestCase):
    def test_trace_json_round_trip(self) -> None:
        builder = EDMBuilder(default_phase="decode")
        begin = builder.phase_begin("decode", 0.0)
        comm = builder.comm(
            "all_gather",
            0.0,
            4.0,
            op=COMM_ALL_GATHER,
            movement=MovementDesc(kind="collective", group_ranks=(0, 1)),
        )
        end = builder.phase_end("decode", 4.0)
        builder.sequence(begin, comm, end, kind=STREAM_ORDER)
        restored = graph_from_dict(graph_to_dict(builder.build()))
        self.assertEqual(restored.dump_uops(), builder.build().dump_uops())

    def test_program_json_round_trip(self) -> None:
        problem = A2AGemmProblem(world_size=2, rows_per_peer=32, k=64, n=128)
        program = build_pipelined_program(problem, chunk_rows=16, buffer_count=2)
        restored = Program.from_dict(program.to_dict())
        self.assertEqual(restored, program)

    def test_fused_program_json_round_trip(self) -> None:
        problem = A2AGemmProblem(world_size=2, rows_per_peer=32, k=64, n=128)
        program = build_fused_program(problem, chunk_rows=16)
        self.assertEqual(Program.from_dict(program.to_dict()), program)


if __name__ == "__main__":
    unittest.main()
