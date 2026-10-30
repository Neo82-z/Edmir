from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'EDM-IR'),str(ROOT/'EDM-IR/tools')]
from edmir.compiler.resource_geometry import ResourceGeometry, axis_interaction
from edmir.compiler.measurement_dataset import P4_ENV
from predict_traffic_schedule import static_candidates


class ResourceGeometryTest(unittest.TestCase):
    def test_split_partitions_cover_operands_once_before_grid_reuse(self):
        # Independently enumerate K intervals and valid tensor elements.
        for m,k,n in ((32,512,128),(64,3072,1536),(192,6144,512),(512,8192,512)):
            for r in static_candidates(m,k,n,P4_ENV).values():
                if r.is_host:continue
                g=ResourceGeometry.from_candidate(r);q=r.split_k_slices
                intervals=[(i*k//q,(i+1)*k//q) for i in range(q)]
                self.assertEqual(sum(hi-lo for lo,hi in intervals),k)
                self.assertEqual(g.logical_a_load_bytes,sum(2*m*(hi-lo)*g.n_tiles for lo,hi in intervals))
                self.assertEqual(g.logical_b_load_bytes,sum(2*n*(hi-lo)*g.m_tiles for lo,hi in intervals))
                self.assertEqual(g.extra_output_read_write_bytes,4*m*n*(q-1))

    def test_same_grid_and_total_loads_do_not_imply_same_working_sets(self):
        a=ResourceGeometry.from_candidate(static_candidates(64,8192,2048,P4_ENV)['splitk4'])
        b=ResourceGeometry.from_candidate(static_candidates(256,8192,512,P4_ENV)['splitk4'])
        self.assertEqual(a.output_tiles,b.output_tiles)
        self.assertEqual(a.logical_a_load_bytes,b.logical_a_load_bytes)
        self.assertEqual(a.logical_b_load_bytes,b.logical_b_load_bytes)
        self.assertEqual(a.a_working_set_bytes*4,b.a_working_set_bytes)
        self.assertEqual(a.b_working_set_bytes,b.b_working_set_bytes*4)

    def test_grouping_changes_A_replay_but_not_unique_B(self):
        rows=static_candidates(64,4096,512,P4_ENV)
        a=ResourceGeometry.from_candidate(rows['splitk4'])
        b=ResourceGeometry.from_candidate(rows['splitk4_groupedn2'])
        self.assertEqual(a.logical_a_load_bytes,2*b.logical_a_load_bytes)
        self.assertEqual(a.logical_b_load_bytes,b.logical_b_load_bytes)
        self.assertEqual(a.b_working_set_bytes,b.b_working_set_bytes)

    def test_q_one_has_no_extra_epilogue_or_semaphore(self):
        g=ResourceGeometry.from_candidate(static_candidates(64,4096,512,P4_ENV)['direct'])
        self.assertEqual(g.extra_output_read_write_bytes,0)
        self.assertEqual(g.serial_epilogue_boundaries_per_tile,0)
        self.assertEqual(g.semaphore_slots,0)

    def test_axes_and_crossed_observation(self):
        for m,n in ((64,1536),(192,512),(512,512)):
            self.assertEqual(axis_interaction(static_candidates(m,3072,n,P4_ENV)['host']),0)
        self.assertGreater(axis_interaction(static_candidates(192,3072,1536,P4_ENV)['host']),0)

    def test_host_and_unaligned_partitions_rejected(self):
        with self.assertRaises(ValueError):ResourceGeometry.from_candidate(static_candidates(64,4096,512,P4_ENV)['host'])
        r=static_candidates(64,4096,512,P4_ENV)['splitk16']
        with self.assertRaises(ValueError):ResourceGeometry.from_candidate(replace(r,k=4097))
        with self.assertRaises(TypeError):ResourceGeometry.from_candidate({})


if __name__=='__main__':unittest.main()
