from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    AnalyticalScheduleFeatures,
    ScheduleMeasurement,
    ScheduleSpec,
    account_schedule,
)


class RooflineAccountingTest(unittest.TestCase):
    def test_fused_account_uses_per_rank_batch_and_remote_replay(self) -> None:
        account = account_schedule(
            ScheduleSpec(
                family="splitk8_groupedn2",
                world_size=4,
                rows_per_peer=64,
                k=8192,
                n=1024,
                measured_ms=0.7685,
                group_id="boundary",
            )
        )
        self.assertEqual(account.batch_rows_per_rank, 256)
        self.assertEqual(account.batch_rows_global, 1024)
        self.assertEqual(account.flops_per_rank, 4_294_967_296)
        self.assertEqual(account.rho_n, 4)
        self.assertEqual(account.gamma, 32)
        self.assertEqual(account.unique_remote_bytes_per_rank, 3_145_728)
        self.assertEqual(account.replayed_remote_bytes_per_rank, 12_582_912)
        self.assertEqual(account.local_activation_bytes_per_rank, 1_048_576)
        self.assertEqual(account.local_weight_bytes_per_rank, 16_777_216)
        self.assertEqual(account.local_output_bytes_per_rank, 524_288)
        self.assertEqual(account.reduction_bytes_per_rank, 7_340_032)
        self.assertEqual(account.lower_bound_bottleneck, "remote")
        self.assertGreater(account.unexplained_ms, 0.7)

    def test_host_staging_is_sequential_in_critical_path_floor(self) -> None:
        account = account_schedule(
            ScheduleSpec(
                family="host",
                world_size=4,
                rows_per_peer=64,
                k=8192,
                n=1024,
                measured_ms=0.8,
            )
        )
        self.assertEqual(account.replayed_remote_bytes_per_rank, 3_145_728)
        self.assertEqual(account.local_staging_bytes_per_rank, 3_145_728)
        self.assertGreater(account.critical_path_floor_ms, account.roofline_floor_ms)

    def test_selector_and_ledger_share_local_byte_semantics(self) -> None:
        row = ScheduleMeasurement(
            group_id="g",
            family="splitk8_groupedn2",
            world_size=4,
            rows_per_peer=64,
            k=8192,
            n=1024,
            measured_ms=0.7685,
            remote_input_bytes=12_582_912,
            remote_a_replay=4,
            split_k_slices=8,
            n_tiles_per_cta=2,
            estimated_ctas_per_op=32,
        )
        selector_features = AnalyticalScheduleFeatures.from_measurement(row)
        ledger = account_schedule(ScheduleSpec.from_measurement(row))
        self.assertEqual(selector_features.local_bytes, ledger.local_bytes_per_rank)
        self.assertEqual(selector_features.flops, ledger.flops_per_rank)
        self.assertEqual(
            selector_features.unique_remote_bytes,
            ledger.unique_remote_bytes_per_rank,
        )


if __name__ == "__main__":
    unittest.main()
