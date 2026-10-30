from __future__ import annotations

import unittest

from edmir.compiler.routed_contract import (
    RouteRecord,
    RoutedTrace,
    build_routed_traffic_ledger,
    validate_fragmentation_sentinel,
)


PLACEMENT = (0, 0, 1, 1)


def make_trace(trace_id: str, assignments: tuple[tuple[int, int], ...]) -> RoutedTrace:
    routes = tuple(
        RouteRecord(
            global_token_id=index,
            source_rank=source,
            destination_rank=PLACEMENT[expert],
            expert_id=expert,
            route_slot=0,
        )
        for index, (source, expert) in enumerate(assignments)
    )
    routes = tuple(sorted(routes, key=lambda route: route.input_key()))
    return RoutedTrace(
        trace_id=trace_id,
        world_size=2,
        k=128,
        n=256,
        dtype="bfloat16",
        phase="synthetic",
        expert_placement=PLACEMENT,
        routes=routes,
    )


LOW_FRAGMENTATION = make_trace(
    "low",
    tuple([(0, 1)] * 4 + [(0, 2)] * 4 + [(1, 0)] * 4 + [(1, 3)] * 4),
)
MEDIUM_FRAGMENTATION = make_trace(
    "center",
    tuple(
        [(0, 0)] * 2
        + [(0, 1)] * 2
        + [(0, 2)] * 4
        + [(1, 0)] * 2
        + [(1, 1)] * 2
        + [(1, 3)] * 4
    ),
)
HIGH_FRAGMENTATION = make_trace(
    "high",
    tuple(
        [(0, 0)] * 2
        + [(0, 1)] * 2
        + [(0, 2)] * 2
        + [(0, 3)] * 2
        + [(1, 0)] * 2
        + [(1, 1)] * 2
        + [(1, 2)] * 2
        + [(1, 3)] * 2
    ),
)


class RoutedContractTest(unittest.TestCase):
    def test_round_trip_and_canonical_output(self) -> None:
        restored = RoutedTrace.from_dict(HIGH_FRAGMENTATION.to_dict())
        self.assertEqual(restored, HIGH_FRAGMENTATION)
        keys = tuple(
            restored.routes[index].output_key()
            for index in restored.canonical_output_indices()
        )
        self.assertEqual(keys, tuple(sorted(keys)))

    def test_traffic_ledger_uses_max_rank_and_bucket_distribution(self) -> None:
        low = build_routed_traffic_ledger(LOW_FRAGMENTATION)
        high = build_routed_traffic_ledger(HIGH_FRAGMENTATION)
        self.assertEqual(low.replay, 2)
        self.assertEqual(low.max_rank_unique_remote_bytes, 4 * 128 * 2)
        self.assertEqual(low.max_rank_replayed_remote_bytes, 8 * 128 * 2)
        self.assertEqual(low.nonempty_remote_buckets, 2)
        self.assertEqual(high.nonempty_remote_buckets, 4)
        self.assertEqual(low.expert_rows, high.expert_rows)
        self.assertEqual(low.cta_supply_by_rank, high.cta_supply_by_rank)

    def test_fragmentation_sentinel_holds_confounders_fixed(self) -> None:
        ledgers = validate_fragmentation_sentinel(
            (LOW_FRAGMENTATION, MEDIUM_FRAGMENTATION, HIGH_FRAGMENTATION)
        )
        self.assertEqual(
            {ledger.nonempty_remote_buckets for ledger in ledgers}, {2, 3, 4}
        )

    def test_fragmentation_sentinel_rejects_changed_expert_work(self) -> None:
        changed = make_trace(
            "changed",
            tuple([(0, 0)] * 4 + [(0, 2)] * 4 + [(1, 0)] * 4 + [(1, 3)] * 4),
        )
        with self.assertRaisesRegex(ValueError, "changed bytes"):
            validate_fragmentation_sentinel(
                (LOW_FRAGMENTATION, MEDIUM_FRAGMENTATION, changed)
            )


if __name__ == "__main__":
    unittest.main()
