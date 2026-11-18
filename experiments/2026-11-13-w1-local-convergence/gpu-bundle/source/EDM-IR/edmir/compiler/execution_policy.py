"""Versioned, opt-in dependency policies; no performance admission implied."""
from dataclasses import replace

LEGACY_POLICY = "resource_queues_v1"
UNIFORM_PEER_POLICY = "uniform_immutable_peer_v1"
POLICIES = (LEGACY_POLICY, UNIFORM_PEER_POLICY)


def dependency_policy(program):
    policy = program.metadata.get("dependency_policy", LEGACY_POLICY)
    if policy not in POLICIES:
        raise ValueError("unknown dependency policy")
    if policy == UNIFORM_PEER_POLICY:
        # Fail closed: this policy is not a general hazard-elimination pass.
        from .schedule import build_fused_program
        from .remote_publication import REMOTE_PUBLICATION_METADATA_KEY
        p = program.problem
        if p.dtype != "float16" or not 2 <= p.world_size <= 8:
            raise ValueError("uniform peer policy requires FP16 and 2..8 ranks")
        expected = build_fused_program(p, chunk_rows=p.rows_per_peer)
        if (program.strategy != expected.strategy or program.chunk_rows != expected.chunk_rows
                or program.buffer_count != expected.buffer_count
                or program.buffers != expected.buffers or program.ops != expected.ops
                or program.metadata.get(REMOTE_PUBLICATION_METADATA_KEY)
                != expected.metadata[REMOTE_PUBLICATION_METADATA_KEY]):
            raise ValueError("uniform peer policy requires the canonical unchunked immutable workload")
    return policy


def with_dependency_policy(program, policy):
    result = replace(program, metadata={**program.metadata, "dependency_policy": policy})
    dependency_policy(result)
    return result
