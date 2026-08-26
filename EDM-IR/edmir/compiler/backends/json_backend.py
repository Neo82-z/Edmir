from __future__ import annotations

import json

from ..model import Program
from ..target import Target
from ..verify import verify_program
from .base import BackendArtifact


class JsonBackend:
    name = "json"

    def emit(self, program: Program, target: Target) -> BackendArtifact:
        verify_program(program.problem, program).require_valid()
        target.validate_program(program)
        resolved_target = target.resolve(program.problem, gemm_m=program.chunk_rows)
        if resolved_target.world_size != program.problem.world_size:
            raise ValueError("target world size does not match Program")
        metadata: dict[str, object] = {
            "target": resolved_target.name,
            "arch": resolved_target.arch,
            "gemm_provider": resolved_target.gemm_provider,
            "gemm_kernel": resolved_target.gemm_kernel,
            "communication_api": resolved_target.communication_api,
        }
        if any(op.kind == "fused_transfer_gemm" for op in program.ops):
            metadata.update(resolved_target.estimate_fused_traffic(program).to_dict())
        return BackendArtifact(
            backend=self.name,
            files={
                "program.json": json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n",
                "target.json": json.dumps(
                    resolved_target.to_dict(), indent=2, sort_keys=True
                )
                + "\n",
            },
            metadata=metadata,
        )


__all__ = ["JsonBackend"]
