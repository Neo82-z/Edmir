from __future__ import annotations

import json

from ..model import Program
from ..target import Target
from .base import BackendArtifact


class JsonBackend:
    name = "json"

    def emit(self, program: Program, target: Target) -> BackendArtifact:
        target.validate_problem(program.problem)
        if target.world_size != program.problem.world_size:
            raise ValueError("target world size does not match Program")
        return BackendArtifact(
            backend=self.name,
            files={
                "program.json": json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n",
                "target.json": json.dumps(target.to_dict(), indent=2, sort_keys=True) + "\n",
            },
            metadata={
                "target": target.name,
                "arch": target.arch,
                "gemm_provider": target.gemm_provider,
            },
        )


__all__ = ["JsonBackend"]
