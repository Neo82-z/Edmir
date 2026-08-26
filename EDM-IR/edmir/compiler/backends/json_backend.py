from __future__ import annotations

import json

from ..model import Program
from ..target import Target
from .base import BackendArtifact


class JsonBackend:
    name = "json"

    def emit(self, program: Program, target: Target) -> BackendArtifact:
        target.validate()
        if target.world_size != program.problem.world_size:
            raise ValueError("target world size does not match Program")
        return BackendArtifact(
            backend=self.name,
            files={
                "program.json": json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n",
                "target.json": json.dumps(target.to_dict(), indent=2, sort_keys=True) + "\n",
            },
            metadata={"target": target.name, "arch": target.arch},
        )


__all__ = ["JsonBackend"]
