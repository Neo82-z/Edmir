from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..model import Program
from ..target import Target


@dataclass(frozen=True, slots=True)
class BackendArtifact:
    backend: str
    files: dict[str, str]
    metadata: dict[str, object] = field(default_factory=dict)

    def write_to(self, directory: str | Path) -> tuple[Path, ...]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for relative_name, content in self.files.items():
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe backend artifact path {relative_name!r}")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written.append(path)
        return tuple(written)


class Backend(Protocol):
    name: str

    def emit(self, program: Program, target: Target) -> BackendArtifact: ...


__all__ = ["Backend", "BackendArtifact"]
