from __future__ import annotations

from dataclasses import dataclass

from .model import Program
from .verify import verify_program


@dataclass(frozen=True, slots=True)
class SlotEpoch:
    buffer: str
    epoch: int
    producer: str
    consumer: str
    actions: tuple[str, ...] = (
        "BeginProduce",
        "Publish",
        "StartConsume",
        "FinishSplit(1)",
        "Release",
    )


@dataclass(frozen=True, slots=True)
class SlotCorrespondenceReport:
    applicable: bool
    valid: bool
    epochs: tuple[SlotEpoch, ...]
    errors: tuple[str, ...] = ()

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("slot correspondence failed: " + "; ".join(self.errors))


def validate_slot_correspondence(program: Program) -> SlotCorrespondenceReport:
    """Project host staging dependencies onto the bounded TLA+ lifecycle.

    A transfer is abstracted as ``BeginProduce; Publish`` and its matching GEMM
    as ``StartConsume; FinishSplit(1); Release``.  This is correspondence for
    compiler-owned host staging only.  A fused Program has no visible staging
    slot and is deliberately reported as not applicable; CUTLASS's internal
    split-K semaphore is outside this claim.
    """

    staging = tuple(buffer for buffer in program.buffers if buffer.role == "staging")
    if not staging:
        return SlotCorrespondenceReport(
            applicable=False,
            valid=True,
            epochs=(),
        )

    verification = verify_program(program.problem, program)
    if not verification.valid:
        return SlotCorrespondenceReport(
            applicable=True,
            valid=False,
            epochs=(),
            errors=verification.errors,
        )

    dependency_keys = {
        (dependency.src, dependency.dst, dependency.kind, dependency.buffer)
        for dependency in verification.dependence.dependencies
    }
    errors: list[str] = []
    epochs: list[SlotEpoch] = []

    for buffer in sorted(staging, key=lambda item: item.name):
        writers = [
            (index, op)
            for index, op in enumerate(program.ops)
            if buffer.name in op.writes
        ]
        if not writers:
            errors.append(f"staging buffer {buffer.name} has no producer")
            continue
        for epoch, (writer_index, writer) in enumerate(writers, start=1):
            next_writer_index = (
                writers[epoch][0] if epoch < len(writers) else len(program.ops)
            )
            readers = [
                op
                for op in program.ops[writer_index + 1 : next_writer_index]
                if buffer.name in op.reads
            ]
            if len(readers) != 1:
                errors.append(
                    f"staging buffer {buffer.name} epoch {epoch} has "
                    f"{len(readers)} consumers, expected exactly one"
                )
                continue
            reader = readers[0]
            raw = (writer.id, reader.id, "read_after_write", buffer.name)
            if raw not in dependency_keys:
                errors.append(
                    f"staging buffer {buffer.name} epoch {epoch} lacks publication edge"
                )
            if epoch < len(writers):
                next_writer = writers[epoch][1]
                reuse = (reader.id, next_writer.id, "buffer_reuse", buffer.name)
                if reuse not in dependency_keys:
                    errors.append(
                        f"staging buffer {buffer.name} epoch {epoch} lacks release edge"
                    )
            epochs.append(
                SlotEpoch(
                    buffer=buffer.name,
                    epoch=epoch,
                    producer=writer.id,
                    consumer=reader.id,
                )
            )

    return SlotCorrespondenceReport(
        applicable=True,
        valid=not errors,
        epochs=tuple(epochs),
        errors=tuple(errors),
    )


__all__ = ["SlotCorrespondenceReport", "SlotEpoch", "validate_slot_correspondence"]
