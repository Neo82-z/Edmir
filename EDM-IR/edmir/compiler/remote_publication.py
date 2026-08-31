from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Program


SCHEMA_REMOTE_PUBLICATION = "edmir.remote_publication.v1"
REMOTE_PUBLICATION_METADATA_KEY = "remote_input_contract"

IMMUTABLE_EXTERNAL = "immutable_external"
EXPLICIT_EPOCH = "explicit_epoch"

PUBLICATION_ACTIONS = {
    "BeginWrite",
    "Publish",
    "Acquire",
    "Consume",
    "Retire",
}


@dataclass(frozen=True, slots=True)
class RemotePublicationContract:
    """Visibility and reuse contract for source activation buffers.

    The current executable Program admits only immutable inputs initialized
    before launch.  ``explicit_epoch`` reserves the contract for future
    producer-generated activations, but cannot become executable until the IR
    exposes matching publish, acquire, completion, and reuse events.
    """

    mode: str
    visibility: str
    reuse: str

    def validate(self) -> None:
        if self.mode == IMMUTABLE_EXTERNAL:
            if self.visibility != "initialized_before_launch":
                raise ValueError(
                    "immutable_external visibility must be initialized_before_launch"
                )
            if self.reuse != "none_during_program":
                raise ValueError(
                    "immutable_external reuse must be none_during_program"
                )
            return
        if self.mode == EXPLICIT_EPOCH:
            if self.visibility != "release_acquire":
                raise ValueError("explicit_epoch visibility must be release_acquire")
            if self.reuse != "all_consumers_complete":
                raise ValueError(
                    "explicit_epoch reuse must be all_consumers_complete"
                )
            return
        raise ValueError(f"unknown remote publication mode {self.mode!r}")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "schema": SCHEMA_REMOTE_PUBLICATION,
            "mode": self.mode,
            "visibility": self.visibility,
            "reuse": self.reuse,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RemotePublicationContract":
        if value.get("schema") != SCHEMA_REMOTE_PUBLICATION:
            raise ValueError(
                f"remote publication contract requires schema "
                f"{SCHEMA_REMOTE_PUBLICATION!r}"
            )
        contract = cls(
            mode=str(value.get("mode", "")),
            visibility=str(value.get("visibility", "")),
            reuse=str(value.get("reuse", "")),
        )
        contract.validate()
        return contract


@dataclass(frozen=True, slots=True)
class RemotePublicationReport:
    valid: bool
    mode: str | None
    executable: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicationEvent:
    """One implementation-visible action in ``RemotePublication.tla``."""

    action: str
    epoch: int
    consumer: int | None = None


@dataclass(frozen=True, slots=True)
class PublicationTraceReport:
    valid: bool
    complete: bool
    actions: tuple[str, ...]
    retired_epochs: tuple[int, ...]
    overwritten_epochs: tuple[int, ...]
    errors: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("invalid publication trace: " + "; ".join(self.errors))


def immutable_external_contract() -> RemotePublicationContract:
    return RemotePublicationContract(
        mode=IMMUTABLE_EXTERNAL,
        visibility="initialized_before_launch",
        reuse="none_during_program",
    )


def validate_remote_publication(program: Program) -> RemotePublicationReport:
    """Reject source-buffer semantics that the executable IR cannot enforce."""

    remote_reads = [
        op
        for op in program.ops
        if op.kind in {"transfer", "fused_transfer_gemm"}
        and op.peer_rank is not None
        and op.peer_rank != op.rank
    ]
    if not remote_reads:
        return RemotePublicationReport(
            valid=True,
            mode=None,
            executable=True,
            errors=(),
        )

    raw = program.metadata.get(REMOTE_PUBLICATION_METADATA_KEY)
    if not isinstance(raw, dict):
        error = (
            "remote input reads require a versioned remote_input_contract; "
            "implicit mutability is not accepted"
        )
        return RemotePublicationReport(False, None, False, (error,))

    try:
        contract = RemotePublicationContract.from_dict(raw)
    except ValueError as exc:
        return RemotePublicationReport(False, None, False, (str(exc),))

    if contract.mode == EXPLICIT_EPOCH:
        error = (
            "explicit_epoch remote inputs require Program-visible publish, "
            "acquire, completion, and reuse events; this runtime path is not implemented"
        )
        return RemotePublicationReport(False, contract.mode, False, (error,))

    return RemotePublicationReport(True, contract.mode, True, ())


def validate_publication_trace(
    events: tuple[PublicationEvent, ...],
    *,
    epoch_count: int,
    consumer_count: int,
    require_complete: bool = True,
) -> PublicationTraceReport:
    """Execute the safe transition relation from ``RemotePublication.tla``.

    This correspondence checker is intentionally independent of CUDA and the
    executable Program.  A future runtime may emit this event surface, but the
    ``explicit_epoch`` compiler path remains closed until those events are
    generated from real synchronization primitives.
    """

    if epoch_count <= 0 or consumer_count <= 0:
        error = "[PUB-TYPE] epoch_count and consumer_count must be positive"
        return PublicationTraceReport(False, False, (), (), (), (error,))

    phase = "Empty"
    current = 0
    next_epoch = 1
    published: set[int] = set()
    acquired: set[tuple[int, int]] = set()
    consumed: set[tuple[int, int]] = set()
    retired: set[int] = set()
    overwritten: set[int] = set()
    actions: list[str] = []
    errors: list[str] = []

    def all_consumed(epoch: int) -> bool:
        return all(
            (epoch, consumer) in consumed
            for consumer in range(1, consumer_count + 1)
        )

    def reject(index: int, code: str, message: str) -> None:
        errors.append(f"event {index}: [{code}] {message}")

    for index, event in enumerate(events):
        if event.action not in PUBLICATION_ACTIONS:
            reject(index, "PUB-TYPE", f"unknown action {event.action!r}")
            break
        if event.epoch < 1 or event.epoch > epoch_count:
            reject(index, "PUB-TYPE", f"epoch {event.epoch} is outside the contract")
            break
        if event.action in {"Acquire", "Consume"}:
            if event.consumer is None or not 1 <= event.consumer <= consumer_count:
                reject(index, "PUB-TYPE", "consumer is outside the contract")
                break
        elif event.consumer is not None:
            reject(index, "PUB-TYPE", f"{event.action} must not name a consumer")
            break

        if event.action == "BeginWrite":
            if event.epoch != next_epoch:
                reject(index, "PUB-EPOCH", "writes must use the next epoch")
                break
            if phase not in {"Empty", "Retired"}:
                reject(
                    index,
                    "PUB-4",
                    "overwrite requires retirement and completion of every consumer",
                )
                break
            if current:
                if not all_consumed(current):
                    reject(index, "PUB-4", "the prior epoch is not fully consumed")
                    break
                overwritten.add(current)
            phase = "Writing"
            current = event.epoch
            next_epoch += 1
        elif event.action == "Publish":
            if phase != "Writing" or event.epoch != current:
                reject(index, "PUB-1", "only the current writing epoch can publish")
                break
            phase = "Published"
            published.add(current)
        elif event.action == "Acquire":
            key = (event.epoch, event.consumer)
            if phase != "Published" or event.epoch != current or current not in published:
                reject(index, "PUB-1", "acquire precedes publication of its epoch")
                break
            if key in acquired:
                reject(index, "PUB-EPOCH", "consumer acquired the epoch more than once")
                break
            acquired.add(key)
        elif event.action == "Consume":
            key = (event.epoch, event.consumer)
            if phase != "Published" or event.epoch != current or key not in acquired:
                reject(index, "PUB-2", "consume precedes the matching acquire")
                break
            if key in consumed:
                reject(index, "PUB-EPOCH", "consumer consumed the epoch more than once")
                break
            consumed.add(key)
        elif event.action == "Retire":
            if phase != "Published" or event.epoch != current:
                reject(index, "PUB-3", "only the current published epoch can retire")
                break
            if not all_consumed(current):
                reject(index, "PUB-3", "retirement precedes completion of every consumer")
                break
            phase = "Retired"
            retired.add(current)
        actions.append(event.action)

    complete = retired == set(range(1, epoch_count + 1))
    if not errors and require_complete and not complete:
        errors.append("[PUB-LIVE] trace ends before every epoch retires")
    return PublicationTraceReport(
        valid=not errors,
        complete=complete,
        actions=tuple(actions),
        retired_epochs=tuple(sorted(retired)),
        overwritten_epochs=tuple(sorted(overwritten)),
        errors=tuple(errors),
    )


__all__ = [
    "EXPLICIT_EPOCH",
    "IMMUTABLE_EXTERNAL",
    "PUBLICATION_ACTIONS",
    "PublicationEvent",
    "PublicationTraceReport",
    "REMOTE_PUBLICATION_METADATA_KEY",
    "RemotePublicationContract",
    "RemotePublicationReport",
    "SCHEMA_REMOTE_PUBLICATION",
    "immutable_external_contract",
    "validate_publication_trace",
    "validate_remote_publication",
]
