from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Program


SCHEMA_REMOTE_PUBLICATION = "edmir.remote_publication.v1"
REMOTE_PUBLICATION_METADATA_KEY = "remote_input_contract"

IMMUTABLE_EXTERNAL = "immutable_external"
EXPLICIT_EPOCH = "explicit_epoch"


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


__all__ = [
    "EXPLICIT_EPOCH",
    "IMMUTABLE_EXTERNAL",
    "REMOTE_PUBLICATION_METADATA_KEY",
    "RemotePublicationContract",
    "RemotePublicationReport",
    "SCHEMA_REMOTE_PUBLICATION",
    "immutable_external_contract",
    "validate_remote_publication",
]
