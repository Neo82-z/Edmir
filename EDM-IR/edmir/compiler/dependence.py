from __future__ import annotations

from dataclasses import dataclass

from .model import BUFFER_ROLES, OP_KINDS, QUEUE_KINDS, Program


EXTERNAL_BUFFER_ROLES = frozenset({"input", "weight"})


@dataclass(frozen=True, slots=True)
class Dependency:
    src: str
    dst: str
    kind: str
    buffer: str | None = None


@dataclass(frozen=True, slots=True)
class DependenceResult:
    dependencies: tuple[Dependency, ...]
    topological_order: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def predecessors(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {op_id: [] for op_id in self.topological_order}
        for dependency in self.dependencies:
            if dependency.src not in result[dependency.dst]:
                result[dependency.dst].append(dependency.src)
        return {op_id: tuple(values) for op_id, values in result.items()}

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("invalid program dependencies: " + "; ".join(self.errors))


def _topological_order(op_ids: tuple[str, ...], dependencies: list[Dependency]) -> tuple[str, ...]:
    predecessors = {op_id: set() for op_id in op_ids}
    successors = {op_id: set() for op_id in op_ids}
    for dependency in dependencies:
        if dependency.src not in successors or dependency.dst not in predecessors:
            continue
        predecessors[dependency.dst].add(dependency.src)
        successors[dependency.src].add(dependency.dst)
    ready = [op_id for op_id in op_ids if not predecessors[op_id]]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dst in op_ids:
            if current not in predecessors[dst]:
                continue
            predecessors[dst].remove(current)
            if not predecessors[dst] and dst not in order and dst not in ready:
                ready.append(dst)
    return tuple(order)


def derive_dependencies(program: Program) -> DependenceResult:
    """Derive queue order and RAW/WAR/WAW hazards from an ordered Program.

    Buffer dependencies carry release/acquire semantics by construction. The
    Program never serializes or asks callers to maintain a second edge list.
    """

    errors: list[str] = []
    buffer_map = program.buffer_map()
    op_map = program.op_map()
    if len(buffer_map) != len(program.buffers):
        errors.append("duplicate buffer name")
    if len(op_map) != len(program.ops):
        errors.append("duplicate op id")

    for buffer in program.buffers:
        if buffer.role not in BUFFER_ROLES:
            errors.append(f"buffer {buffer.name} has unknown role {buffer.role!r}")
        if not 0 <= buffer.rank < program.problem.world_size:
            errors.append(f"buffer {buffer.name} has invalid rank {buffer.rank}")
        if buffer.size_bytes < 0:
            errors.append(f"buffer {buffer.name} has negative size")

    dependencies: list[Dependency] = []
    dependency_keys: set[tuple[str, str, str, str | None]] = set()

    def add(src: str, dst: str, kind: str, buffer: str | None = None) -> None:
        if src == dst:
            return
        key = (src, dst, kind, buffer)
        if key not in dependency_keys:
            dependency_keys.add(key)
            dependencies.append(Dependency(src, dst, kind, buffer))

    last_on_resource: dict[str, str] = {}
    last_writer: dict[str, str] = {}
    active_readers: dict[str, set[str]] = {name: set() for name in buffer_map}

    for op in program.ops:
        if op.kind not in OP_KINDS:
            errors.append(f"op {op.id} has unknown kind {op.kind!r}")
        if op.queue not in QUEUE_KINDS:
            errors.append(f"op {op.id} has unknown queue {op.queue!r}")
        if not 0 <= op.rank < program.problem.world_size:
            errors.append(f"op {op.id} has invalid rank {op.rank}")
        if op.kind == "transfer":
            if op.queue != "comm":
                errors.append(f"transfer {op.id} must use the comm queue")
            if op.peer_rank is None or not 0 <= op.peer_rank < program.problem.world_size:
                errors.append(f"transfer {op.id} has invalid peer rank")
        elif op.kind == "gemm" and op.queue != "compute":
            errors.append(f"GEMM {op.id} must use the compute queue")
        elif op.kind == "barrier" and op.queue != "control":
            errors.append(f"barrier {op.id} must use the control queue")

        for resource in op.resources():
            previous = last_on_resource.get(resource)
            if previous is not None:
                add(previous, op.id, "queue")
            last_on_resource[resource] = op.id

        missing = [name for name in (*op.reads, *op.writes) if name not in buffer_map]
        for name in missing:
            errors.append(f"op {op.id} references unknown buffer {name}")

        for name in op.reads:
            buffer = buffer_map.get(name)
            if buffer is None:
                continue
            writer = last_writer.get(name)
            if writer is not None:
                add(writer, op.id, "read_after_write", name)
            elif buffer.role not in EXTERNAL_BUFFER_ROLES:
                errors.append(f"uninitialized read: op {op.id} reads {name}")
            active_readers[name].add(op.id)

        for name in op.writes:
            if name not in buffer_map:
                continue
            writer = last_writer.get(name)
            if writer is not None:
                add(writer, op.id, "write_after_write", name)
            for reader in sorted(active_readers[name]):
                add(reader, op.id, "buffer_reuse", name)
            active_readers[name].clear()
            last_writer[name] = op.id

    op_ids = tuple(op.id for op in program.ops)
    order = _topological_order(op_ids, dependencies)
    if len(order) != len(set(op_ids)):
        errors.append("derived dependency graph contains a cycle")
        order = tuple(dict.fromkeys(op_ids))

    return DependenceResult(
        dependencies=tuple(dependencies),
        topological_order=order,
        errors=tuple(dict.fromkeys(errors)),
    )


__all__ = ["DependenceResult", "Dependency", "derive_dependencies"]
