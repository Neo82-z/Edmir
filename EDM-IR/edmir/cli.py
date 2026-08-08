from __future__ import annotations

import argparse

from .compiler import (
    A2AGemmProblem,
    CalibrationProfile,
    CompilerOptions,
    Program,
    TopologySnapshot,
    compile_a2a_gemm,
    simulate_program,
    verify_program,
)
from .printing import format_critical_path, format_summary, format_uops
from .serialization import graph_from_dict, read_json, write_json
from .topology import probe_topology


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _load_program(path: str) -> Program:
    return Program.from_dict(read_json(path))


def _analyze(args: argparse.Namespace) -> int:
    graph = graph_from_dict(read_json(args.trace))
    print(format_uops(graph))
    print(format_critical_path(graph))
    print(format_summary(graph))
    return 0


def _probe(args: argparse.Namespace) -> int:
    topology = probe_topology()
    write_json(args.out, topology.to_dict())
    print(f"wrote {args.out}")
    return 0


def _compile(args: argparse.Namespace) -> int:
    problem = A2AGemmProblem.from_dict(read_json(args.problem))
    profile = CalibrationProfile.from_dict(read_json(args.profile))
    topology = TopologySnapshot.from_dict(read_json(args.topology))
    options = CompilerOptions(
        chunk_rows=_parse_csv_ints(args.chunk_rows) if args.chunk_rows else (),
        buffer_counts=_parse_csv_ints(args.buffers),
        min_gain_ratio=args.min_gain_ratio,
    )
    result = compile_a2a_gemm(problem, profile, topology, options)
    write_json(args.out, result.chosen.program.to_dict())
    for candidate in result.candidates:
        print(
            f"candidate strategy={candidate.program.strategy} chunk_rows={candidate.program.chunk_rows} "
            f"buffers={candidate.program.buffer_count} makespan_us={candidate.makespan_us:.3f} "
            f"ecr={candidate.ecr:.6f} edmr={candidate.edmr:.6f}"
        )
    print(f"chosen={result.chosen.program.strategy} out={args.out}")
    if result.fallback_reason:
        print(f"fallback={result.fallback_reason}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    program = _load_program(args.program)
    report = verify_program(program.problem, program)
    if report.valid:
        print("valid")
        return 0
    for error in report.errors:
        print(f"error={error}")
    return 1


def _inspect_program(args: argparse.Namespace) -> int:
    program = _load_program(args.program)
    report = verify_program(program.problem, program)
    print(
        f"program strategy={program.strategy} chunk_rows={program.chunk_rows} "
        f"buffers={program.buffer_count} ops={len(program.ops)} valid={str(report.valid).lower()}"
    )
    for op in program.ops:
        peer = "" if op.peer_rank is None else f" peer={op.peer_rank}"
        print(
            f"{op.id} kind={op.kind} rank={op.rank}{peer} "
            f"queue={op.queue} resources={','.join(op.resources())}"
        )
        print(f"  reads={','.join(op.reads) or '-'}")
        print(f"  writes={','.join(op.writes) or '-'}")
    if args.dependencies:
        print("dependencies:")
        for dependency in report.dependence.dependencies:
            buffer = "" if dependency.buffer is None else f" buffer={dependency.buffer}"
            print(f"  {dependency.src} -> {dependency.dst} kind={dependency.kind}{buffer}")
    for error in report.errors:
        print(f"error={error}")
    return 0 if report.valid else 1


def _simulate(args: argparse.Namespace) -> int:
    program = _load_program(args.program)
    profile = CalibrationProfile.from_dict(read_json(args.profile))
    result = simulate_program(program, profile)
    print(f"strategy={result.strategy}")
    print(f"makespan_us={result.makespan_us:.3f}")
    print(f"ecr={result.exposure.ecr:.6f}")
    print(f"edmr={result.exposure.edmr:.6f}")
    if args.timings:
        for timing in result.timings:
            print(f"{timing.op_id} {timing.start_us:.3f}->{timing.end_us:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edmir", description="EDM-IR analysis and schedule compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze an edmir.trace.v1 JSON graph")
    analyze.add_argument("trace")
    analyze.set_defaults(func=_analyze)

    probe = subparsers.add_parser("probe", help="capture the local NVIDIA P2P topology")
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=_probe)

    compile_parser = subparsers.add_parser("compile", help="compile a static All2All to GEMM problem")
    compile_parser.add_argument("problem")
    compile_parser.add_argument("--profile", required=True)
    compile_parser.add_argument("--topology", required=True)
    compile_parser.add_argument("--chunk-rows", default="")
    compile_parser.add_argument("--buffers", default="2")
    compile_parser.add_argument("--min-gain-ratio", type=float, default=0.0)
    compile_parser.add_argument("--out", required=True)
    compile_parser.set_defaults(func=_compile)

    verify = subparsers.add_parser("verify", help="verify an edmir.program.v1 program")
    verify.add_argument("program")
    verify.set_defaults(func=_verify)

    inspect_program = subparsers.add_parser(
        "inspect", help="print Program ops and optionally their derived dependencies"
    )
    inspect_program.add_argument("program")
    inspect_program.add_argument("--dependencies", action="store_true")
    inspect_program.set_defaults(func=_inspect_program)

    simulate = subparsers.add_parser("simulate", help="simulate an edmir.program.v1 program")
    simulate.add_argument("program")
    simulate.add_argument("--profile", required=True)
    simulate.add_argument("--timings", action="store_true")
    simulate.set_defaults(func=_simulate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
