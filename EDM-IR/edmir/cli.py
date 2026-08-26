from __future__ import annotations

import argparse

from .compiler import (
    A2AGemmProblem,
    CalibrationProfile,
    CompilerPipeline,
    CompilerOptions,
    DataflowModule,
    Program,
    TopologySnapshot,
    Target,
    build_a2a_gemm_module,
    compile_a2a_gemm,
    evaluate_cost_model,
    fit_execution_overhead,
    fit_gemm_profile,
    fit_local_copy_profile,
    fit_pairwise_nccl_profile,
    read_runtime_measurements,
    create_backend,
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
            f"device_us={candidate.device_makespan_us:.3f} overhead_us={candidate.overhead_us:.3f} "
            f"ecr={candidate.ecr:.6f} edmr={candidate.edmr:.6f}"
        )
    print(f"chosen={result.chosen.program.strategy} out={args.out}")
    if result.fallback_reason:
        print(f"fallback={result.fallback_reason}")
    return 0


def _frontend(args: argparse.Namespace) -> int:
    problem = A2AGemmProblem.from_dict(read_json(args.problem))
    module = build_a2a_gemm_module(problem)
    write_json(args.out, module.to_dict())
    print(f"wrote {args.out} nodes={len(module.nodes)} values={len(module.values)}")
    return 0


def _compile_graph(args: argparse.Namespace) -> int:
    module = DataflowModule.from_dict(read_json(args.graph))
    profile = CalibrationProfile.from_dict(read_json(args.profile))
    topology = TopologySnapshot.from_dict(read_json(args.topology))
    target = Target(
        name=args.target,
        kind="cuda",
        arch=args.arch,
        topology=topology,
    )
    options = CompilerOptions(
        chunk_rows=_parse_csv_ints(args.chunk_rows) if args.chunk_rows else (),
        buffer_counts=_parse_csv_ints(args.buffers),
        min_gain_ratio=args.min_gain_ratio,
    )
    result = CompilerPipeline().compile(
        module,
        target=target,
        profile=profile,
        backend=create_backend(args.backend),
        options=options,
    )
    written = result.artifact.write_to(args.out_dir)
    for trace in result.pass_trace:
        print(f"pass={trace.name} changed={str(trace.changed).lower()}")
    print(
        f"lowered={result.match.all_to_all.id}->{result.match.gemm.id} "
        f"strategy={result.compile_result.chosen.program.strategy} "
        f"backend={result.artifact.backend}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def _fit_nccl(args: argparse.Namespace) -> int:
    base_profile = CalibrationProfile.from_dict(read_json(args.base_profile))
    profile = fit_pairwise_nccl_profile(base_profile, args.logs)
    write_json(args.out, profile.to_dict())
    pairs = len(profile.pairwise_comm_curves)
    samples = sum(len(curve.samples) for curve in profile.pairwise_comm_curves)
    print(f"wrote {args.out} directed_pairs={pairs} samples={samples}")
    print("warning=GEMM and fallback parameters are inherited from the base profile")
    return 0


def _fit_gemm(args: argparse.Namespace) -> int:
    base_profile = CalibrationProfile.from_dict(read_json(args.base_profile))
    profile = fit_gemm_profile(base_profile, args.log, dtype=args.dtype)
    write_json(args.out, profile.to_dict())
    print(f"wrote {args.out} gemm_samples={len(profile.gemm_samples)}")
    return 0


def _fit_copy(args: argparse.Namespace) -> int:
    base_profile = CalibrationProfile.from_dict(read_json(args.base_profile))
    profile = fit_local_copy_profile(base_profile, args.log)
    write_json(args.out, profile.to_dict())
    print(f"wrote {args.out} local_copy_samples={len(profile.local_copy_curve)}")
    return 0


def _fit_runtime(args: argparse.Namespace) -> int:
    base_profile = CalibrationProfile.from_dict(read_json(args.base_profile))
    measurements = read_runtime_measurements(args.measurements)
    profile = fit_execution_overhead(base_profile, measurements, ridge=args.ridge)
    report = evaluate_cost_model(profile, measurements)
    write_json(args.out_profile, profile.to_dict())
    write_json(args.out_report, report)
    print(f"wrote {args.out_profile}")
    print(f"wrote {args.out_report}")
    for split in ("train", "heldout"):
        metrics = report[split]
        mape = metrics["mape"]
        label = "n/a" if mape is None else f"{100.0 * mape:.2f}%"
        print(f"{split}_rows={metrics['rows']} {split}_mape={label}")
    direction = report["direction_accuracy"]
    label = "n/a" if direction is None else f"{100.0 * direction:.2f}%"
    print(f"direction_accuracy={label}")
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
    print(f"device_makespan_us={result.device_makespan_us:.3f}")
    print(f"overhead_us={result.overhead_us:.3f}")
    print(
        f"features op_dispatch={result.features.op_dispatch_count} "
        f"event_record={result.features.event_record_count} "
        f"event_wait={result.features.event_wait_count} "
        f"buffer_reuse={result.features.buffer_reuse_count}"
    )
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
    compile_parser.add_argument("--min-gain-ratio", type=float, default=None)
    compile_parser.add_argument("--out", required=True)
    compile_parser.set_defaults(func=_compile)

    frontend = subparsers.add_parser(
        "frontend", help="lower an a2a_gemm problem description to typed dataflow IR"
    )
    frontend.add_argument("problem")
    frontend.add_argument("--out", required=True)
    frontend.set_defaults(func=_frontend)

    compile_graph = subparsers.add_parser(
        "compile-graph", help="run Dataflow IR through Program lowering and a backend"
    )
    compile_graph.add_argument("graph")
    compile_graph.add_argument("--profile", required=True)
    compile_graph.add_argument("--topology", required=True)
    compile_graph.add_argument("--target", default="cuda")
    compile_graph.add_argument("--arch", default="unknown")
    compile_graph.add_argument("--backend", choices=("json", "cpp"), default="json")
    compile_graph.add_argument("--chunk-rows", default="")
    compile_graph.add_argument("--buffers", default="2")
    compile_graph.add_argument("--min-gain-ratio", type=float, default=None)
    compile_graph.add_argument("--out-dir", required=True)
    compile_graph.set_defaults(func=_compile_graph)

    fit_nccl = subparsers.add_parser(
        "fit-nccl", help="attach pairwise nccl-tests sendrecv curves to a profile"
    )
    fit_nccl.add_argument("--base-profile", required=True)
    fit_nccl.add_argument("--logs", required=True)
    fit_nccl.add_argument("--out", required=True)
    fit_nccl.set_defaults(func=_fit_nccl)

    fit_gemm = subparsers.add_parser(
        "fit-gemm", help="attach shape-specific JSON-lines GEMM measurements to a profile"
    )
    fit_gemm.add_argument("--base-profile", required=True)
    fit_gemm.add_argument("--log", required=True)
    fit_gemm.add_argument("--dtype", default="float16")
    fit_gemm.add_argument("--out", required=True)
    fit_gemm.set_defaults(func=_fit_gemm)

    fit_copy = subparsers.add_parser(
        "fit-copy", help="attach JSON-lines local D2D copy measurements to a profile"
    )
    fit_copy.add_argument("--base-profile", required=True)
    fit_copy.add_argument("--log", required=True)
    fit_copy.add_argument("--out", required=True)
    fit_copy.set_defaults(func=_fit_copy)

    fit_runtime = subparsers.add_parser(
        "fit-runtime", help="fit plan-feature overheads and evaluate train/heldout measurements"
    )
    fit_runtime.add_argument("--base-profile", required=True)
    fit_runtime.add_argument("--measurements", required=True)
    fit_runtime.add_argument("--ridge", type=float, default=1e-6)
    fit_runtime.add_argument("--out-profile", required=True)
    fit_runtime.add_argument("--out-report", required=True)
    fit_runtime.set_defaults(func=_fit_runtime)

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
