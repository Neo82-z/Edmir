from __future__ import annotations

import argparse

from .compiler import (
    A2AGemmProblem,
    CalibrationProfile,
    CompilerPipeline,
    CompilerOptions,
    DataflowModule,
    PassManager,
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
    lower_fused_a2a_gemm,
    match_a2a_gemm,
    read_runtime_measurements,
    create_backend,
    simulate_program,
    verify_program,
)
from .printing import format_critical_path, format_summary, format_uops
from .runtime_capabilities import probe_runtime_capabilities
from .serialization import graph_from_dict, read_json, write_json
from .topology import probe_topology
from .trace_validation import read_observed_gpu_events, validate_trace


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


def _probe_runtime(args: argparse.Namespace) -> int:
    report = probe_runtime_capabilities()
    write_json(args.out, report)
    ready = bool(report["features"]["fused_a2a_gemm_runtime_ready"])
    print(f"wrote {args.out}")
    print(
        f"gpus={len(report['gpus'])} "
        f"nccl={report['nccl']['version']} "
        f"nccl_device_api={str(report['nccl']['device_api_ready']).lower()} "
        f"fused_a2a_gemm_runtime={str(ready).lower()}"
    )
    for blocker in report["features"]["fused_a2a_gemm_blockers"]:
        print(f"blocker={blocker}")
    for error in report["probe_errors"]:
        print(f"probe_error={error}")
    return 2 if args.require_fused and not ready else 0


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
        gemm_provider=args.gemm_provider,
        gemm_kernel=args.gemm_kernel,
        communication_api=args.communication_api,
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
        f"backend={result.artifact.backend} "
        f"provider={result.artifact.metadata['gemm_provider']} "
        f"kernel={result.artifact.metadata['gemm_kernel']} "
        f"communication_api={result.artifact.metadata['communication_api']}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def _lower_fused(args: argparse.Namespace) -> int:
    module = DataflowModule.from_dict(read_json(args.graph))
    topology = TopologySnapshot.from_dict(read_json(args.topology))
    normalized, pass_trace = PassManager().run(module)
    match = match_a2a_gemm(normalized)
    target = Target(
        name=args.target,
        kind="cuda",
        arch=args.arch,
        gemm_provider=args.gemm_provider,
        gemm_kernel=args.gemm_kernel,
        communication_api="nccl_device",
        topology=topology,
    )
    program = lower_fused_a2a_gemm(match, chunk_rows=args.chunk_rows)
    verify_program(program.problem, program).require_valid()
    artifact = create_backend(args.backend).emit(program, target)
    written = artifact.write_to(args.out_dir)
    for trace in pass_trace:
        print(f"pass={trace.name} changed={str(trace.changed).lower()}")
    print(
        f"lowered={match.all_to_all.id}->{match.gemm.id} strategy=fused "
        f"communication_api=nccl_device backend={artifact.backend}"
    )
    if artifact.backend == "json":
        print("warning=fused JSON is a legality-checked semantic prototype")
    else:
        print(
            "warning=fused C++ requires EDMIR_ENABLE_NCCL_DEVICE=ON and remains "
            "excluded from cost-based selection"
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
    profile = fit_gemm_profile(
        base_profile,
        args.log,
        dtype=args.dtype,
        gemm_provider=args.gemm_provider,
        gemm_kernel=args.gemm_kernel
        or ("library_default" if args.gemm_provider == "cublas" else "auto"),
    )
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


def _validate_trace(args: argparse.Namespace) -> int:
    program = _load_program(args.program)
    events = read_observed_gpu_events(args.events)
    report = validate_trace(
        program,
        events,
        iteration=args.iteration,
        tolerance_us=args.tolerance_us,
    )
    if args.out:
        write_json(args.out, report.to_dict())
        print(f"wrote {args.out}")
    print(
        f"valid={str(report.valid).lower()} iteration={report.iteration} "
        f"covered_ops={report.covered_ops}/{report.expected_ops} "
        f"gpu_events={report.gpu_events} "
        f"dependency_device_checks={report.dependency_device_checks} "
        f"gpu_span_us={report.gpu_span_us:.3f} "
        f"comm_compute_overlap_us={report.comm_compute_overlap_us:.3f}"
    )
    for violation in report.violations:
        print(f"violation={violation.kind}: {violation.message}")
    return 0 if report.valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edmir", description="EDM-IR analysis and schedule compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="analyze an edmir.trace.v1 JSON graph")
    analyze.add_argument("trace")
    analyze.set_defaults(func=_analyze)

    probe = subparsers.add_parser("probe", help="capture the local NVIDIA P2P topology")
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=_probe)

    probe_runtime = subparsers.add_parser(
        "probe-runtime", help="capture CUDA/NCCL capabilities and fused-runtime blockers"
    )
    probe_runtime.add_argument("--out", required=True)
    probe_runtime.add_argument("--require-fused", action="store_true")
    probe_runtime.set_defaults(func=_probe_runtime)

    compile_parser = subparsers.add_parser("compile", help="compile a static All2All to GEMM problem")
    compile_parser.add_argument("problem")
    compile_parser.add_argument("--profile", required=True)
    compile_parser.add_argument("--topology", required=True)
    compile_parser.add_argument("--chunk-rows", default="")
    compile_parser.add_argument("--buffers", default="1,2,4")
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
    compile_graph.add_argument(
        "--gemm-provider", choices=("cublas", "cutlass"), default="cublas"
    )
    compile_graph.add_argument(
        "--gemm-kernel",
        choices=(
            "auto",
            "library_default",
            "sm80_128x128x32",
            "sm80_64x128x32",
            "sm80_128x64x32",
        ),
        default="auto",
    )
    compile_graph.add_argument(
        "--communication-api",
        choices=("host_nccl", "nccl_device"),
        default="host_nccl",
    )
    compile_graph.add_argument("--backend", choices=("json", "cpp"), default="json")
    compile_graph.add_argument("--chunk-rows", default="")
    compile_graph.add_argument("--buffers", default="1,2,4")
    compile_graph.add_argument("--min-gain-ratio", type=float, default=None)
    compile_graph.add_argument("--out-dir", required=True)
    compile_graph.set_defaults(func=_compile_graph)

    lower_fused = subparsers.add_parser(
        "lower-fused",
        help="emit a legality-checked fused All2All->GEMM JSON prototype",
    )
    lower_fused.add_argument("graph")
    lower_fused.add_argument("--topology", required=True)
    lower_fused.add_argument("--chunk-rows", required=True, type=int)
    lower_fused.add_argument("--target", default="cuda-device-prototype")
    lower_fused.add_argument("--arch", default="sm_80")
    lower_fused.add_argument(
        "--gemm-provider",
        choices=("cuda_reference", "cutlass"),
        default="cuda_reference",
    )
    lower_fused.add_argument(
        "--gemm-kernel",
        choices=(
            "auto",
            "library_default",
            "sm80_128x128x32",
            "sm80_64x128x32",
            "sm80_128x64x32",
            "lsa_naive_f16",
        ),
        default="auto",
    )
    lower_fused.add_argument("--out-dir", required=True)
    lower_fused.add_argument("--backend", choices=("json", "cpp"), default="json")
    lower_fused.set_defaults(func=_lower_fused)

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
    fit_gemm.add_argument("--gemm-provider", choices=("cublas", "cutlass"), default="cublas")
    fit_gemm.add_argument("--gemm-kernel")
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

    validate_trace_parser = subparsers.add_parser(
        "validate-trace", help="compare mapped Nsight GPU events with Program dependencies"
    )
    validate_trace_parser.add_argument("program")
    validate_trace_parser.add_argument("events", help="edm_cuda_events.csv from nsys_export")
    validate_trace_parser.add_argument("--iteration", type=int)
    validate_trace_parser.add_argument("--tolerance-us", type=float, default=0.5)
    validate_trace_parser.add_argument("--out")
    validate_trace_parser.set_defaults(func=_validate_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
