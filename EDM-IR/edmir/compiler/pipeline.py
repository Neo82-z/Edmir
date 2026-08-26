from __future__ import annotations

from dataclasses import dataclass

from .backends import Backend, BackendArtifact
from .compile import compile_a2a_gemm
from .dataflow import DataflowModule
from .lowering import A2AGemmMatch, match_a2a_gemm
from .model import CalibrationProfile, CompileResult, CompilerOptions
from .passes import PassManager, PassTrace
from .target import Target


@dataclass(frozen=True, slots=True)
class PipelineResult:
    module: DataflowModule
    match: A2AGemmMatch
    compile_result: CompileResult
    artifact: BackendArtifact
    pass_trace: tuple[PassTrace, ...]


class CompilerPipeline:
    """Frontend Dataflow -> normalized graph -> Program -> backend artifact."""

    def __init__(self, pass_manager: PassManager | None = None) -> None:
        self.pass_manager = pass_manager or PassManager()

    def compile(
        self,
        module: DataflowModule,
        *,
        target: Target,
        profile: CalibrationProfile,
        backend: Backend,
        options: CompilerOptions | None = None,
    ) -> PipelineResult:
        target.validate()
        profile.validate()
        normalized, pass_trace = self.pass_manager.run(module)
        match = match_a2a_gemm(normalized)
        target.validate_problem(match.problem)
        if match.problem.world_size != target.world_size:
            raise ValueError("frontend world size does not match target topology")
        compile_result = compile_a2a_gemm(
            match.problem,
            profile,
            target.topology,
            options,
        )
        artifact = backend.emit(compile_result.chosen.program, target)
        return PipelineResult(
            module=normalized,
            match=match,
            compile_result=compile_result,
            artifact=artifact,
            pass_trace=pass_trace,
        )


__all__ = ["CompilerPipeline", "PipelineResult"]
