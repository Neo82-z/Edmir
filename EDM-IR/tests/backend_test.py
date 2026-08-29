from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edmir.compiler import (  # noqa: E402
    A2AGemmProblem,
    CalibrationProfile,
    CompilerOptions,
    CompilerPipeline,
    CppManifestBackend,
    FusedScheduleContract,
    JsonBackend,
    Target,
    TopologySnapshot,
    build_a2a_gemm_module,
    build_fused_program,
    build_sequential_program,
)


def topology(world_size: int = 4) -> TopologySnapshot:
    return TopologySnapshot(
        world_size=world_size,
        devices=tuple("test-gpu" for _ in range(world_size)),
        p2p=tuple(tuple(True for _ in range(world_size)) for _ in range(world_size)),
    )


def profile() -> CalibrationProfile:
    return CalibrationProfile(
        comm_latency_us=1.0,
        comm_bandwidth_gbps=50.0,
        gemm_launch_us=0.5,
        gemm_tflops=100.0,
        source="backend-test",
    )


class BackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = A2AGemmProblem(world_size=4, rows_per_peer=32, k=128, n=256)
        self.module = build_a2a_gemm_module(self.problem)
        self.target = Target("test-cuda", topology(), arch="sm_89")
        self.options = CompilerOptions(chunk_rows=(16, 32), buffer_counts=(1, 2))

    def compile_with(self, backend):
        return CompilerPipeline().compile(
            self.module,
            target=self.target,
            profile=profile(),
            backend=backend,
            options=self.options,
        )

    def cutlass_profile(self) -> CalibrationProfile:
        return replace(profile(), gemm_provider="cutlass", gemm_kernel="auto")

    def test_json_backend_emits_selected_program(self) -> None:
        result = self.compile_with(JsonBackend())
        self.assertIn('"schema": "edmir.program.v1"', result.artifact.files["program.json"])
        self.assertIn('"schema": "edmir.target.v1"', result.artifact.files["target.json"])
        self.assertEqual(result.match.problem, self.problem)
        self.assertTrue(result.compile_result.candidates)

    def test_target_json_round_trip(self) -> None:
        self.assertEqual(Target.from_dict(self.target.to_dict()), self.target)

    def test_grouped_n2_schedule_lowers_to_the_executable_contract(self) -> None:
        problem = replace(self.problem, n=512)
        program = build_fused_program(problem, chunk_rows=32)
        direct_target = Target(
            "device-direct",
            topology(),
            arch="sm_80",
            gemm_provider="cutlass",
            gemm_kernel="sm80_64x128x32",
            communication_api="nccl_device",
        )
        grouped_target = replace(
            direct_target,
            name="device-grouped-n2",
            fused_schedule=FusedScheduleContract(n_tiles_per_cta=2),
        )

        direct = direct_target.estimate_fused_traffic(program)
        grouped = grouped_target.estimate_fused_traffic(program)
        self.assertEqual(direct.n_tiles, 4)
        self.assertEqual(direct.remote_a_replay, 4)
        self.assertEqual(grouped.n_tiles, 4)
        self.assertEqual(grouped.n_tiles_per_cta, 2)
        self.assertEqual(grouped.remote_a_replay, 2)
        self.assertEqual(
            grouped.estimated_remote_a_load_bytes,
            direct.estimated_remote_a_load_bytes // 2,
        )
        self.assertEqual(grouped.estimated_ctas_per_op, 2)

        artifact = JsonBackend().emit(program, grouped_target)
        emitted_target = json.loads(artifact.files["target.json"])
        self.assertEqual(emitted_target["fused_schedule"]["n_tiles_per_cta"], 2)
        self.assertTrue(artifact.metadata["schedule_execution_ready"])
        self.assertEqual(Target.from_dict(emitted_target), grouped_target)
        cpp_artifact = CppManifestBackend().emit(program, grouped_target)
        self.assertIn(
            "FusedScheduleDesc kFusedSchedule{2}",
            cpp_artifact.files["edmir_program.hpp"],
        )
        analysis_only = replace(
            grouped_target,
            fused_schedule=FusedScheduleContract(n_tiles_per_cta=4),
        )
        with self.assertRaisesRegex(ValueError, "analysis-only"):
            CppManifestBackend().emit(program, analysis_only)

    def test_grouped_n_schedule_rejects_nonfused_and_nondevice_targets(self) -> None:
        grouped = FusedScheduleContract(n_tiles_per_cta=2)
        with self.assertRaisesRegex(ValueError, "gemm_provider=cutlass"):
            replace(self.target, fused_schedule=grouped).validate()
        with self.assertRaisesRegex(ValueError, "communication_api=nccl_device"):
            Target(
                "bad-grouped-host",
                topology(),
                arch="sm_80",
                gemm_provider="cutlass",
                gemm_kernel="sm80_64x128x32",
                fused_schedule=grouped,
            ).validate()
        with self.assertRaisesRegex(ValueError, "all-fused Program"):
            Target(
                "bad-grouped-program",
                topology(),
                arch="sm_80",
                gemm_provider="cutlass",
                gemm_kernel="sm80_64x128x32",
                communication_api="nccl_device",
                fused_schedule=grouped,
            ).validate_program(build_sequential_program(self.problem))

    def test_fused_program_requires_device_api_and_json_can_preserve_semantics(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        with self.assertRaisesRegex(ValueError, "communication_api=nccl_device"):
            JsonBackend().emit(program, self.target)
        with self.assertRaisesRegex(ValueError, "gemm_provider=cutlass or cuda_reference"):
            JsonBackend().emit(
                program, replace(self.target, communication_api="nccl_device")
            )
        device_target = Target(
            "device-prototype",
            topology(),
            arch="sm_80",
            gemm_provider="cutlass",
            communication_api="nccl_device",
        )
        artifact = JsonBackend().emit(program, device_target)
        target_json = json.loads(artifact.files["target.json"])
        program_json = json.loads(artifact.files["program.json"])
        self.assertEqual(target_json["communication_api"], "nccl_device")
        self.assertEqual(program_json["strategy"], "fused")

    def test_cpp_backend_emits_reference_and_cutlass_fused_runtimes(self) -> None:
        program = build_fused_program(self.problem, chunk_rows=16)
        device_target = Target(
            "device-prototype",
            topology(),
            arch="sm_80",
            gemm_provider="cutlass",
            communication_api="nccl_device",
        )
        cutlass_artifact = CppManifestBackend().emit(program, device_target)
        cutlass_header = cutlass_artifact.files["edmir_program.hpp"]
        self.assertIn("GemmProvider::kCUTLASS", cutlass_header)
        self.assertIn("GemmKernel::kSm80_64x128x32", cutlass_header)
        self.assertIn("CommunicationApi::kNCCLDevice", cutlass_header)
        self.assertIn("FusedScheduleDesc kFusedSchedule{1}", cutlass_header)
        self.assertEqual(cutlass_artifact.metadata["remote_ops"], 24)
        self.assertEqual(cutlass_artifact.metadata["remote_input_bytes"], 98_304)
        self.assertEqual(cutlass_artifact.metadata["remote_a_replay"], 2)
        self.assertEqual(cutlass_artifact.metadata["split_k_slices"], 1)
        self.assertEqual(cutlass_artifact.metadata["estimated_ctas_per_op"], 2)
        self.assertEqual(
            cutlass_artifact.metadata["estimated_remote_a_load_bytes"], 196_608
        )
        reference_target = replace(device_target, gemm_provider="cuda_reference")
        artifact = CppManifestBackend().emit(program, reference_target)
        header = artifact.files["edmir_program.hpp"]
        self.assertIn("OpKind::kFusedTransferGemm", header)
        self.assertIn("GemmProvider::kCudaReference", header)
        self.assertIn("GemmKernel::kLsaNaiveF16", header)
        self.assertIn("CommunicationApi::kNCCLDevice", header)
        self.assertEqual(artifact.metadata["remote_a_replay"], 256)
        self.assertEqual(artifact.metadata["estimated_ctas_per_op"], 16)

        split_problem = replace(self.problem, k=256)
        split_program = build_fused_program(split_problem, chunk_rows=16)
        split_target = replace(
            device_target, gemm_kernel="sm80_64x128x32_splitk8"
        )
        split_artifact = CppManifestBackend().emit(split_program, split_target)
        self.assertIn(
            "GemmKernel::kSm80_64x128x32SplitK8",
            split_artifact.files["edmir_program.hpp"],
        )
        self.assertEqual(split_artifact.metadata["split_k_slices"], 8)
        self.assertEqual(split_artifact.metadata["estimated_ctas_per_op"], 16)

    def test_cpp_backend_contains_buffers_ops_accesses_and_dependencies(self) -> None:
        result = self.compile_with(CppManifestBackend())
        header = result.artifact.files["edmir_program.hpp"]
        self.assertIn("kBuffers", header)
        self.assertIn("kOps", header)
        self.assertIn("kAccesses", header)
        self.assertIn("kDependencies", header)
        self.assertIn('kArch = "sm_89"', header)
        self.assertIn("kGemmProvider = GemmProvider::kCuBLAS", header)
        self.assertIn("kGemmKernel = GemmKernel::kLibraryDefault", header)
        self.assertIn("kCommunicationApi = CommunicationApi::kHostNCCL", header)

    def test_cutlass_target_accepts_sm80_fp16_aligned_problem(self) -> None:
        target = Target(
            "test-cutlass", topology(), arch="sm_80", gemm_provider="cutlass"
        )
        result = CompilerPipeline().compile(
            self.module,
            target=target,
            profile=self.cutlass_profile(),
            backend=CppManifestBackend(),
            options=self.options,
        )
        self.assertIn(
            "kGemmProvider = GemmProvider::kCUTLASS",
            result.artifact.files["edmir_program.hpp"],
        )
        self.assertIn(
            "kGemmKernel = GemmKernel::kSm80_64x128x32",
            result.artifact.files["edmir_program.hpp"],
        )
        emitted_target = json.loads(result.artifact.files["target.json"])
        self.assertEqual(emitted_target["gemm_kernel"], "sm80_64x128x32")

    def test_cutlass_auto_kernel_uses_lowered_gemm_m(self) -> None:
        target = Target(
            "test-cutlass", topology(), arch="sm_80", gemm_provider="cutlass"
        )
        small_m = target.resolve(self.problem, gemm_m=96)
        large_m = target.resolve(self.problem, gemm_m=128)
        self.assertEqual(small_m.gemm_kernel, "sm80_64x128x32")
        self.assertEqual(large_m.gemm_kernel, "sm80_128x128x32")

    def test_cutlass_target_materializes_each_explicit_kernel(self) -> None:
        for kernel in (
            "sm80_128x128x32",
            "sm80_64x128x32",
            "sm80_128x64x32",
        ):
            with self.subTest(kernel=kernel):
                target = Target(
                    "test-cutlass",
                    topology(),
                    arch="sm_80",
                    gemm_provider="cutlass",
                    gemm_kernel=kernel,
                )
                result = CompilerPipeline().compile(
                    self.module,
                    target=target,
                    profile=replace(
                        self.cutlass_profile(), gemm_kernel=kernel
                    ),
                    backend=CppManifestBackend(),
                    options=self.options,
                )
                emitted_target = json.loads(result.artifact.files["target.json"])
                self.assertEqual(emitted_target["gemm_kernel"], kernel)

    def test_cutlass_target_rejects_unsupported_arch_dtype_and_alignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "arch=sm_80"):
            Target(
                "bad-arch", topology(), arch="sm_89", gemm_provider="cutlass"
            ).validate_problem(self.problem)
        with self.assertRaisesRegex(ValueError, "float16"):
            Target(
                "bad-dtype", topology(), arch="sm_80", gemm_provider="cutlass"
            ).validate_problem(
                A2AGemmProblem(4, 32, 128, 256, dtype="float32")
            )
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            Target(
                "bad-alignment", topology(), arch="sm_80", gemm_provider="cutlass"
            ).validate_problem(A2AGemmProblem(4, 32, 130, 256))
        with self.assertRaisesRegex(ValueError, "cuBLAS targets"):
            Target(
                "bad-cublas-kernel",
                topology(),
                arch="sm_80",
                gemm_provider="cublas",
                gemm_kernel="sm80_128x128x32",
            ).validate_problem(self.problem)
        with self.assertRaisesRegex(ValueError, "CUTLASS targets"):
            Target(
                "bad-cutlass-kernel",
                topology(),
                arch="sm_80",
                gemm_provider="cutlass",
                gemm_kernel="library_default",
            ).validate_problem(self.problem)
        with self.assertRaisesRegex(ValueError, "fused NCCL Device API"):
            Target(
                "bad-split-k-communication",
                topology(),
                arch="sm_80",
                gemm_provider="cutlass",
                gemm_kernel="sm80_64x128x32_splitk8",
            ).validate_problem(replace(self.problem, k=256))
        with self.assertRaisesRegex(ValueError, "K divisible by 256"):
            Target(
                "bad-split-k-alignment",
                topology(),
                arch="sm_80",
                gemm_provider="cutlass",
                gemm_kernel="sm80_64x128x32_splitk8",
                communication_api="nccl_device",
            ).validate_problem(self.problem)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is unavailable")
    def test_generated_cpp_header_links_to_and_passes_runtime_validation(self) -> None:
        result = self.compile_with(CppManifestBackend())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result.artifact.write_to(root)
            smoke = root / "smoke.cpp"
            executable = root / "smoke"
            smoke.write_text(
                '#include "edmir_program.hpp"\n'
                '#include "edmir/runtime/validate.hpp"\n'
                "int main() {\n"
                "  return edmir::runtime::validate_program(edmir_generated::program()).valid() "
                "? 0 : 1;\n"
                "}\n",
                encoding="utf-8",
            )
            runtime_root = Path(__file__).resolve().parents[1] / "runtime"
            completed = subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-I",
                    str(runtime_root / "include"),
                    str(smoke),
                    str(runtime_root / "src" / "validate.cpp"),
                    "-o",
                    str(executable),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            executed = subprocess.run(
                [str(executable)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_pipeline_rejects_target_world_size_mismatch(self) -> None:
        wrong_target = Target("wrong", topology(2), arch="sm_89")
        with self.assertRaisesRegex(ValueError, "world size"):
            CompilerPipeline().compile(
                self.module,
                target=wrong_target,
                profile=profile(),
                backend=JsonBackend(),
            )

    def test_pipeline_rejects_profile_provider_or_kernel_mismatch(self) -> None:
        cutlass = Target(
            "test-cutlass", topology(), arch="sm_80", gemm_provider="cutlass"
        )
        with self.assertRaisesRegex(ValueError, "provider does not match"):
            CompilerPipeline().compile(
                self.module,
                target=cutlass,
                profile=profile(),
                backend=JsonBackend(),
            )
        explicit = replace(cutlass, gemm_kernel="sm80_64x128x32")
        with self.assertRaisesRegex(ValueError, "kernel does not match"):
            CompilerPipeline().compile(
                self.module,
                target=explicit,
                profile=replace(
                    self.cutlass_profile(), gemm_kernel="sm80_128x128x32"
                ),
                backend=JsonBackend(),
            )


if __name__ == "__main__":
    unittest.main()
