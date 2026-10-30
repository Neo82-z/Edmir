"""Conditional SM80 split-K model, with no fitted constants or default policy.

Mainloops are independent; only epilogues form per-output-tile chains. A CTA
keeps its residency slot while waiting. The recurrence is exact for the stated
identical-slot scenario, NOT for CUDA's unspecified placement/admission order.
No end-to-end empirical coefficient is interpreted as an isolated service time.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from .relative_selection import ScheduleCandidate
from .two_stage_selection import parse_schedule_family

MAX_CTAS = 4096
ADMISSION_ORDERS = ("partition_major", "tile_major")


def positive_int(name, value, maximum=None):
    if type(value) is not int or value <= 0 or (maximum and value > maximum):
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class KernelCase:
    environment_id: str
    world_size: int
    m: int
    k: int
    n: int
    q: int
    group_n: int
    precision: str = "fp16_fp32acc_fp16out"

    def validate(self):
        if not isinstance(self.environment_id, str) or not self.environment_id.strip():
            raise ValueError("explicit execution environment required")
        for name in ("world_size", "m", "k", "n", "q", "group_n"):
            positive_int(name, getattr(self, name), 1_048_576)
        if not 2 <= self.world_size <= 8 or self.q not in (1, 2, 4, 8, 16) or self.group_n not in (1, 2):
            raise ValueError("unsupported kernel geometry")
        if self.precision != "fp16_fp32acc_fp16out":
            raise ValueError("this source contract is FP16; BF16 requires a separate implementation")
        if self.k % (32 * self.q) or self.n % 8:
            raise ValueError("aligned nonempty K partitions and N alignment required")
        if self.ctas > MAX_CTAS:
            raise ValueError("bounded local model CTA limit exceeded")

    @classmethod
    def from_candidate(cls, candidate: ScheduleCandidate, *, precision):
        if not isinstance(candidate, ScheduleCandidate):
            raise TypeError("static candidate without timing labels required")
        candidate.validate()
        if candidate.is_host:
            raise ValueError("host needs its own communication and GEMM graph")
        q, g = parse_schedule_family(candidate.family)
        case = cls(candidate.environment_id, candidate.world_size, candidate.rows_per_peer,
                   candidate.k, candidate.n, q, g, precision)
        case.validate()
        return case

    @property
    def chains(self):
        return ((self.m + 63) // 64) * ((self.n + 128*self.group_n - 1) // (128*self.group_n))

    @property
    def ctas(self):
        return self.chains * self.q

    @property
    def depth(self):
        return self.k // (32*self.q)

    def counts(self):
        self.validate()
        tm, tn, tk = 64, 128*self.group_n, 32
        mt, nt = (self.m+tm-1)//tm, (self.n+tn-1)//tn
        return {
            "scope": "one per-source GEMM on one destination GPU",
            "shape_P_M_K_N": [self.world_size, self.m, self.k, self.n],
            "precision": self.precision, "tile": [tm, tn, tk],
            "mainloop_tasks": self.ctas, "epilogue_tasks": self.ctas,
            "independent_output_chains": self.chains,
            "serial_epilogue_edges": self.chains*(self.q-1),
            "serial_mainloop_edges": 0, "gemm_kernel_launches": 1,
            "depth_iterations": self.depth, "warps_per_cta": 4*self.group_n,
            "threads_per_cta": 128*self.group_n,
            "mainloop_operand_storage_payload_bytes": 3*tk*(tm+tn)*2,
            "storage_payload_is_compiled_shared_size": False,
            "logical_accumulator_values_per_thread": tm*tn//(128*self.group_n),
            "accumulator_values_are_compiled_register_count": False,
            "valid_output_fraction": self.m*self.n/(self.chains*tm*tn),
            "logical_A_load_bytes_before_cache_reuse": 2*self.m*self.k*nt,
            "logical_B_load_bytes_before_cache_reuse": 2*self.k*self.n*mt,
            "logical_output_write_bytes": 2*self.m*self.n*self.q,
            "logical_extra_output_read_bytes": 2*self.m*self.n*(self.q-1),
            "physical_traffic_bytes": None,
        }


def resource_envelope(case: KernelCase, *, sm_count=108):
    """Architectural upper bound, not measured/available occupancy.

    SM80: 64 warps, 32 blocks, 164 KiB shared memory/SM; 1 KiB reserved/block.
    The staged operand payload is only a lower bound on the shared union size.
    Registers, carveout, allocation granularity and other kernels may lower R.
    """
    case.validate()
    positive_int("sm_count", sm_count, 1024)
    counts = case.counts()
    shared_floor = counts["mainloop_operand_storage_payload_bytes"] + 1024
    upper = min(32, 64//counts["warps_per_cta"], (164*1024)//shared_floor)
    return {
        "resident_ctas_per_sm_upper_bound": upper,
        "actual_resident_ctas_per_sm": None,
        "isolated_one_cta_per_sm_scenario_slots": sm_count,
        "architectural_slot_upper_bound": sm_count*upper,
        "requires_verified_kernel_launchability": True,
        "missing": ["compiled_kernel_registers", "compiled_dynamic_and_static_shared_bytes",
                    "occupancy_api_result_with_actual_launch_settings", "concurrent_resource_use"],
        "source": "https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy",
    }


@dataclass(frozen=True)
class ServiceScenario:
    """Explicit conditional services; never inferred from full-program labels.

    The model accepts arbitrary common time units. Numerical fixtures must use
    a unit such as 'synthetic_units'; none are default latency predictions.
    mainloop includes independent setup before the semaphore wait; epilogue
    includes ordered output update and release. q1 and serial q>1 may have
    different first-epilogue service and must not silently share a calibration.
    Width, locality, contention and precision-dependent services are caller
    inputs, not automatically inferred by the recurrence.
    """
    mainloop: float
    first_epilogue: float
    later_epilogue: float
    unit: str
    assumption: str

    def validate(self):
        for value in (self.mainloop, self.first_epilogue, self.later_epilogue):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1e12:
                raise ValueError("bounded finite positive service required")
        if not isinstance(self.unit, str) or not self.unit.strip() or not isinstance(self.assumption, str) or not self.assumption.strip():
            raise ValueError("explicit service units and assumptions required")


def conditional_bounds(case: KernelCase, service: ServiceScenario, *, slots):
    """Work/span lower bounds for the identical-slot conditional problem.

    Serial mainloop edges must NOT be added. Waiting consumes slots but is not
    counted as useful service in the work bound; neither bound is a GPU time.
    """
    case.validate(); service.validate(); positive_int("slots", slots, 32768)
    chain = service.first_epilogue + (case.q-1)*service.later_epilogue
    span = service.mainloop + chain
    useful = case.ctas*service.mainloop + case.chains*chain
    return {
        "unit": service.unit, "span": span, "useful_slot_work": useful,
        "work_over_slots": useful/slots,
        "lower_bound": max(span, useful/slots),
        "all_fit": case.ctas <= slots,
        "all_fit_exact_scenario_time": span if case.ctas <= slots else None,
        "whole_program_prediction_us": None,
        "is_hardware_lower_bound": False,
    }


def simulate(case: KernelCase, service: ServiceScenario, *, slots, order="partition_major"):
    """Bounded O(B log(min(B,slots))) recurrence with resident waiting.

    start is the earliest free slot in the specified admission order.
    end[t,s] = max(start[t,s]+mainloop, end[t,s-1]) + epilogue[s].
    Both allowed orders admit each chain's predecessor first, avoiding a
    scheduler-induced deadlock. Real CUDA admission is not asserted to do so.
    """
    bounds = conditional_bounds(case, service, slots=slots)
    if order not in ADMISSION_ORDERS:
        raise ValueError("unsupported admission order")
    if order == "partition_major":
        tasks = ((t, s) for s in range(case.q) for t in range(case.chains))
    else:
        tasks = ((t, s) for t in range(case.chains) for s in range(case.q))
    available = [0.] * min(slots, case.ctas)
    previous = [0.] * case.chains
    wait_sum, completed, makespan = 0., 0, 0.
    for tile, partition in tasks:
        start = heapq.heappop(available)
        ready = start + service.mainloop
        begin_epilogue = max(ready, previous[tile])
        end = begin_epilogue + (service.first_epilogue if partition == 0 else service.later_epilogue)
        wait_sum += begin_epilogue-ready
        previous[tile] = end
        makespan = max(makespan, end)
        heapq.heappush(available, end)
        completed += 1
    if makespan + 1e-8*max(1., makespan) < bounds["lower_bound"]:
        raise AssertionError("conditional work/span bound violated")
    return {**bounds, "order": order, "scenario_makespan": makespan,
            "resident_wait_slot_time": wait_sum, "completed_ctas": completed,
            "compiler_admission": False, "hardware_scheduling_verified": False,
            "service_assumption": service.assumption}


def unresolved_prediction(case: KernelCase):
    """Fail closed rather than invent microseconds from geometry or old fits."""
    return {"counts": case.counts(), "resources": resource_envelope(case),
            "prediction_us": None, "chosen": None, "compiler_admission": False,
            "required_evidence": [
                "mainloop service versus depth, grid load, CTA width and input locality",
                "first and accumulating epilogue service; exposed semaphore wait",
                "kernel resources and concurrent execution conditions",
                "same-identity host communication/GEMM and end-to-end submission costs"],
            "empirical_coefficients_are_valid_service_inputs": False}
