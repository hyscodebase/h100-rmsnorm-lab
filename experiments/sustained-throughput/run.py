"""Run one frozen 32k-row RMSNorm correctness, latency, and sustained checkpoint."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kernels import rmsnorm_into
from telemetry import InWindowSampler

PLAN_PATH = ROOT / "plans/sustained-throughput.json"
PLAN = json.loads(PLAN_PATH.read_text())
ROWS = PLAN["rows"]
WIDTHS = tuple(PLAN["widths"])
SEEDS = tuple(PLAN["seeds"])
EPS = PLAN["eps"]
OPS = PLAN["ops_per_graph"]
SUSTAINED_CALLS = PLAN["sustained_graph_calls_per_block"]
LATENCY_ROUNDS = PLAN["latency_rounds"]
STRICT_RTOL = PLAN["strict_correctness"]["rtol"]
STRICT_ATOL = PLAN["strict_correctness"]["atol"]
CROSS_RTOL = PLAN["benchmark_crosscheck"]["rtol"]
CROSS_ATOL = PLAN["benchmark_crosscheck"]["atol"]
BACKENDS = ("triton_4w_1r", "triton_8w_1r", "triton_4w_4r", "torch_compile")
COMPUTE_SOURCES = (
    "kernels.py", "experiments/sustained-throughput/run.py",
    "experiments/sustained-throughput/telemetry.py",
    "experiments/sustained-throughput/PLAN.md", "plans/sustained-throughput.json",
)
GIB = 1024**3
FIXTURE_SEED_OFFSET = {name: index * 1_000_003
                       for index, name in enumerate(PLAN["fixtures"])}


class GateFailure(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_tensor(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def driver_version():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                             "--format=csv,noheader", "--id=0"],
                            text=True, capture_output=True, check=False, timeout=5)
    return next(iter(result.stdout.splitlines()), None) if result.returncode == 0 else None


def rmsnorm_reference(x, weight):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()


def torch_rmsnorm_into(x, weight, output, eps):
    xf = x.float()
    output.copy_(xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float())
    return output


def memory_snapshot(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "free_bytes": int(free), "total_bytes": int(total),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def make_fixture(name, width, seed, weight, device):
    generator = torch.Generator(device=device).manual_seed(
        seed ^ (width << 12) ^ FIXTURE_SEED_OFFSET[name])
    shape = (ROWS, width)
    if name in ("random", "zero_rows", "tiny", "large", "alternating", "outlier"):
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        if name == "zero_rows":
            x[::2].zero_()
        elif name == "tiny":
            x.mul_(1e-5)
        elif name == "large":
            x.mul_(1024)
        elif name == "alternating":
            signs = torch.where(torch.arange(ROWS, device=device)[:, None] % 2 == 0, 1.0, -1.0)
            x.mul_(signs)
        elif name == "outlier":
            x[:, width // 2].mul_(1024)
    elif name == "constant":
        x = torch.full(shape, -0.625, dtype=torch.bfloat16, device=device)
    elif name == "column_ramp":
        row = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
        x = row[None, :].expand(ROWS, -1).to(torch.bfloat16).contiguous()
    else:
        raise ValueError(f"Unknown fixture {name}.")
    return x, weight


def capture_graph(call, calls_per_graph):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls_per_graph):
            call()
    stream.synchronize()
    return graph


def event_window(graph, replay_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replay_count):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def check_output(name, output, reference):
    actual = output.float()
    strict = bool(torch.isclose(actual, reference, rtol=STRICT_RTOL,
                                atol=STRICT_ATOL).all().item())
    cross = bool(torch.isclose(actual, reference, rtol=CROSS_RTOL,
                               atol=CROSS_ATOL).all().item())
    max_error = float((actual - reference).abs().max().item())
    return {"backend": name, "strict_passed": strict, "crosscheck_passed": cross,
            "max_abs_error": max_error}


def call_factory(name, x, weight, output, compiled):
    if name == "triton_4w_1r":
        return partial(rmsnorm_into, x, weight, output, eps=EPS,
                       num_warps=4, rows_per_program=1)
    if name == "triton_8w_1r":
        return partial(rmsnorm_into, x, weight, output, eps=EPS,
                       num_warps=8, rows_per_program=1)
    if name == "triton_4w_4r":
        return partial(rmsnorm_into, x, weight, output, eps=EPS,
                       num_warps=4, rows_per_program=4)
    return partial(compiled, x, weight, output, EPS)


def output_pool_estimate():
    output_bytes = ROWS * max(WIDTHS) * PLAN["output_element_bytes"]
    outputs_bytes = output_bytes * PLAN["persistent_output_buffers_per_checkpoint"]
    input_bytes = ROWS * max(WIDTHS) * PLAN["input_element_bytes"]
    reference_bytes = ROWS * max(WIDTHS) * PLAN["reference_element_bytes"]
    live_bytes = outputs_bytes + input_bytes + reference_bytes
    graph_bytes = ROWS * max(WIDTHS) * PLAN["input_element_bytes"] * 2 * SUSTAINED_CALLS
    expected = {"one_output_bytes": PLAN["estimated_one_output_bytes"],
                "persistent_outputs_bytes": PLAN["estimated_persistent_outputs_bytes"],
                "input_bytes": PLAN["estimated_input_bytes"],
                "fp32_reference_bytes": PLAN["estimated_fp32_reference_bytes"],
                "live_buffers_bytes": PLAN["estimated_live_buffers_bytes"],
                "sustained_graph_block_logical_bytes": PLAN["estimated_sustained_graph_block_nominal_bytes_at_max_width"]}
    actual = {"one_output_bytes": output_bytes,
              "persistent_outputs_bytes": outputs_bytes, "input_bytes": input_bytes,
              "fp32_reference_bytes": reference_bytes, "live_buffers_bytes": live_bytes,
              "sustained_graph_block_logical_bytes": graph_bytes}
    if actual != expected:
        raise ValueError(f"Frozen persistent-output memory estimate changed: {actual}")
    return {**actual, "live_buffers_gib": live_bytes / GIB,
            "sustained_graph_block_logical_gib": graph_bytes / GIB,
            "persistent_output_buffers_per_checkpoint": len(BACKENDS)}


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in COMPUTE_SOURCES}


@torch.inference_mode()
def run_checkpoint(seed, width):
    started_utc = utc_now()
    wall_start = time.perf_counter()
    record = {"status": "running", "seed": seed, "rows": ROWS, "width": width,
              "source_revision": os.environ.get("BENCH_REVISION"),
              "source_sha256": source_hashes(), "started_utc": started_utc,
              "checkpoints": [], "latency_intervals": [], "sustained_phases": [],
              "correctness": {"fixtures": [], "timed_input": [],
                              "post_latency_replay": [], "post_sustained_replay": []},
              "memory": {}}
    current_stage = "preflight"
    device = None
    sampler = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
            raise GateFailure("Requires exactly one BF16-capable CUDA GPU.")
        device = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(device)
        capability = list(torch.cuda.get_device_capability(device))
        if "H100" not in torch.cuda.get_device_name(device) or capability != [9, 0]:
            raise GateFailure("This registered experiment requires an NVIDIA H100 (CC 9.0).")
        estimate = output_pool_estimate()
        record["graph_memory_estimate"] = estimate
        record["environment"] = {"gpu": torch.cuda.get_device_name(device),
                                  "compute_capability": capability,
                                  "total_memory_bytes": int(props.total_memory),
                                  "sm_count": int(props.multi_processor_count),
                                  "driver": driver_version(), "cuda_runtime": torch.version.cuda,
                                  "torch": str(torch.__version__), "triton": triton.__version__,
                                  "python": platform.python_version()}
        free_before, total = torch.cuda.mem_get_info(device)
        record["memory"]["before_allocations"] = memory_snapshot(device)
        if int(free_before) < PLAN["free_memory_guard_bytes"]:
            raise GateFailure(f"Free-memory floor failed before allocations: {int(free_before)} bytes.")

        current_stage = "persistent_allocation"
        generator = torch.Generator(device=device).manual_seed(seed ^ width)
        x = torch.randn((ROWS, width), device=device, dtype=torch.bfloat16, generator=generator)
        weight = torch.randn((width,), device=device, dtype=torch.bfloat16, generator=generator)
        reference = rmsnorm_reference(x, weight)
        outputs = {name: torch.empty_like(x) for name in BACKENDS}
        torch.cuda.synchronize(device)
        record["memory"]["after_persistent_allocation"] = memory_snapshot(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        if int(free_after) < PLAN["free_memory_guard_bytes"]:
            raise GateFailure(f"Free-memory floor failed after persistent allocations: {int(free_after)} bytes.")
        record["input_sha256"] = sha256_tensor(x)
        record["weight_sha256"] = sha256_tensor(weight)
        output_ptrs = {name: int(out.data_ptr()) for name, out in outputs.items()}
        if len(set(output_ptrs.values())) != len(BACKENDS):
            raise GateFailure("Persistent output buffers are not distinct across backends.")
        if any(pointer in (int(x.data_ptr()), int(weight.data_ptr())) for pointer in output_ptrs.values()):
            raise GateFailure("A persistent output buffer aliases input or weight storage.")
        record["buffer_ownership"] = {"outputs_per_backend": 1, "data_ptr_by_backend": output_ptrs,
                                       "distinct_across_backends": True,
                                       "aliases_input_or_weight": False,
                                       "calls_per_latency_graph": OPS,
                                       "calls_per_sustained_graph_block": SUSTAINED_CALLS}
        torch.compiler.reset()
        compiled = torch.compile(torch_rmsnorm_into, fullgraph=True, dynamic=False)

        current_stage = "correctness_fixtures"
        for fixture in PLAN["fixtures"]:
            fx, fw = make_fixture(fixture, width, seed, weight, device)
            expected = rmsnorm_reference(fx, fw)
            checks = []
            for name in BACKENDS:
                call = call_factory(name, fx, fw, outputs[name], compiled)
                call()
            torch.cuda.synchronize(device)
            for name in BACKENDS:
                check = check_output(name, outputs[name], expected)
                check["fixture"] = fixture
                checks.append(check)
                record["correctness"]["fixtures"].append(check)
            if not all(check["strict_passed"] and check["crosscheck_passed"] for check in checks):
                raise GateFailure(f"Numerical fixture failed: {fixture}.")
            del fx, expected

        current_stage = "timed_input_correctness"
        timed_reference = rmsnorm_reference(x, weight)
        timed_checks = []
        for name in BACKENDS:
            call = call_factory(name, x, weight, outputs[name], compiled)
            call()
        torch.cuda.synchronize(device)
        for name in BACKENDS:
            check = check_output(name, outputs[name], timed_reference)
            timed_checks.append(check)
            record["correctness"]["timed_input"].append(check)
        if not all(check["strict_passed"] and check["crosscheck_passed"] for check in timed_checks):
            raise GateFailure("Numerical timed-input gate failed before timing.")
        correctness_peak = memory_snapshot(device)
        record["memory"]["correctness_peak"] = correctness_peak
        torch.cuda.reset_peak_memory_stats(device)
        capture_baseline = memory_snapshot(device)
        record["memory"]["capture_baseline"] = capture_baseline

        current_stage = "graph_capture"
        calls = {name: call_factory(name, x, weight, outputs[name], compiled)
                 for name in BACKENDS}
        latency_graphs, sustained_graphs = {}, {}
        capture_snapshots = []
        baseline_reserved = capture_baseline["reserved_bytes"]
        for name in BACKENDS:
            latency_graphs[name] = capture_graph(calls[name], OPS)
            sustained_graphs[name] = capture_graph(calls[name], SUSTAINED_CALLS)
            torch.cuda.synchronize(device)
            snap = memory_snapshot(device)
            delta = max(0, snap["peak_reserved_bytes"] - baseline_reserved)
            snap.update(backend=name, capture_reserved_delta_bytes=delta)
            capture_snapshots.append(snap)
            if delta > PLAN["capture_reserved_delta_ceiling_bytes"]:
                raise GateFailure(f"Capture reserved-delta ceiling failed after {name}: {delta} bytes.")
        record["memory"]["per_backend_capture"] = capture_snapshots
        record["memory"]["capture_reserved_delta_peak_bytes"] = max(
            snap["capture_reserved_delta_bytes"] for snap in capture_snapshots)
        record["memory"]["capture_reserved_delta_ceiling_bytes"] = PLAN[
            "capture_reserved_delta_ceiling_bytes"]

        current_stage = "latency_phase"
        rng = random.Random(seed ^ (width << 8))
        intervals = []
        for round_id in range(LATENCY_ROUNDS):
            order = list(BACKENDS)
            rng.shuffle(order)
            for order_index, name in enumerate(order):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                latency_graphs[name].replay()
                end.record()
                end.synchronize()
                elapsed_us = float(start.elapsed_time(end) * 1000.0)
                intervals.append({"round": round_id, "order_index": order_index,
                                  "backend": name, "graph_elapsed_us": elapsed_us,
                                  "ops_per_graph": OPS, "us_per_op": elapsed_us / OPS})
        record["latency_intervals"] = intervals
        torch.cuda.synchronize(device)
        for name in BACKENDS:
            check = check_output(name, outputs[name], timed_reference)
            record["correctness"]["post_latency_replay"].append(check)
        if not all(check["strict_passed"] and check["crosscheck_passed"]
                   for check in record["correctness"]["post_latency_replay"]):
            raise GateFailure("A backend failed output verification after latency replay.")

        current_stage = "sustained_phase"
        phase_order = list(BACKENDS)
        random.Random(seed ^ (width << 4) ^ 0x6400).shuffle(phase_order)
        for name in phase_order:
            current_stage = f"sustained_calibration:{name}"
            pilot_start = torch.cuda.Event(enable_timing=True)
            pilot_end = torch.cuda.Event(enable_timing=True)
            pilot_start.record()
            for _ in range(128):
                sustained_graphs[name].replay()
            pilot_end.record()
            pilot_end.synchronize()
            pilot_seconds = pilot_start.elapsed_time(pilot_end) / 1000.0
            median_block_seconds = pilot_seconds / 128.0
            if not math.isfinite(median_block_seconds) or median_block_seconds <= 0:
                raise GateFailure(f"Invalid graph-block calibration for {name}.")
            per_window_replays = math.ceil(PLAN["sustained_target_seconds_per_window"] /
                                           median_block_seconds)
            # Warm the device before the measured phase so telemetry is bounded to active work.
            preroll_count = math.ceil(1.25 / median_block_seconds)
            preroll_seconds = event_window(sustained_graphs[name], preroll_count)
            sampler = InWindowSampler(PLAN["telemetry_interval_seconds"])
            sampler.start()
            window_starts, window_ends = [], []
            host_gaps = []
            current_stage = f"sustained_enqueue:{name}"
            phase_start_utc = utc_now()
            phase_start_ns = time.perf_counter_ns()
            previous_enqueue_end = None
            for _window in range(PLAN["sustained_event_windows_per_phase"]):
                if previous_enqueue_end is not None:
                    host_gaps.append((time.perf_counter_ns() - previous_enqueue_end) / 1e9)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(per_window_replays):
                    sustained_graphs[name].replay()
                end.record()
                window_starts.append(start)
                window_ends.append(end)
                previous_enqueue_end = time.perf_counter_ns()
            current_stage = f"sustained_sync:{name}"
            torch.cuda.synchronize(device)
            phase_end_ns = time.perf_counter_ns()
            phase_end_utc = utc_now()
            samples_all = sampler.stop()
            sampler = None
            phase_samples = [sample for sample in samples_all
                             if phase_start_ns <= sample.get("arrival_monotonic_ns", 0) <= phase_end_ns]
            event_durations = [start.elapsed_time(end) / 1000.0
                               for start, end in zip(window_starts, window_ends)]
            event_gaps = [window_ends[index].elapsed_time(window_starts[index + 1]) / 1000.0
                          for index in range(len(window_ends) - 1)]
            phase_seconds = sum(event_durations)
            telemetry_times = [sample.get("arrival_monotonic_ns") for sample in phase_samples
                               if sample.get("error") is None]
            telemetry_gaps = [(b - a) / 1e9 for a, b in zip(telemetry_times, telemetry_times[1:])]
            telemetry_valid = all(sample.get("error") is None for sample in phase_samples)
            telemetry_required = PLAN["telemetry_required_fields"]
            telemetry_field_coverage = {
                field: sum(sample.get(field) is not None for sample in phase_samples) / len(phase_samples)
                if phase_samples else 0.0 for field in telemetry_required}
            phase = {"backend": name, "phase_start_utc": phase_start_utc,
                     "phase_end_utc": phase_end_utc,
                     "phase_start_monotonic_ns": phase_start_ns,
                     "phase_end_monotonic_ns": phase_end_ns,
                     "phase_host_wall_seconds": (phase_end_ns - phase_start_ns) / 1e9,
                     "telemetry_process": "single nvidia-smi --loop=1 process, phase bounded",
                     "telemetry_samples": phase_samples,
                     "telemetry_query_interval_seconds": PLAN["telemetry_interval_seconds"],
                     "telemetry_sample_count": len(phase_samples),
                     "telemetry_field_coverage": telemetry_field_coverage,
                     "telemetry_max_gap_seconds": max(telemetry_gaps, default=None),
                     "telemetry_gate_passed": bool(
                         telemetry_valid
                         and len(phase_samples) >= PLAN["minimum_active_telemetry_samples"]
                         and telemetry_gaps
                         and max(telemetry_gaps) <= PLAN["maximum_active_telemetry_gap_seconds"]
                         and all(value >= PLAN["telemetry_required_field_coverage_fraction"]
                                 for value in telemetry_field_coverage.values())),
                     "pilot_replays": 128, "pilot_event_seconds": pilot_seconds,
                     "pilot_seconds_per_graph_block": median_block_seconds,
                     "unmeasured_preroll_replays": preroll_count,
                     "unmeasured_preroll_event_seconds": preroll_seconds,
                     "graph_calls_per_block": SUSTAINED_CALLS,
                     "replays_per_window": per_window_replays,
                     "window_replay_counts": [per_window_replays] * len(event_durations),
                     "window_event_seconds": event_durations,
                     "window_event_gaps_seconds": event_gaps,
                     "window_host_enqueue_gaps_seconds": host_gaps,
                     "window_count": len(event_durations),
                     "event_seconds": phase_seconds,
                     "logical_input_plus_output_bytes": int(
                         ROWS * width * PLAN["input_element_bytes"] * 2
                         * SUSTAINED_CALLS * sum([per_window_replays] * len(event_durations))),
                     "logical_input_plus_output_GB_per_s": (
                         ROWS * width * PLAN["input_element_bytes"] * 2 * SUSTAINED_CALLS
                         * sum([per_window_replays] * len(event_durations)) / phase_seconds / 1e9),
                     "memory_before_phase": memory_snapshot(device)}
            phase["gate_passed"] = bool(
                phase_seconds >= PLAN["sustained_min_seconds"]
                and phase_seconds <= PLAN["sustained_max_seconds"]
                and max(event_durations, default=float("inf")) <= PLAN["maximum_event_window_seconds"]
                and max(event_gaps, default=float("inf"))
                    <= PLAN["maximum_event_gap_between_windows_seconds"]
                and phase["telemetry_gate_passed"])
            # Output validation occurs after all queued windows have completed.
            output_check = check_output(name, outputs[name], timed_reference)
            output_check["phase"] = "post_sustained_replay"
            record["correctness"]["post_sustained_replay"].append(output_check)
            phase["memory_after_phase"] = memory_snapshot(device)
            record["sustained_phases"].append(phase)
            if not output_check["strict_passed"] or not output_check["crosscheck_passed"]:
                raise GateFailure(f"Output verification failed after sustained replay: {name}.")
            if not phase["gate_passed"]:
                raise GateFailure(f"Sustained duration, queue-gap, or telemetry gate failed: {name}.")

        current_stage = "complete"
        record["status"] = "complete"
    except GateFailure as error:
        record["status"] = "gate_failed"
        record["failure"] = {"stage": current_stage, "type": type(error).__name__,
                              "message": str(error)}
    except Exception as error:  # preserve a failed checkpoint rather than losing partial evidence
        record["status"] = "execution_failed"
        record["failure"] = {"stage": current_stage, "type": type(error).__name__,
                              "message": str(error)}
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception as sampler_error:
                record.setdefault("cleanup_errors", []).append(str(sampler_error))
        if device is not None:
            try:
                torch.cuda.synchronize(device)
                record["memory"]["final"] = memory_snapshot(device)
                record["memory"]["capture_and_timing_peak_allocated_bytes"] = int(
                    record["memory"]["final"]["peak_allocated_bytes"])
                record["memory"]["capture_and_timing_peak_reserved_bytes"] = int(
                    record["memory"]["final"]["peak_reserved_bytes"])
            except Exception as memory_error:
                record.setdefault("cleanup_errors", []).append(str(memory_error))
        record["finished_utc"] = utc_now()
        record["wall_seconds"] = time.perf_counter() - wall_start
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    args = parser.parse_args()
    print(json.dumps(run_checkpoint(args.seed, args.width), indent=2, allow_nan=False))
