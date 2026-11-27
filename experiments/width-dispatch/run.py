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
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kernels import rmsnorm_into
from telemetry import InWindowSampler

PLAN_PATH = ROOT / "plans/width-dispatch.json"
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
    "kernels.py", "experiments/width-dispatch/run.py",
    "experiments/width-dispatch/telemetry.py",
    "experiments/width-dispatch/PLAN.md", "plans/width-dispatch.json",
)
GIB = 1024**3
FIXTURE_SEED_OFFSET = {name: index * 1_000_003
                       for index, name in enumerate(PLAN["fixtures"])}


class GateFailure(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def utc_epoch_ns(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000)


def sha256_tensor(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def driver_version(record):
    """Read the loaded driver before allocation and retain bounded diagnostics."""
    import re

    probes = record["driver_probes"] = []
    def capture_output(probe, stdout, stderr):
        for name, data in (("stdout", stdout or b""), ("stderr", stderr or b"")):
            probe[f"{name}_bytes"] = len(data)
            probe[f"{name}_sha256"] = hashlib.sha256(data).hexdigest()

    smi = {"source": "nvidia-smi"}
    probes.append(smi)
    started = time.monotonic()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                 "--format=csv,noheader"],
                                capture_output=True, check=False, timeout=5)
        rows = [line.strip() for line in result.stdout.decode("ascii", "replace").splitlines()
                if line.strip()]
        capture_output(smi, result.stdout, result.stderr)
        smi.update(exit_code=result.returncode, rows=len(rows))
        if result.returncode:
            smi["status"] = "nonzero_exit"
        elif len(set(rows)) == 1 and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", rows[0]):
            smi.update(status="ok", version=rows[0])
            return rows[0]
        else:
            smi["status"] = "invalid_output"
    except subprocess.TimeoutExpired as error:
        capture_output(smi, error.stdout, error.stderr)
        smi.update(status="timeout", timeout_seconds=5)
    except OSError as error:
        smi.update(status="os_error", errno=error.errno)
    finally:
        smi["elapsed_seconds"] = time.monotonic() - started

    module = {"source": "/sys/module/nvidia/version"}
    probes.append(module)
    started = time.monotonic()
    try:
        value = Path(module["source"]).read_bytes().decode("ascii", "replace").strip()
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", value):
            module.update(status="ok", version=value)
            return value
        module["status"] = "invalid_output"
    except OSError as error:
        module.update(status="os_error", errno=error.errno)
    finally:
        module["elapsed_seconds"] = time.monotonic() - started
    return None


def normalized_device_uuid(value):
    if isinstance(value, (bytes, bytearray)) and len(value) == 16:
        value = str(uuid.UUID(bytes=bytes(value)))
    text = str(value or "").strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    normalized = "".join(character for character in text if character in "0123456789abcdef")
    return normalized if len(normalized) == 32 else None


def resolve_telemetry_device(expected_name, expected_uuid):
    """Resolve the first CUDA-visible GPU to a verified nvidia-smi selector."""
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is unavailable.")
    expected_uuid = normalized_device_uuid(expected_uuid)
    if expected_uuid is None:
        raise RuntimeError("PyTorch did not expose a verifiable CUDA device UUID.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selector = visible.split(",", 1)[0].strip()
        if not selector or selector in {"-1", "NoDevFiles"}:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not identify a usable GPU.")
        mapping = "first CUDA_VISIBLE_DEVICES selector; GPU UUID and name verified"
    else:
        try:
            listed = subprocess.run(["nvidia-smi", "-L"], text=True,
                                    capture_output=True, check=False, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Cannot enumerate nvidia-smi devices ({type(error).__name__}).") from None
        visible_gpus = [line for line in listed.stdout.splitlines() if line.startswith("GPU ")]
        if listed.returncode != 0 or len(visible_gpus) != 1:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is unset and nvidia-smi does not expose exactly one GPU.")
        selector = "0"
        mapping = "single nvidia-smi-visible GPU index; GPU UUID and name verified"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader",
                                 f"--id={selector}"], text=True, capture_output=True,
                                check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot verify nvidia-smi device ({type(error).__name__}).") from None
    rows = [line.strip().split(",", 1) for line in result.stdout.splitlines() if line.strip()]
    if (result.returncode != 0 or len(rows) != 1 or len(rows[0]) != 2
            or normalized_device_uuid(rows[0][0]) != expected_uuid
            or rows[0][1].strip() != expected_name):
        raise RuntimeError("nvidia-smi selector does not resolve to the active CUDA GPU.")
    return selector, mapping


def rmsnorm_reference(x, weight):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()


def torch_rmsnorm_into(x, weight, output, eps):
    xf = x.float()
    output.copy_(xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float())
    return output


def memory_snapshot(device):
    free, total = torch.cuda.mem_get_info(device)
    snapshot = {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "free_bytes": int(free), "total_bytes": int(total),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
    snapshot["captured_monotonic_ns"] = time.perf_counter_ns()
    return snapshot


def make_fixture(name, width, seed, weight, device):
    generator = torch.Generator(device=device).manual_seed(
        seed ^ (width << 12) ^ FIXTURE_SEED_OFFSET[name])
    shape = (ROWS, width)
    if name == "alternating":
        magnitudes = torch.rand(shape, dtype=torch.bfloat16, device=device, generator=generator)
        signs = torch.where(torch.arange(ROWS, device=device)[:, None] % 2 == 0, 1.0, -1.0)
        signs = signs.to(torch.bfloat16)
        x = magnitudes * signs
    elif name in ("random", "zero_rows", "tiny", "large", "outlier"):
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        if name == "zero_rows":
            x[::2].zero_()
        elif name == "tiny":
            x.mul_(1e-5)
        elif name == "large":
            x.mul_(1024)
        elif name == "outlier":
            x[:, width // 2].mul_(1024)
    elif name == "constant":
        x = torch.full(shape, -0.625, dtype=torch.bfloat16, device=device)
    elif name == "column_ramp":
        row = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
        x = row[None, :].expand(ROWS, -1).to(torch.bfloat16).contiguous()
    else:
        raise ValueError(f"Unknown fixture {name}.")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Fixture {name} must preserve the registered BF16 input dtype; got {x.dtype}.")
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
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item())
    strict = finite and bool(torch.isclose(actual, reference, rtol=STRICT_RTOL,
                                           atol=STRICT_ATOL).all().item())
    cross = finite and bool(torch.isclose(actual, reference, rtol=CROSS_RTOL,
                                          atol=CROSS_ATOL).all().item())
    max_error = float((actual - reference).abs().max().item()) if finite else None
    return {"backend": name, "finite": finite, "strict_passed": strict,
            "crosscheck_passed": cross, "max_abs_error": max_error}


def validate_captured_graph(name, graph_kind, graph, output, reference, device):
    output.fill_(float("nan"))
    torch.cuda.synchronize(device)
    graph.replay()
    torch.cuda.synchronize(device)
    check = check_output(name, output, reference)
    check["graph_kind"] = graph_kind
    return check


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
              "attempt_id": os.environ.get("BENCH_ATTEMPT_ID"),
              "run_id": os.environ.get("BENCH_RUN_ID"),
              "source_revision": os.environ.get("BENCH_REVISION"),
              "source_sha256": source_hashes(), "started_utc": started_utc,
              "checkpoints": [], "latency_intervals": [], "sustained_phases": [],
              "correctness": {"fixtures": [], "timed_input": [],
                              "post_latency_replay": [], "post_sustained_replay": [],
                              "graph_write_checks": []},
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
        driver = driver_version(record)
        if not driver:
            raise GateFailure("Cannot establish required NVIDIA driver provenance before measurement.")
        estimate = output_pool_estimate()
        record["graph_memory_estimate"] = estimate
        record["environment"] = {"gpu": torch.cuda.get_device_name(device),
                                  "compute_capability": capability,
                                  "total_memory_bytes": int(props.total_memory),
                                  "sm_count": int(props.multi_processor_count),
                                  "driver": driver, "cuda_runtime": torch.version.cuda,
                                  "torch": str(torch.__version__), "triton": triton.__version__,
                                  "python": platform.python_version()}
        record["memory"]["before_allocations"] = memory_snapshot(device)
        free_before = record["memory"]["before_allocations"]["free_bytes"]
        if free_before < PLAN["free_memory_guard_bytes"]:
            raise GateFailure(f"Free-memory floor failed before allocations: {free_before} bytes.")

        current_stage = "persistent_allocation"
        generator = torch.Generator(device=device).manual_seed(seed ^ width)
        x = torch.randn((ROWS, width), device=device, dtype=torch.bfloat16, generator=generator)
        weight = torch.randn((width,), device=device, dtype=torch.bfloat16, generator=generator)
        reference = rmsnorm_reference(x, weight)
        outputs = {name: torch.empty_like(x) for name in BACKENDS}
        torch.cuda.synchronize(device)
        record["memory"]["after_persistent_allocation"] = memory_snapshot(device)
        free_after = record["memory"]["after_persistent_allocation"]["free_bytes"]
        if free_after < PLAN["free_memory_guard_bytes"]:
            raise GateFailure(f"Free-memory floor failed after persistent allocations: {free_after} bytes.")
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
        timed_reference = reference
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
        record["memory"]["per_backend_capture"] = capture_snapshots
        record["memory"]["capture_reserved_delta_peak_bytes"] = 0
        record["memory"]["capture_reserved_delta_ceiling_bytes"] = PLAN[
            "capture_reserved_delta_ceiling_bytes"]
        for name in BACKENDS:
            latency_graphs[name] = capture_graph(calls[name], OPS)
            sustained_graphs[name] = capture_graph(calls[name], SUSTAINED_CALLS)
            torch.cuda.synchronize(device)
            snap = memory_snapshot(device)
            delta = max(0, snap["peak_reserved_bytes"] - baseline_reserved)
            snap.update(backend=name, capture_reserved_delta_bytes=delta)
            capture_snapshots.append(snap)
            record["memory"]["capture_reserved_delta_peak_bytes"] = max(
                record["memory"]["capture_reserved_delta_peak_bytes"], delta)
            if delta > PLAN["capture_reserved_delta_ceiling_bytes"]:
                raise GateFailure(f"Capture reserved-delta ceiling failed after {name}: {delta} bytes.")

        current_stage = "captured_graph_write_validation"
        for name in BACKENDS:
            for graph_kind, graph in (("latency", latency_graphs[name]),
                                      ("sustained", sustained_graphs[name])):
                check = validate_captured_graph(name, graph_kind, graph, outputs[name],
                                                timed_reference, device)
                record["correctness"]["graph_write_checks"].append(check)
                if not check["strict_passed"] or not check["crosscheck_passed"]:
                    raise GateFailure(f"Captured {graph_kind} graph did not fully write the expected output: {name}.")

        current_stage = "latency_phase"
        record["memory"]["before_latency"] = memory_snapshot(device)
        if record["memory"]["before_latency"]["free_bytes"] < PLAN["free_memory_guard_bytes"]:
            raise GateFailure("Free-memory floor failed immediately before latency measurement: "
                              f"{record['memory']['before_latency']['free_bytes']} bytes.")
        record["latency_start_monotonic_ns"] = time.perf_counter_ns()
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
        telemetry_device_id, telemetry_device_mapping = None, None
        telemetry_device_error = None
        try:
            telemetry_device_id, telemetry_device_mapping = resolve_telemetry_device(
                torch.cuda.get_device_name(device), getattr(props, "uuid", None))
        except Exception as error:
            telemetry_device_error = f"{type(error).__name__}: {error}"
        for name in phase_order:
            current_stage = f"sustained_preroll:{name}"
            # Warm graph replay before the pilot so replay-count calibration reflects sustained clocks.
            preroll_count = 128
            preroll_seconds = event_window(sustained_graphs[name], preroll_count)
            current_stage = f"sustained_calibration:{name}"
            pilot_replays = 128
            pilot_seconds = event_window(sustained_graphs[name], pilot_replays)
            median_block_seconds = pilot_seconds / pilot_replays
            if not math.isfinite(median_block_seconds) or median_block_seconds <= 0:
                raise GateFailure(f"Invalid graph-block calibration for {name}.")
            per_window_replays = math.ceil(PLAN["sustained_target_seconds_per_window"] /
                                           median_block_seconds)
            current_stage = f"sustained_enqueue:{name}"
            memory_before_phase = memory_snapshot(device)
            if memory_before_phase["free_bytes"] < PLAN["free_memory_guard_bytes"]:
                raise GateFailure(f"Free-memory floor failed before sustained phase {name}: "
                                  f"{memory_before_phase['free_bytes']} bytes.")
            sampler = InWindowSampler(PLAN["telemetry_interval_seconds"],
                                      device_id=telemetry_device_id)
            telemetry_process_description = sampler.process_description
            telemetry_start_error = None
            if telemetry_device_error is not None:
                telemetry_start_error = telemetry_device_error
            try:
                if telemetry_start_error is None:
                    sampler.start()
                else:
                    raise RuntimeError(telemetry_start_error)
            except Exception as error:
                # Telemetry failure disqualifies activity claims, not the compute
                # measurement. Stop any partially started poller and continue.
                telemetry_start_error = f"{type(error).__name__}: {error}"
                try:
                    sampler.stop()
                except Exception as cleanup_error:
                    telemetry_start_error += (
                        f"; cleanup {type(cleanup_error).__name__}: {cleanup_error}")
                sampler = None
            window_starts, window_ends = [], []
            host_gaps = []
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
            if sampler is not None:
                try:
                    samples_all = sampler.stop()
                    telemetry_reader_error = sampler.error
                except Exception as error:
                    samples_all = []
                    telemetry_reader_error = f"{type(error).__name__}: {error}"
            else:
                samples_all = []
                telemetry_reader_error = None
            sampler = None
            phase_start_utc_ns = utc_epoch_ns(phase_start_utc)
            phase_end_utc_ns = utc_epoch_ns(phase_end_utc)
            phase_samples = []
            for sample in samples_all:
                query_ns = sample.get("query_unix_ns")
                if isinstance(query_ns, int) and phase_start_utc_ns <= query_ns <= phase_end_utc_ns:
                    phase_samples.append(sample)
                elif (sample.get("error") is not None
                      and phase_start_ns <= sample.get("arrival_monotonic_ns", 0) <= phase_end_ns):
                    phase_samples.append(sample)
            event_durations = [start.elapsed_time(end) / 1000.0
                               for start, end in zip(window_starts, window_ends)]
            event_gaps = [window_ends[index].elapsed_time(window_starts[index + 1]) / 1000.0
                          for index in range(len(window_ends) - 1)]
            phase_seconds = sum(event_durations)
            telemetry_times = [sample.get("query_unix_ns") for sample in phase_samples
                               if sample.get("error") is None
                               and isinstance(sample.get("query_unix_ns"), int)]
            telemetry_query_ordered = all(b > a for a, b in zip(telemetry_times, telemetry_times[1:]))
            telemetry_gaps = [(b - a) / 1e9 for a, b in zip(telemetry_times, telemetry_times[1:])]
            telemetry_start_gap = ((telemetry_times[0] - phase_start_utc_ns) / 1e9
                                   if telemetry_times else None)
            telemetry_end_gap = ((phase_end_utc_ns - telemetry_times[-1]) / 1e9
                                 if telemetry_times else None)
            measured_telemetry_gaps = (telemetry_gaps + [telemetry_start_gap, telemetry_end_gap]
                                       if telemetry_times else [])
            telemetry_max_gap = max(measured_telemetry_gaps, default=None)
            telemetry_valid = (telemetry_start_error is None and telemetry_reader_error is None
                               and all(sample.get("error") is None
                                       and isinstance(sample.get("query_unix_ns"), int)
                                       for sample in phase_samples)
                               and telemetry_query_ordered)
            telemetry_required = PLAN["telemetry_required_fields"]
            telemetry_field_coverage = {
                field: sum(sample.get(field) is not None for sample in phase_samples) / len(phase_samples)
                if phase_samples else 0.0 for field in telemetry_required}
            phase = {"backend": name, "phase_start_utc": phase_start_utc,
                     "phase_end_utc": phase_end_utc,
                     "phase_start_monotonic_ns": phase_start_ns,
                     "phase_end_monotonic_ns": phase_end_ns,
                     "phase_host_wall_seconds": (phase_end_ns - phase_start_ns) / 1e9,
                     "telemetry_process": telemetry_process_description,
                     "telemetry_device_verified": telemetry_device_id is not None,
                     "telemetry_device_mapping": telemetry_device_mapping,
                     "telemetry_start_error": telemetry_start_error,
                     "telemetry_reader_error": telemetry_reader_error,
                     "telemetry_samples": phase_samples,
                     "telemetry_query_interval_seconds": PLAN["telemetry_interval_seconds"],
                     "telemetry_sample_count": len(phase_samples),
                     "telemetry_field_coverage": telemetry_field_coverage,
                     "telemetry_start_gap_seconds": telemetry_start_gap,
                     "telemetry_end_gap_seconds": telemetry_end_gap,
                     "telemetry_max_gap_seconds": telemetry_max_gap,
                     "telemetry_gate_passed": bool(
                         telemetry_valid
                         and len(phase_samples) >= PLAN["minimum_active_telemetry_samples"]
                         and telemetry_max_gap is not None
                         and telemetry_max_gap <= PLAN["maximum_active_telemetry_gap_seconds"]
                         and telemetry_device_id is not None
                         and all(value >= PLAN["telemetry_required_field_coverage_fraction"]
                                 for value in telemetry_field_coverage.values())),
                     "pilot_replays": pilot_replays, "pilot_event_seconds": pilot_seconds,
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
                     "memory_before_phase": memory_before_phase}
            phase["compute_gate_passed"] = bool(
                phase_seconds >= PLAN["sustained_min_seconds"]
                and phase_seconds <= PLAN["sustained_max_seconds"]
                and max(event_durations, default=float("inf")) <= PLAN["maximum_event_window_seconds"]
                and all(math.isfinite(gap) and 0 <= gap
                        <= PLAN["maximum_event_gap_between_windows_seconds"]
                        for gap in event_gaps)
            )
            # Sparse or delayed NVML polls disqualify activity claims but do not stop
            # correctness/performance measurements for the remaining backends.
            phase["activity_telemetry_eligible"] = phase["telemetry_gate_passed"]
            phase["gate_passed"] = phase["compute_gate_passed"]
            # Output validation occurs after all queued windows have completed.
            output_check = check_output(name, outputs[name], timed_reference)
            output_check["phase"] = "post_sustained_replay"
            record["correctness"]["post_sustained_replay"].append(output_check)
            phase["memory_after_phase"] = memory_snapshot(device)
            record["sustained_phases"].append(phase)
            if not output_check["strict_passed"] or not output_check["crosscheck_passed"]:
                raise GateFailure(f"Output verification failed after sustained replay: {name}.")
            if not phase["compute_gate_passed"]:
                raise GateFailure(f"Sustained duration or queue-gap gate failed: {name}.")

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
