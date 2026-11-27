"""Run one correctness-gated, throughput-scale RMSNorm shape/seed checkpoint."""
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from kernels import rmsnorm

PLAN_PATH = ROOT / "plans/throughput-neighborhood.json"
PLAN = json.loads(PLAN_PATH.read_text())
ROWS = PLAN["rows"]
WIDTHS = tuple(PLAN["widths"])
SEEDS = tuple(PLAN["seeds"])
EPS = PLAN["eps"]
ROUNDS = PLAN["rounds"]
OPS = PLAN["ops_per_graph"]
STRICT_RTOL = PLAN["strict_correctness"]["rtol"]
STRICT_ATOL = PLAN["strict_correctness"]["atol"]
BENCH_RTOL = PLAN["benchmark_crosscheck"]["rtol"]
BENCH_ATOL = PLAN["benchmark_crosscheck"]["atol"]
BACKEND_NAMES = ("triton_4w_1r", "triton_8w_1r", "triton_4w_2r", "torch_compile")
COMPUTE_SOURCES = ("kernels.py", "experiments/throughput-neighborhood/run.py",
                   "experiments/throughput-neighborhood/PLAN.md", "plans/throughput-neighborhood.json")
GIB = 1024**3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def driver_version():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            text=True, capture_output=True, check=False)
    return next(iter(result.stdout.splitlines()), None)


def sha256_tensor(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def rmsnorm_reference(x, weight):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()


def torch_rmsnorm(x, weight):
    return F.rms_norm(x, (x.shape[-1],), weight, eps=EPS)


def graph_output_estimate(width):
    max_width = max(WIDTHS)
    bytes_per_tensor = ROWS * max_width * PLAN["output_element_bytes"]
    outputs_per_graph = OPS
    backends = len(BACKEND_NAMES)
    pool_bytes = bytes_per_tensor * outputs_per_graph * backends
    if bytes_per_tensor != PLAN["estimated_output_bytes_per_graph_op"]:
        raise ValueError("The frozen maximum-width output-byte estimate changed.")
    if pool_bytes != PLAN["estimated_worst_case_graph_output_pool_bytes"]:
        raise ValueError("The frozen worst-case graph-output-pool estimate changed.")
    return {"width_under_test": width, "max_width_for_estimate": max_width,
            "one_output_tensor_bytes_at_max_width": bytes_per_tensor,
            "ops_per_graph": outputs_per_graph, "backend_graph_count": backends,
            "worst_case_graph_output_pool_bytes": pool_bytes,
            "worst_case_graph_output_pool_gib": pool_bytes / GIB}


def device_memory_snapshot(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "free_bytes": int(free), "total_bytes": int(total),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def make_fixture(name, width, seed, weight, device):
    g = torch.Generator(device=device).manual_seed(seed ^ (width << 12) ^ FIXTURE_SEED_OFFSET[name])
    shape = (ROWS, width)
    if name in ("random", "zero_rows", "tiny", "large", "alternating", "outlier"):
        x = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g)
        if name == "zero_rows":
            x[::2].zero_()
        elif name == "tiny":
            x.mul_(1e-5)
        elif name == "large":
            x.mul_(1024)
        elif name == "alternating":
            sign = torch.where(torch.arange(ROWS, device=device)[:, None] % 2 == 0, 1.0, -1.0)
            x.mul_(sign)
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


FIXTURES = ("random", "zero_rows", "constant", "tiny", "large", "alternating", "column_ramp", "outlier")
FIXTURE_SEED_OFFSET = {name: index * 1_000_003 for index, name in enumerate(FIXTURES)}


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(OPS):
            call()
    stream.synchronize()
    return graph


def sha_sources():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in COMPUTE_SOURCES}


@torch.inference_mode()
def run(seed, width):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError("The registered benchmark requires one BF16-capable CUDA GPU.")
    estimate = graph_output_estimate(width)
    device = torch.device("cuda", 0)
    started = utc_now()
    wall_start = time.perf_counter()
    checks = []
    intervals = []
    torch.compiler.reset()
    compiled = torch.compile(torch_rmsnorm, fullgraph=True, dynamic=False)
    # One shared small weight, while fixture activations are generated and released one at a time.
    weight_gen = torch.Generator(device=device).manual_seed(seed ^ width)
    weight = torch.randn((width,), device=device, dtype=torch.bfloat16, generator=weight_gen)
    for fixture in FIXTURES:
        x, weight = make_fixture(fixture, width, seed, weight, device)
        reference = rmsnorm_reference(x, weight)
        calls = (
            ("triton_4w_1r", partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=1)),
            ("triton_8w_1r", partial(rmsnorm, x, weight, eps=EPS, num_warps=8, rows_per_program=1)),
            ("triton_4w_2r", partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=2)),
            ("torch_compile", partial(compiled, x, weight)),
        )
        for name, call in calls:
            actual = call().float()
            torch.cuda.synchronize(device)
            error = float((actual - reference).abs().max().item())
            strict = bool(torch.isclose(actual, reference, rtol=STRICT_RTOL, atol=STRICT_ATOL).all().item())
            cross = bool(torch.isclose(actual, reference, rtol=BENCH_RTOL, atol=BENCH_ATOL).all().item())
            checks.append({"fixture": fixture, "backend": name,
                           "strict_passed": strict, "strict_max_abs_error": error,
                           "crosscheck_passed": cross, "crosscheck_max_abs_error": error})
            if not strict or not cross:
                raise RuntimeError(f"Correctness failed before timing for {fixture}/{name}.")
        del x, reference, calls

    # The timed input is freshly generated; every backend uses these same addresses and values.
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((ROWS, width), device=device, dtype=torch.bfloat16, generator=g)
    weight = torch.randn((width,), device=device, dtype=torch.bfloat16, generator=g)
    timed_calls = {
        "triton_4w_1r": partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=1),
        "triton_8w_1r": partial(rmsnorm, x, weight, eps=EPS, num_warps=8, rows_per_program=1),
        "triton_4w_2r": partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=2),
        "torch_compile": partial(compiled, x, weight),
    }
    timing_checks = []
    ref = rmsnorm_reference(x, weight)
    for name, call in timed_calls.items():
        output = call().float()
        torch.cuda.synchronize(device)
        error = float((output - ref).abs().max().item())
        strict = bool(torch.isclose(output, ref, rtol=STRICT_RTOL, atol=STRICT_ATOL).all().item())
        cross = bool(torch.isclose(output, ref, rtol=BENCH_RTOL, atol=BENCH_ATOL).all().item())
        timing_checks.append({"backend": name, "strict_passed": strict, "strict_max_abs_error": error,
                              "crosscheck_passed": cross, "crosscheck_max_abs_error": error})
        if not strict or not cross:
            raise RuntimeError(f"Correctness failed on timed input for {name}.")
    del ref

    correctness_peak = device_memory_snapshot(device)
    free, total = torch.cuda.mem_get_info(device)
    if int(free) < PLAN["free_memory_guard_bytes"]:
        raise RuntimeError(f"Pre-capture free-memory guard failed: {int(free)} bytes free.")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    pre_capture = device_memory_snapshot(device)
    reserved_baseline = pre_capture["reserved_bytes"]
    graphs = {}
    capture_snapshots = []
    for name, call in timed_calls.items():
        graphs[name] = capture(call)
        torch.cuda.synchronize(device)
        snapshot = device_memory_snapshot(device)
        delta = max(0, snapshot["peak_reserved_bytes"] - reserved_baseline)
        snapshot.update(backend=name, capture_reserved_delta_bytes=delta)
        capture_snapshots.append(snapshot)
        if delta > PLAN["measured_output_pool_ceiling_bytes"]:
            raise RuntimeError(f"Capture-pool ceiling exceeded after {name}: {delta} bytes.")
    max_capture_delta = max(s["capture_reserved_delta_bytes"] for s in capture_snapshots)
    # This conservative allocator-reserved delta includes the captured output pool and any capture-time workspace.
    if max_capture_delta > PLAN["measured_output_pool_ceiling_bytes"]:
        raise RuntimeError(f"Capture-pool ceiling exceeded: {max_capture_delta} bytes.")

    order_rng = random.Random(seed ^ (width << 8))
    names = list(timed_calls)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    for round_id in range(ROUNDS):
        order = names.copy()
        order_rng.shuffle(order)
        for order_index, name in enumerate(order):
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            elapsed = float(start.elapsed_time(end) * 1000.0)
            intervals.append({"round": round_id, "order_index": order_index, "backend": name,
                              "graph_elapsed_us": elapsed, "ops_per_graph": OPS,
                              "us_per_op": elapsed / OPS})
    del graphs
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    final_memory = device_memory_snapshot(device)
    sources = sha_sources()
    return {
        "seed": seed, "rows": ROWS, "width": width,
        "code_revision": os.environ["BENCH_REVISION"],
        "source_sha256": sources, "started_utc": started, "finished_utc": utc_now(),
        "wall_seconds": time.perf_counter() - wall_start,
        "environment": {"gpu": torch.cuda.get_device_name(device),
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                        "total_memory_bytes": int(properties.total_memory),
                        "sm_count": int(properties.multi_processor_count),
                        "driver": driver_version(), "cuda_runtime": torch.version.cuda,
                        "torch": str(torch.__version__), "triton": triton.__version__,
                        "python": platform.python_version()},
        "method": {"dtype": "bfloat16", "accumulation": "float32", "eps": EPS,
                   "strict_rtol": STRICT_RTOL, "strict_atol": STRICT_ATOL,
                   "benchmark_rtol": BENCH_RTOL, "benchmark_atol": BENCH_ATOL,
                   "timing": "CUDA events around one graph replay; 4 calls per captured graph",
                   "cache": "same input addresses; no explicit L2 flush",
                   "rounds": ROUNDS, "ops_per_graph": OPS,
                   "backend_order_shuffled_each_round": True,
                   "timed_backends": list(timed_calls)},
        "input_sha256": sha256_tensor(x), "weight_sha256": sha256_tensor(weight),
        "graph_output_estimate": estimate,
        "memory_preflight": {"minimum_free_bytes": PLAN["free_memory_guard_bytes"],
                              "output_pool_ceiling_bytes": PLAN["measured_output_pool_ceiling_bytes"],
                              "correctness_peak_before_capture": correctness_peak,
                              "pre_capture": pre_capture,
                              "per_backend_capture": capture_snapshots,
                              "capture_reserved_delta_peak_bytes": max_capture_delta,
                              "final": final_memory,
                              "capture_and_timing_peak_allocated_bytes": final_memory["peak_allocated_bytes"],
                              "capture_and_timing_peak_reserved_bytes": final_memory["peak_reserved_bytes"],
                              "full_checkpoint_peak_allocated_bytes": max(
                                  correctness_peak["peak_allocated_bytes"], final_memory["peak_allocated_bytes"]),
                              "full_checkpoint_peak_reserved_bytes": max(
                                  correctness_peak["peak_reserved_bytes"], final_memory["peak_reserved_bytes"])},
        "correctness": {"strict_gate": {"rtol": STRICT_RTOL, "atol": STRICT_ATOL,
                                          "passed": all(c["strict_passed"] for c in checks + timing_checks)},
                        "benchmark_crosscheck": {"rtol": BENCH_RTOL, "atol": BENCH_ATOL,
                                                  "passed": all(c["crosscheck_passed"] for c in checks + timing_checks)},
                        "fixtures": checks, "timed_input": timing_checks},
        "timing_intervals": intervals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.width), indent=2, allow_nan=False))
