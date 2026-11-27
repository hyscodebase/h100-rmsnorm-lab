"""One correctness-gated BF16 RMSNorm neighborhood checkpoint on CUDA."""
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
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from kernels import rmsnorm

ROWS = 1024
WIDTHS = (4103, 4104, 4105, 4112)
SEEDS = (44, 45, 46)
EPS = 1e-6
STRICT_RTOL = 5e-3
STRICT_ATOL = 1e-6
BENCH_RTOL = 0.016
BENCH_ATOL = 1e-5
ROUNDS = 21
OPS = 32
SOURCES = ("kernels.py", "experiments/width-neighborhood/run.py", "experiments/width-neighborhood/PLAN.md", "plans/width-neighborhood.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def driver_version():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            text=True, capture_output=True, check=False)
    return next(iter(result.stdout.splitlines()), None)


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def eager(x, weight):
    return F.rms_norm(x, (x.shape[-1],), weight, eps=EPS)


def reference(x, weight):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()


def make_fixtures(width, seed, device):
    g = torch.Generator(device=device).manual_seed(seed ^ (width << 12))
    base = torch.randn((ROWS, width), dtype=torch.float32, device=device, generator=g)
    weight = torch.randn((width,), dtype=torch.float32, device=device, generator=g).to(torch.bfloat16)
    fixtures = []
    patterns = {
        "random": base,
        "zero_rows": base.clone(),
        "constant": torch.full_like(base, -0.625),
        "tiny": base * 1e-5,
        "large": base * 1024,
        "alternating": base * torch.where(torch.arange(ROWS, device=device)[:, None] % 2 == 0, 1.0, -1.0),
        "column_ramp": torch.linspace(-1.0, 1.0, width, device=device)[None, :].expand(ROWS, -1).contiguous(),
        "outlier": base.clone(),
    }
    patterns["zero_rows"][::2].zero_()
    patterns["outlier"][:, width // 2] *= 1e4
    for name, tensor in patterns.items():
        fixtures.append((name, tensor.to(torch.bfloat16).contiguous(), weight))
    return fixtures


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
    return graph


@torch.inference_mode()
def run(seed, width):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError("The registered benchmark requires one BF16-capable CUDA GPU.")
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    started = utc_now()
    clock = time.perf_counter()
    checks = []
    timing = []
    torch.compiler.reset()
    compiled = torch.compile(eager, fullgraph=True, dynamic=False)
    for fixture_name, x, weight in make_fixtures(width, seed, device):
        ref = reference(x, weight)
        for name, call in (
            ("triton_4w_1r", partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=1)),
            ("triton_8w_1r", partial(rmsnorm, x, weight, eps=EPS, num_warps=8, rows_per_program=1)),
            ("triton_4w_2r", partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=2)),
            ("torch_compile", None),
        ):
            if name == "torch_compile":
                call = partial(compiled, x, weight)
            actual = call().float()
            torch.cuda.synchronize()
            error = (actual - ref).abs()
            strict_passed = bool(torch.isclose(actual, ref, rtol=STRICT_RTOL, atol=STRICT_ATOL).all().item())
            benchmark_passed = bool(torch.isclose(actual, ref, rtol=BENCH_RTOL, atol=BENCH_ATOL).all().item())
            checks.append({"fixture": fixture_name, "backend": name,
                           "strict_passed": strict_passed, "strict_max_abs_error": float(error.max().item()),
                           "benchmark_passed": benchmark_passed,
                           "benchmark_max_abs_error": float(error.max().item())})
            if not strict_passed or not benchmark_passed:
                raise RuntimeError(f"Correctness failed for {fixture_name}/{name}.")
    # Use one common input set for timing; the fingerprints prove every backend saw the same tensors.
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((ROWS, width), device=device, dtype=torch.bfloat16, generator=g)
    weight = torch.randn((width,), device=device, dtype=torch.bfloat16, generator=g)
    calls = {
        "triton_4w_1r": partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=1),
        "triton_8w_1r": partial(rmsnorm, x, weight, eps=EPS, num_warps=8, rows_per_program=1),
        "triton_4w_2r": partial(rmsnorm, x, weight, eps=EPS, num_warps=4, rows_per_program=2),
        "torch_compile": partial(compiled, x, weight),
    }
    timing_correctness = []
    for name, call in calls.items():
        actual = call().float()
        ref = reference(x, weight)
        torch.cuda.synchronize()
        error = float((actual - ref).abs().max().item())
        passed = bool(torch.isclose(actual, ref, rtol=STRICT_RTOL, atol=STRICT_ATOL).all().item())
        timing_correctness.append({"backend": name, "strict_passed": passed, "strict_max_abs_error": error})
        if not passed:
            raise RuntimeError(f"Strict correctness failed on timed input for {name}.")
    graphs = {name: capture(call) for name, call in calls.items()}
    names = list(graphs)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); end.synchronize()
    rng = random.Random(seed ^ width)
    for round_id in range(ROUNDS):
        order = names.copy()
        rng.shuffle(order)
        for order_index, name in enumerate(order):
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            elapsed = float(start.elapsed_time(end) * 1000.0)
            timing.append({"round": round_id, "order_index": order_index, "backend": name,
                           "graph_elapsed_us": elapsed, "ops_per_graph": OPS,
                           "us_per_op": elapsed / OPS})
    del graphs
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(device)
    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
    return {
        "seed": seed, "rows": ROWS, "width": width, "code_revision": os.environ["BENCH_REVISION"],
        "source_sha256": sources, "started_utc": started, "finished_utc": utc_now(),
        "wall_seconds": time.perf_counter() - clock,
        "environment": {"gpu": torch.cuda.get_device_name(device),
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                        "total_memory_bytes": int(properties.total_memory), "sm_count": int(properties.multi_processor_count),
                        "driver": driver_version(), "cuda_runtime": torch.version.cuda,
                        "torch": str(torch.__version__), "triton": triton.__version__,
                        "python": platform.python_version()},
        "method": {"dtype": "bfloat16", "accumulation": "float32", "eps": EPS,
                   "correctness_rtol": 0.016, "correctness_atol": 1e-5,
                   "timing": "CUDA events around one graph replay, 32 calls per replay",
                   "cache": "same input addresses; no explicit L2 flush", "rounds": ROUNDS,
                   "ops_per_graph": OPS, "backend_order_shuffled_each_round": True,
                   "timed_backends": list(calls)},
        "input_sha256": digest(x), "weight_sha256": digest(weight),
        "torch_cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "torch_cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "correctness": {"strict_gate": {"rtol": STRICT_RTOL, "atol": STRICT_ATOL,
                                          "passed": all(item["strict_passed"] for item in checks)},
                        "benchmark_crosscheck": {"rtol": BENCH_RTOL, "atol": BENCH_ATOL,
                                                  "passed": all(item["benchmark_passed"] for item in checks)},
                        "fixtures": checks, "timing_input": timing_correctness},
        "timing_intervals": timing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.width), indent=2, allow_nan=False))
