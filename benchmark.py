"""Benchmark BF16 RMSNorm forward on a single CUDA GPU using graph replay."""

import argparse
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from kernels import rmsnorm

EPS = 1e-6


def eager_rmsnorm(x, weight):
    return F.rms_norm(x, (x.shape[-1],), weight, eps=EPS)


def capture(call, ops):
    # The first call can compile. Warm up on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(ops):
            call()
    return graph


def measure(graphs, ops, rounds, rng):
    times = {name: [] for name in graphs}
    names = list(graphs)
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Allocate the underlying CUDA events outside the first measurement.
    start.record()
    end.record()
    end.synchronize()
    for _ in range(rounds):
        rng.shuffle(names)
        for name in names:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / ops)
    return times


def driver_version():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    return next(iter(result.stdout.splitlines()), None)


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("A CUDA GPU with BF16 support is required.")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    results = []
    for width in args.widths:
        for rows in args.rows:
            x = torch.randn((rows, width), device=device, dtype=torch.bfloat16, generator=generator)
            w = torch.randn(width, device=device, dtype=torch.bfloat16, generator=generator)
            reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS) * w.float()
            calls = {"torch_eager": lambda: eager_rmsnorm(x, w)}
            for tile in args.row_tiles:
                suffix = "" if tile == 1 else f"_{tile}r"
                for warps in (4, 8):
                    calls[f"triton_{warps}w{suffix}"] = partial(
                        rmsnorm, x, w, num_warps=warps, rows_per_program=tile
                    )
            if args.compiled:
                # Specialize each shape, matching the static Triton launch.
                torch.compiler.reset()
                compiled = torch.compile(eager_rmsnorm, fullgraph=True, dynamic=False)
                calls["torch_compile"] = lambda: compiled(x, w)
            for call in calls.values():
                torch.testing.assert_close(call().float(), reference, rtol=0.016, atol=1e-5)
            graphs = {name: capture(call, args.ops) for name, call in calls.items()}
            samples = measure(graphs, args.ops, args.rounds, rng)
            for name, values in samples.items():
                quartiles = statistics.quantiles(values, n=4, method="inclusive")
                results.append({
                    "rows": rows, "width": width, "backend": name,
                    "median_us": statistics.median(values),
                    "q1_us": quartiles[0], "q3_us": quartiles[2],
                    "samples_us": values,
                })
            del graphs
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("BENCH_REVISION"),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "driver": driver_version(), "cuda_runtime": torch.version.cuda,
            "torch": str(torch.__version__), "triton": triton.__version__,
            "python": platform.python_version(),
        },
        "method": {
            "timing": "CUDA events around one graph replay, microseconds per operation",
            "cache": "same input addresses; no explicit L2 flush",
            "ops_per_graph": args.ops, "rounds": args.rounds, "seed": args.seed,
            "row_tiles": args.row_tiles,
            "dtype": "bfloat16", "eps": EPS, "backends_shuffled_each_round": True,
            "reference": "FP32 RMSNorm, rtol=0.016, atol=1e-5",
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 1024])
    parser.add_argument("--widths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument(
        "--row-tiles", nargs="+", type=int, choices=(1, 2, 4), default=[1],
        help="Triton rows per program to compare. Include 1 for the baseline (default: 1).",
    )
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--ops", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compiled", action="store_true", help="Include shape-specialized TorchInductor")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if 1 not in args.row_tiles or len(set(args.row_tiles)) != len(args.row_tiles):
        parser.error("--row-tiles must include 1 and list each tile only once")
    if min(args.rows) < 1 or min(args.widths) < 1 or max(args.widths) > 16384:
        parser.error("rows must be positive and widths must be between 1 and 16384")
    if args.ops < 1 or args.rounds < 2:
        parser.error("--ops must be positive and --rounds must be at least two")
    result = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
