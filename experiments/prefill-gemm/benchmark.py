"""Measure sustained torch.mm throughput for one BF16 prefill projection."""

import argparse
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import torch


FIELDS = {
    "utilization.gpu": "gpu_busy_pct",
    "utilization.memory": "memory_busy_pct",
    "memory.used": "device_memory_used_mib",
    "power.draw": "device_power_w",
    "clocks.current.sm": "sm_clock_mhz",
}


def smi(fields, device=None):
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    if device is not None:
        command += ["-i", device]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=5)
    except subprocess.TimeoutExpired:
        raise ValueError("nvidia-smi query timed out.") from None
    if result.returncode:
        raise ValueError(f"nvidia-smi query failed for {fields} (exit {result.returncode}).")
    return result.stdout.strip()


def telemetry_device(props):
    # CUDA ordinals can be remapped by a container or CUDA_VISIBLE_DEVICES.
    # Use the UUID internally; publish only aggregate hardware characteristics.
    visible = smi("uuid").splitlines()
    uuid = str(getattr(props, "uuid", "") or "").lower().removeprefix("gpu-")
    matches = [item for item in visible if item.lower().removeprefix("gpu-") == uuid]
    if matches:
        return matches[0]
    if len(visible) == 1 and not uuid:
        return visible[0]
    raise ValueError("Could not match CUDA device 0 to one nvidia-smi GPU.")


def number_or_none(raw):
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def sample_device(device, start, stop, samples, errors):
    deadline = start
    while not stop.wait(max(0, deadline - time.perf_counter())):
        before = time.perf_counter() - start
        try:
            raw = smi(",".join(FIELDS), device)
            values = raw.split(",")
            if len(values) != len(FIELDS) or "\n" in raw:
                raise ValueError("unexpected nvidia-smi row")
            finished = time.perf_counter() - start
            samples.append({"query_start_s": before, "query_end_s": finished,
                            **dict(zip(FIELDS.values(), map(number_or_none, values)))})
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append({"query_start_s": before, "error_type": type(exc).__name__})
        # Skip any missed intervals rather than issuing a burst of queries.
        deadline = start + math.floor(time.perf_counter() - start) + 1


def telemetry_summary(samples, errors, elapsed):
    in_window = [s for s in samples if 0 <= s["query_start_s"] <= s["query_end_s"] <= elapsed]
    summary = {}
    for field in FIELDS.values():
        values = [s[field] for s in in_window if s[field] is not None]
        summary[field] = {"count": len(values), "mean": statistics.fmean(values) if values else None,
                          "peak_observed": max(values) if values else None}
    if not summary["gpu_busy_pct"]["count"]:
        raise ValueError("No valid GPU utilization sample was captured inside the timed window.")
    return {
        "scope": "Device-wide nvidia-smi readings; mean and maximum over observed in-window samples",
        "requested_period_s": 1, "samples": in_window, "summary": summary,
        "queries_crossing_window_end": len(samples) - len(in_window), "query_errors": errors,
    }


def check_output(a, b, out, seed):
    rng = random.Random(seed)
    rows = sorted(rng.sample(range(a.shape[0]), min(16, a.shape[0])))
    cols = sorted(rng.sample(range(b.shape[1]), min(16, b.shape[1])))
    ri = torch.tensor(rows, device=a.device)
    ci = torch.tensor(cols, device=b.device)
    reference = a.index_select(0, ri).float() @ b.index_select(1, ci).float()
    observed = out[ri[:, None], ci].float()
    # BF16 outputs round to nearest; a small absolute term covers cancellation
    # around zero and differing FP32 accumulation order in the reference.
    rtol = 0.005
    atol = max(1e-5, 2e-5 * math.sqrt(a.shape[1]))
    torch.testing.assert_close(observed, reference, rtol=rtol, atol=atol)
    return {"passed": True, "reference": "FP32 dense products over full K, TF32 disabled",
            "sampled_rows": rows, "sampled_columns": cols, "entries": observed.numel(),
            "rtol": rtol, "atol": atol, "max_absolute_error": (observed - reference).abs().max().item()}


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A CUDA GPU supporting BF16 is required.")
    if not shutil.which("nvidia-smi"):
        raise ValueError("nvidia-smi is required for synchronized device telemetry.")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    props = torch.cuda.get_device_properties(0)
    device = telemetry_device(props)
    driver = smi("driver_version", device)
    m, n, k = args.m, args.n, args.k
    operand_bytes = 2 * (m * k + k * n + m * n)
    reference_bytes = 8 * (min(m, 16) + min(n, 16)) * k
    free_bytes, _ = torch.cuda.mem_get_info()
    if operand_bytes + reference_bytes > 0.8 * free_bytes:
        raise ValueError("Operands and correctness scratch exceed 80% of currently free GPU memory.")
    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda:0", generator=generator)
    b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda:0", generator=generator)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda:0")
    torch.mm(a, b, out=out)
    correctness = check_output(a, b, out, args.seed)
    torch.cuda.synchronize()

    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    warm_start = time.perf_counter()
    warm_iters = 0
    while time.perf_counter() - warm_start < 2:
        start_event.record()
        for _ in range(8):
            torch.mm(a, b, out=out)
        end_event.record()
        end_event.synchronize()
        warm_iters += 8
    warm_s = time.perf_counter() - warm_start
    batch_iters = max(1, min(1024, round(0.2 * warm_iters / warm_s)))

    stop = threading.Event()
    samples, errors, batches = [], [], []
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    sampler = threading.Thread(target=sample_device, args=(device, start, stop, samples, errors))
    sampler.start()
    try:
        while time.perf_counter() - start < args.seconds:
            start_event.record()
            for _ in range(batch_iters):
                torch.mm(a, b, out=out)
            end_event.record()
            end_event.synchronize()
            elapsed_gpu_s = start_event.elapsed_time(end_event) / 1000
            if not math.isfinite(elapsed_gpu_s) or elapsed_gpu_s <= 0:
                raise ValueError("Invalid CUDA event duration.")
            batches.append({"iterations": batch_iters, "event_elapsed_s": elapsed_gpu_s})
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        sampler.join()
    iterations = sum(b["iterations"] for b in batches)
    gpu_s = sum(b["event_elapsed_s"] for b in batches)
    flops_each = 2 * m * n * k
    total_flops = iterations * flops_each
    return {
        "timed_start_utc": started_utc, "code_revision": os.getenv("BENCH_REVISION"),
        "environment": {"gpu": props.name, "compute_capability": [props.major, props.minor],
                        "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
                        "driver": driver, "cuda_runtime": torch.version.cuda, "torch": torch.__version__,
                        "python": platform.python_version()},
        "workload": {"operation": "torch.mm(A[M,K], B[K,N], out=C[M,N])",
                     "dtype": "bfloat16", "m": m, "k": k, "n": n, "seed": args.seed,
                     "operand_bytes": operand_bytes, "bf16_reduced_precision_reduction": False,
                     "reuse": "same contiguous A, B and preallocated C throughout"},
        "correctness": correctness,
        "timing": {"requested_seconds": args.seconds, "warmup_s": warm_s,
                   "warmup_iterations": warm_iters, "batch_iterations": batch_iters,
                   "wall_elapsed_s": elapsed, "cuda_event_total_s": gpu_s,
                   "iterations": iterations, "flops_per_iteration": flops_each,
                   "flops_formula": "2*M*N*K; one multiply-add counts as two operations",
                   "wall_tflops": total_flops / elapsed / 1e12,
                   "cuda_event_tflops": total_flops / gpu_s / 1e12, "batches": batches},
        "telemetry": telemetry_summary(samples, errors, elapsed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8192, help="number of token rows")
    parser.add_argument("--k", type=int, default=4096, help="input hidden width")
    parser.add_argument("--n", type=int, default=11008, help="projection width")
    parser.add_argument("--seconds", type=float, default=35, help="minimum wall time of the measured phase")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.m, args.k, args.n) < 1 or not 0 <= args.seed < 2**63:
        parser.error("matrix dimensions must be positive and seed must be in [0, 2**63).")
    if not math.isfinite(args.seconds) or args.seconds < 5:
        parser.error("--seconds must be finite and at least 5.")
    try:
        print(json.dumps(run(args), indent=2, allow_nan=False))
    except (AssertionError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        sys.exit(f"Benchmark failed: {exc}")


if __name__ == "__main__":
    main()
