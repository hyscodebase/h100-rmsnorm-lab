"""Run a preregistered H100 observer trial or a read-only sensor preflight."""
import argparse
import ctypes
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from sensors import Observers

BASELINE = Path(__file__).resolve().parents[1] / "prefill-gemm" / "benchmark.py"
spec = importlib.util.spec_from_file_location("prefill_gemm_baseline", BASELINE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

M, K, N = 8192, 4096, 11008
SECONDS = 45
METRICS = tuple(baseline.FIELDS.values())
VERSIONS = ("binding_version", "nvml_version", "driver_version", "nvidia_smi_version")
UUID_PATTERN = re.compile(r"(?:GPU-|MIG-)?[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}", re.I)


def source_revision():
    revision = os.getenv("BENCH_REVISION")
    if revision and not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})(?:-modified)?", revision):
        raise ValueError("BENCH_REVISION must identify the Git version of the numerical sources.")
    return revision


def cuda_driver_uuid():
    """Read visible CUDA ordinal zero when this PyTorch build omits the UUID property."""
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    uuid_bytes = (ctypes.c_ubyte * 16)()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0):
        raise ValueError("Cannot resolve CUDA-visible device zero.")
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    if get_uuid(ctypes.byref(uuid_bytes), device):
        raise ValueError("Cannot read the CUDA device UUID.")
    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def device_identity():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A CUDA H100 with BF16 support is required.")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name or (props.major, props.minor) != (9, 0):
        raise ValueError("The observer experiment requires an NVIDIA H100.")
    raw = getattr(props, "uuid", None) or cuda_driver_uuid()
    if isinstance(raw, bytes):
        raw = str(uuid.UUID(bytes=raw)) if len(raw) == 16 else raw.decode("ascii")
    try:
        cuda_uuid = uuid.UUID(str(raw).lower().removeprefix("gpu-"))
    except ValueError:
        raise ValueError("Invalid physical UUID returned by CUDA.") from None
    matches = [value for value in baseline.smi("uuid").splitlines()
               if value.lower() == f"gpu-{cuda_uuid}"]
    if len(matches) != 1:
        raise ValueError("Could not match the CUDA device UUID to exactly one physical NVIDIA GPU.")
    return props, matches[0]


def environment(props, device, versions):
    return {
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
        "driver": baseline.smi("driver_version", device), "cuda_runtime": torch.version.cuda,
        "torch": str(torch.__version__), "python": platform.python_version(),
        "observer_versions": {key: versions[key] for key in VERSIONS},
    }


def check_preflight(records, elapsed):
    for record in records:
        start, end = record["query_start_s"], record["query_end_s"]
        if record["status"] != "valid" or start is None or end is None or not 0 <= start <= end <= elapsed:
            continue
        values = {key: record["values"].get(key) for key in METRICS}
        if values["gpu_busy_pct"] is None:
            continue
        if all(v is None or (type(v) in (int, float) and math.isfinite(v) and v >= 0
                             and (not k.endswith("_pct") or v <= 100)) for k, v in values.items()):
            return {"passed": True, "values": values}
    return {"passed": False, "values": None}


def preflight(observers):
    start = time.perf_counter()
    observers.start(start)
    time.sleep(2.5)
    end = time.perf_counter()
    records = observers.finish(end)
    checks = {key: check_preflight(records[key], end - start) for key in ("cli", "nvml")}
    return {"preflight": True, "physical_uuid_matched": True, "passed": all(check["passed"] for check in checks.values()),
            "checks": checks}


@torch.inference_mode()
def measure(seed, arm, observers):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    operand_bytes = 2 * (M * K + K * N + M * N)
    reference_bytes = 8 * (min(M, 16) + min(N, 16)) * K
    if operand_bytes + reference_bytes > 0.8 * torch.cuda.mem_get_info()[0]:
        raise ValueError("Operands and correctness scratch exceed 80% of currently free GPU memory.")
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0", generator=generator)
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda:0", generator=generator)
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda:0")
    torch.mm(a, b, out=out)
    correctness = baseline.check_output(a, b, out, seed)
    torch.cuda.synchronize()
    first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))

    def batch(size):
        first.record()
        for _ in range(size):
            torch.mm(a, b, out=out)
        last.record()
        last.synchronize()

    warm_start = time.perf_counter()
    warm_iters = 0
    while time.perf_counter() - warm_start < 2:
        batch(8)
        warm_iters += 8
    warm_s = time.perf_counter() - warm_start
    batch_iters = max(1, min(1024, round(0.2 * warm_iters / warm_s)))

    batches = []
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    observers.start(start)
    while time.perf_counter() - start < SECONDS:
        batch(batch_iters)
        gpu_seconds = first.elapsed_time(last) / 1000
        if not math.isfinite(gpu_seconds) or gpu_seconds <= 0:
            raise ValueError("Invalid duration from CUDA events.")
        batches.append({"iterations": batch_iters, "event_elapsed_s": gpu_seconds})
    end = time.perf_counter()
    readings = observers.finish(end)
    elapsed = end - start
    gpu_s = sum(b["event_elapsed_s"] for b in batches)
    iterations = sum(b["iterations"] for b in batches)
    flops = 2 * M * K * N
    return {
        "seed": seed, "arm": arm, "timed_start_utc": started_utc,
        "workload": {"operation": "torch.mm(A[M,K], B[K,N], out=C[M,N])", "dtype": "bfloat16",
                     "m": M, "k": K, "n": N, "seed": seed, "operand_bytes": operand_bytes,
                     "bf16_reduced_precision_reduction": False, "reuse": "same contiguous A, B and preallocated C throughout"},
        "correctness": correctness,
        "timing": {"requested_seconds": SECONDS, "warmup_s": warm_s, "warmup_iterations": warm_iters,
                   "batch_iterations": batch_iters, "wall_elapsed_s": elapsed, "cuda_event_total_s": gpu_s,
                   "iterations": iterations, "flops_per_iteration": flops,
                   "flops_formula": "2*M*N*K; one multiply-add counts as two operations",
                   "wall_tflops": flops * iterations / elapsed / 1e12,
                   "cuda_event_tflops": flops * iterations / gpu_s / 1e12, "batches": batches},
        "observers": readings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(51, 57))
    parser.add_argument("--arm", choices=("A", "B", "C"))
    parser.add_argument("--preflight", action="store_true", help="Check device mapping and both readers without GEMM.")
    args = parser.parse_args()
    if args.preflight and (args.arm is not None or args.seed is not None):
        parser.error("--preflight takes no seed or arm")
    if not args.preflight and (args.seed is None or args.arm is None):
        parser.error("a measured trial requires --seed and --arm")
    revision = source_revision()
    props, device = device_identity()
    observers = Observers("C" if args.preflight else args.arm, device)
    try:
        info = environment(props, device, observers.prepare())
        result = preflight(observers) if args.preflight else measure(args.seed, args.arm, observers)
    finally:
        observers.close()
    result.update(environment=info, code_revision=revision)
    payload = json.dumps(result, allow_nan=False)
    if UUID_PATTERN.search(payload):
        raise ValueError("Observer output contained a private device identifier.")
    print(payload)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, AttributeError, AssertionError, OSError, subprocess.SubprocessError) as exc:
        sys.exit("Observer trial failed: " + UUID_PATTERN.sub("[device]", str(exc)))
