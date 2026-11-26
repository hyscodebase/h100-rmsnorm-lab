"""Checkpoint contract for one sustained BF16 prefill projection per shape and seed."""
import math
import statistics
from itertools import product

SOURCE = "experiments/prefill-gemm/benchmark.py"
TELEMETRY = ("gpu_busy_pct", "memory_busy_pct", "device_memory_used_mib", "device_power_w", "sm_clock_mhz")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_plan(plan):
    shapes = plan.get("shapes")
    require(isinstance(shapes, list) and shapes, "GEMM requires a nonempty shapes array.")
    keys = ("m", "k", "n")
    dims = []
    for shape in shapes:
        require(isinstance(shape, dict) and set(shape) == set(keys)
                and all(type(shape[k]) is int and shape[k] > 0 for k in keys),
                "Each shape must specify positive integers m, k and n.")
        dims.append(tuple(shape[k] for k in keys))
    require(len(set(dims)) == len(dims), "GEMM shapes must be distinct.")
    require(finite(plan.get("seconds")) and plan["seconds"] >= 5, "GEMM seconds must be finite and at least 5.")
    require(max(plan["seeds"]) < 2**63, "GEMM seeds must be smaller than 2**63.")


def tasks(plan):
    for seed, shape in product(plan["seeds"], plan["shapes"]):
        m, k, n = (shape[key] for key in ("m", "k", "n"))
        yield f"s{seed}-m{m}-k{k}-n{n}", (seed, m, k, n)


def arguments(shape, plan):
    seed, m, k, n = shape
    return ["--m", str(m), "--k", str(k), "--n", str(n), "--seconds", str(plan["seconds"]), "--seed", str(seed)]


def validate_telemetry(telemetry, elapsed):
    samples = telemetry["samples"]
    require(isinstance(samples, list) and samples, "Missing GEMM device telemetry samples.")
    for sample in samples:
        start, end = sample["query_start_s"], sample["query_end_s"]
        require(finite(start) and finite(end) and 0 <= start <= end <= elapsed, "Telemetry falls outside the timed window.")
    for field in TELEMETRY:
        values = [s[field] for s in samples if s[field] is not None]
        require(all(finite(v) and v >= 0 for v in values), f"Invalid telemetry sample for {field}.")
        if field.endswith("_pct"):
            require(all(v <= 100 for v in values), f"Invalid telemetry percentage for {field}.")
        if field == "gpu_busy_pct":
            require(values, "Missing GPU utilization evidence.")
        stat = telemetry["summary"][field]
        require(stat["count"] == len(values), f"Incorrect telemetry count for {field}.")
        if values:
            require(stat["peak_observed"] == max(values) and math.isclose(stat["mean"], statistics.fmean(values)),
                    f"Telemetry statistics differ from raw samples for {field}.")
        else:
            require(stat["peak_observed"] is None and stat["mean"] is None, f"Unexpected telemetry summary for {field}.")


def validate_result(result, shape, plan):
    seed, m, k, n = shape
    work = result["workload"]
    require([work[key] for key in ("seed", "m", "k", "n")] == [seed, m, k, n]
            and work["dtype"] == "bfloat16" and work["bf16_reduced_precision_reduction"] is False
            and work["operation"] == "torch.mm(A[M,K], B[K,N], out=C[M,N])"
            and work["reuse"] == "same contiguous A, B and preallocated C throughout"
            and work["operand_bytes"] == 2 * (m * k + k * n + m * n), "GEMM workload differs from plan.")
    check = result["correctness"]
    require(check["passed"] is True and check["reference"] == "FP32 dense products over full K, TF32 disabled"
            and check["rtol"] == 0.005 and math.isclose(check["atol"], max(1e-5, 2e-5 * math.sqrt(k)))
            and finite(check["max_absolute_error"]) and check["max_absolute_error"] >= 0,
            "Missing or invalid GEMM accuracy evidence.")
    for key, bound in (("sampled_rows", m), ("sampled_columns", n)):
        indices = check[key]
        require(len(set(indices)) == len(indices) == min(16, bound)
                and all(type(i) is int and 0 <= i < bound for i in indices), f"Invalid {key} in accuracy check.")
    require(check["entries"] == min(16, m) * min(16, n), "Incorrect count of sampled output entries.")

    timing = result["timing"]
    elapsed = timing["wall_elapsed_s"]
    require(timing["requested_seconds"] == plan["seconds"] and finite(elapsed) and elapsed >= plan["seconds"],
            "GEMM timing did not cover the requested measurement window.")
    require(finite(timing["warmup_s"]) and timing["warmup_s"] >= 2 and type(timing["warmup_iterations"]) is int
            and timing["warmup_iterations"] > 0, "Missing GEMM warmup evidence.")
    batches = timing["batches"]
    require(isinstance(batches, list) and batches and type(timing["batch_iterations"]) is int
            and timing["batch_iterations"] > 0, "Missing timed GEMM batches.")
    for batch in batches:
        require(type(batch["iterations"]) is int and batch["iterations"] == timing["batch_iterations"]
                and finite(batch["event_elapsed_s"]) and batch["event_elapsed_s"] > 0,
                "Invalid raw GEMM batch.")
    iterations = sum(b["iterations"] for b in batches)
    gpu_seconds = sum(b["event_elapsed_s"] for b in batches)
    # Synchronized, sequential event intervals fit inside the host window, up to clock tolerance.
    require(gpu_seconds <= elapsed * 1.01 + 0.002, "Summed CUDA event time exceeds the sequential wall window.")
    flops = 2 * m * k * n
    require(timing["iterations"] == iterations and timing["flops_per_iteration"] == flops
            and math.isclose(timing["cuda_event_total_s"], gpu_seconds)
            and math.isclose(timing["wall_tflops"], iterations * flops / elapsed / 1e12)
            and math.isclose(timing["cuda_event_tflops"], iterations * flops / gpu_seconds / 1e12),
            "GEMM throughput differs from shape and raw timings.")
    validate_telemetry(result["telemetry"], elapsed)
