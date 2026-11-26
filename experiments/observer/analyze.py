"""Validate raw observer checkpoints and apply the preregistered paired decision."""

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

ORDER = dict(zip(range(51, 57), ("ABC", "BCA", "CAB", "CBA", "BAC", "ACB")))
READERS = {"A": {"cli"}, "B": {"nvml"}, "C": {"cli", "nvml"}}
METRICS = {"gpu_busy_pct", "memory_busy_pct", "device_memory_used_mib", "device_power_w", "sm_clock_mhz"}
STATES = {"valid", "skipped", "error", "timeout", "crossing"}
QUERY_DEADLINE_S = 5.0
TIMESTAMP_TOLERANCE_S = 1e-6
M, K, N = 8192, 4096, 11008
GEMM_FLOPS = 2 * M * K * N
ENV_KEYS = ("gpu", "compute_capability", "total_memory_bytes", "sm_count", "driver", "cuda_runtime", "torch", "python")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(n):
    return type(n) in (int, float) and math.isfinite(n)


def utc(stamp):
    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    need(moment.utcoffset() == timedelta(0), "Timed start must be UTC.")
    return moment


def validate_slots(slots, elapsed):
    need(isinstance(slots, list) and len(slots) == 45, "Each observer must account for 45 scheduled slots.")
    need(all(type(s["slot"]) is int for s in slots) and {s["slot"] for s in slots} == set(range(45)),
         "Missing, invalid or duplicate slot IDs.")
    previous_end = 0.0
    for slot in sorted(slots, key=lambda s: s["slot"]):
        need(set(slot) == {"slot", "status", "query_start_s", "query_end_s", "values"}, "Unexpected slot schema.")
        status, values = slot["status"], slot["values"]
        need(status in STATES and isinstance(values, dict) and set(values) == METRICS, "Invalid slot status or metrics.")
        for field, v in values.items():
            need(v is None or (finite(v) and v >= 0 and (not field.endswith("_pct") or v <= 100)),
                 f"Invalid value in {field}.")
        start, end = slot["query_start_s"], slot["query_end_s"]
        if status == "skipped":
            need(start is None and end is None and all(v is None for v in values.values()),
                 "A skipped slot cannot have read offsets or measurements.")
            continue
        need(finite(start) and finite(end) and start >= 0 and start <= end, "Invalid query offsets.")
        need(start + TIMESTAMP_TOLERANCE_S >= slot["slot"], "Query started before its scheduled slot.")
        need(start + TIMESTAMP_TOLERANCE_S >= previous_end, "Queries from the same reader overlap.")
        previous_end = end
        if status == "crossing":
            need(end > elapsed, "A crossing read must end beyond the measurement window.")
        else:
            need(end <= elapsed, "A read ending after the window must be marked crossing.")
        if status == "valid":
            need(end - start <= QUERY_DEADLINE_S + TIMESTAMP_TOLERANCE_S,
                 "A valid query exceeded the five-second deadline.")
            need(values["gpu_busy_pct"] is not None, "Valid activity reading has no GPU activity.")


def validate(result, seed, arm):
    """Raise ValueError if a checkpoint cannot participate in the experiment."""
    try:
        need(seed in ORDER and arm in READERS and result["seed"] == seed and result["arm"] == arm,
             "Checkpoint seed/arm mismatch.")
        utc(result["timed_start_utc"])
        need(re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", result["code_revision"]), "Invalid code revision.")
        env = result["environment"]
        need(all(isinstance(env[key], str) and env[key].strip() for key in ("driver", "cuda_runtime", "torch", "python", "gpu"))
             and "H100" in env["gpu"] and env["compute_capability"] == [9, 0]
             and all(type(env[key]) is int and env[key] > 0 for key in ("total_memory_bytes", "sm_count")),
             "Expected the recorded H100 environment.")
        versions = env["observer_versions"]
        need(isinstance(versions, dict) and set(versions) == {"binding_version", "nvml_version", "driver_version", "nvidia_smi_version"}
             and all(v is None or (isinstance(v, str) and v.strip()) for v in versions.values()),
             "Invalid observer versions.")
        work = result["workload"]
        need((work["seed"], work["m"], work["k"], work["n"]) == (seed, M, K, N)
             and work["dtype"] == "bfloat16" and work["bf16_reduced_precision_reduction"] is False
             and work["operation"] == "torch.mm(A[M,K], B[K,N], out=C[M,N])"
             and work["reuse"] == "same contiguous A, B and preallocated C throughout"
             and work["operand_bytes"] == 2 * (M*K + K*N + M*N), "Unexpected GEMM workload.")
        check = result["correctness"]
        need(check["passed"] is True and check["reference"] == "FP32 dense products over full K, TF32 disabled"
             and check["entries"] == 256 and check["rtol"] == 0.005 and math.isclose(check["atol"], 2e-5 * math.sqrt(K))
             and finite(check["max_absolute_error"]) and check["max_absolute_error"] >= 0, "Invalid FP32 check.")
        for key, bound in (("sampled_rows", M), ("sampled_columns", N)):
            indices = check[key]
            need(len(indices) == len(set(indices)) == 16 and all(type(i) is int and 0 <= i < bound for i in indices),
                 f"Invalid {key}.")
        t = result["timing"]
        elapsed = t["wall_elapsed_s"]
        need(t["requested_seconds"] == 45 and finite(elapsed) and 45 <= elapsed <= 90 and finite(t["warmup_s"])
             and t["warmup_s"] >= 2 and type(t["warmup_iterations"]) is int and t["warmup_iterations"] > 0,
             "Invalid warmup or measurement window.")
        batches = t["batches"]
        need(isinstance(batches, list) and batches and type(t["batch_iterations"]) is int and t["batch_iterations"] > 0,
             "Missing batch timing evidence.")
        for b in batches:
            need(type(b["iterations"]) is int and b["iterations"] == t["batch_iterations"]
                 and finite(b["event_elapsed_s"]) and b["event_elapsed_s"] > 0, "Invalid batch.")
        count = sum(b["iterations"] for b in batches)
        gpu_time = sum(b["event_elapsed_s"] for b in batches)
        need(t["flops_per_iteration"] == GEMM_FLOPS and t["iterations"] == count
             and math.isclose(t["cuda_event_total_s"], gpu_time)
             and math.isclose(t["wall_tflops"], count * GEMM_FLOPS / elapsed / 1e12)
             and math.isclose(t["cuda_event_tflops"], count * GEMM_FLOPS / gpu_time / 1e12),
             "Throughput or duration differs from the raw batches.")
        need(set(result["observers"]) == READERS[arm], "Incorrect readers for this arm.")
        for slots in result["observers"].values():
            validate_slots(slots, elapsed)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Malformed checkpoint: {exc}") from exc


def readings(slots):
    return {s["slot"]: ((s["query_start_s"] + s["query_end_s"])/2, s["values"]["gpu_busy_pct"])
            for s in slots if s["status"] == "valid"}


def coverage(slots, elapsed):
    valid = readings(slots)
    edges = [0, *sorted(mid for mid, _ in valid.values()), elapsed]
    gap = max(b-a for a, b in zip(edges, edges[1:]))
    return len(valid), gap, len(valid) >= 34 and gap <= 5


def agreement(observers):
    cli, nvml = (readings(observers[k]) for k in ("cli", "nvml"))
    differences = [abs(cli[i][1]-nvml[i][1]) for i in cli.keys() & nvml.keys() if abs(cli[i][0]-nvml[i][0]) <= 0.5]
    matched = len(differences)
    return matched, 45 - matched, statistics.median(differences) if matched >= 20 else None


def load(directory):
    need(directory.is_dir(), "A directory of raw checkpoint JSON files is required.")
    expected = {f"s{seed}-{arm}.json" for seed, arms in ORDER.items() for arm in arms}
    need(all(path.name in expected for path in directory.glob("s*-*.json")), "Unexpected checkpoint filename.")
    records = {}
    anchor = None
    versions = {}
    previous_end = None
    for seed, arms in ORDER.items():
        prior_check = None
        for arm in arms:
            path = directory / f"s{seed}-{arm}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            try:
                validate(r, seed, arm)
                env = r["environment"]
                fingerprint = [r["code_revision"], *(env[k] for k in ENV_KEYS)]
                need(anchor is None or anchor == fingerprint, "Code revision or hardware/software environment changed.")
                anchor = fingerprint
                for k, v in env["observer_versions"].items():
                    if v is not None:
                        # CLI and the NVML C API can format the same installed version differently.
                        key = (arm, k)
                        need(key not in versions or versions[key] == v, f"Observer version changed: {k}.")
                        versions[key] = v
                indices = [r["correctness"][k] for k in ("sampled_rows", "sampled_columns")]
                need(prior_check is None or prior_check == indices, "Within-seed correctness sampling differs by arm.")
                prior_check = indices
                start = utc(r["timed_start_utc"])
                need(previous_end is None or start >= previous_end, "Timed windows overlap or violate registered run order.")
                previous_end = start + timedelta(seconds=r["timing"]["wall_elapsed_s"])
            except ValueError as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
            records[seed, arm] = r
    return records


def decision(records):
    if len(records) != 18:
        return None
    passes = {}
    for arm, reader in (("A", "cli"), ("B", "nvml")):
        passes[arm] = sum(coverage(records[s, arm]["observers"][reader],
                                   records[s, arm]["timing"]["wall_elapsed_s"])[2] for s in ORDER)
    ratios = [records[s, "B"]["timing"]["wall_tflops"] / records[s, "A"]["timing"]["wall_tflops"] for s in ORDER]
    eligible = [difference for s in ORDER if (difference := agreement(records[s, "C"]["observers"])[2]) is not None]
    gates = (passes["B"] >= 5 and passes["B"] - passes["A"] >= 2,
             statistics.median(ratios) >= 0.98, len(eligible) >= 4 and all(d <= 5 for d in eligible))
    return passes, ratios, eligible, gates


def show(records):
    print(f"Completed checkpoints: {len(records)}/18. Times are in seconds.\n")
    print("| Seed | Arm | Reader | Activity samples | Max gap | Coverage | Skipped | Errors | Timeouts | Crossing | Wall TFLOP/s |")
    print("|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for (seed, arm), r in records.items():
        t = r["timing"]
        for reader in sorted(r["observers"]):
            slots = r["observers"][reader]
            n, gap, passed = coverage(slots, t["wall_elapsed_s"])
            counts = [sum(s["status"] == status for s in slots) for status in ("skipped", "error", "timeout", "crossing")]
            print(f"| {seed} | {arm} | {reader} | {n} | {gap:.3f} | {'Pass' if passed else 'Fail'} | "
                  + " | ".join(map(str, counts)) + f" | {t['wall_tflops']:.3f} |")
    print("\n| Seed | Standalone B/A wall TFLOP/s ratio | C matched slots | C unmatched slots | Median absolute difference (pp) |")
    print("|---:|---:|---:|---:|---:|")
    for s in ORDER:
        ratio = "pending"
        if (s, "A") in records and (s, "B") in records:
            ratio = f"{records[s, 'B']['timing']['wall_tflops'] / records[s, 'A']['timing']['wall_tflops']:.5f}"
        cells = ["pending"]*3
        if (s, "C") in records:
            matched, unmatched, diff = agreement(records[s, "C"]["observers"])
            cells = [str(matched), str(unmatched), "insufficient pairs" if diff is None else f"{diff:.2f}"]
        print(f"| {s} | {ratio} | " + " | ".join(cells) + " |")
    final = decision(records)
    if final is None:
        print("\nDecision deferred until all 18 planned checkpoints complete.")
        return
    passes, ratios, eligible, gates = final
    verdict = ["pass" if value else "fail" for value in gates]
    print(f"\nCoverage gate {verdict[0]} (B {passes['B']}/6, A {passes['A']}/6). "
          f"Throughput gate {verdict[1]} (median paired B/A {statistics.median(ratios):.5f}). "
          f"Agreement gate {verdict[2]} ({len(eligible)}/6 seeds with at least 20 pairs).")
    print("Adopt B for this host, driver and shape." if all(gates) else "Retain A for this host, driver and shape.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    try:
        show(load(parser.parse_args().directory))
    except (ValueError, OSError) as exc:
        sys.exit(f"Observer analysis failed: {exc}")
