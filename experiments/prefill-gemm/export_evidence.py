"""Export validated GEMM runs and optional UTC-aligned device measurements as CSV.

The index uses the original checkpoint for every primary coverage and activity decision.
Batch latency percentiles describe CUDA event duration per matrix multiplication in each
batch. Supplemental coverage counts distinct one-second bins with a utilization reading
in the half-open wall window [start, end). An occupied bin is an observation, not a
continuous second of activity. Blank CSV fields denote missing measurements.
Energy integrates adjacent observed power values at most 2.5 seconds apart with the
trapezoid rule. Its paired duration is the time those intervals cover.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from summarize import activity_coverage, read_results

METRICS = ("util_pct", "used_mib", "total_mib", "power_w", "temperature_c", "sm_clock_mhz",
           "memory_clock_mhz", "power_limit_w", "memory_busy_pct")
PRIMARY = ("gpu_busy_pct", "memory_busy_pct", "device_memory_used_mib", "device_power_w", "sm_clock_mhz")


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("A measurement timestamp has no UTC offset.")
    return stamp.astimezone(timezone.utc)


def iso(stamp):
    return stamp.isoformat(timespec="microseconds")


def merge_values(samples, stamp, clean):
    existing = samples.setdefault(stamp, {})
    if any(field in existing and existing[field] != v for field, v in clean.items()):
        raise ValueError(f"Conflicting measurements at {iso(stamp)}.")
    existing.update(clean)


def merge_sample(samples, item, *, from_csv=False):
    stamp = utc(item["timestamp_utc"])
    clean = {}
    for field in METRICS:
        value = item.get(field)
        if value is None or (from_csv and value == ""):
            continue
        if from_csv:
            value = float(value)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Invalid numerical measurement in {field}.")
        if field.endswith("_pct") and not 0 <= value <= 100:
            raise ValueError(f"Invalid percentage in {field}.")
        clean[field] = float(value)
    if clean or any(field in item for field in METRICS):
        # A timestamped failed reading breaks a power interval even if every metric is null.
        merge_values(samples, stamp, clean)


def load_samples(directory, windows):
    samples = {}
    if directory:
        files = sorted(directory.glob("gemm-*.jsonl"))
        if not files:
            print("No supplementary GEMM logs found; retaining any existing sidecars.", file=sys.stderr)
        for path in files:
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        stamp = utc(item["timestamp_utc"])
                        if any(start <= stamp < end for start, end, _, _ in windows):
                            merge_sample(samples, item)
    return samples


def existing_samples(path, start, end):
    samples = {}
    if path.exists():
        with path.open(newline="") as stream:
            rows = csv.DictReader(stream)
            fields = rows.fieldnames or []
            if "timestamp_utc" not in fields or set(fields) - {"timestamp_utc", *METRICS} or len(fields) != len(set(fields)):
                raise ValueError("Existing supplementary CSV contains unexpected columns.")
            for row in rows:
                if None in row:
                    raise ValueError("Existing supplementary CSV contains extra cells.")
                merge_sample(samples, row, from_csv=True)
        if any(not start <= stamp < end for stamp in samples):
            raise ValueError("Existing sidecar falls outside this checkpoint's window. Use a separate output directory.")
    return samples


def csv_text(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def save(path, content):
    if path.is_file() and path.read_bytes() == content.encode():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", encoding="utf-8", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(content)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def percentile(ordered, fraction):
    index = (len(ordered) - 1) * fraction
    lo = math.floor(index)
    return ordered[lo] + (ordered[math.ceil(index)] - ordered[lo]) * (index - lo)


def observed_energy(readings, elapsed):
    energy_j = duration_s = 0.0
    previous = None
    for stamp, watts in sorted(readings, key=lambda reading: reading[0]):
        if not 0 <= stamp <= elapsed:
            continue
        if watts is None:
            previous = None
            continue
        if not math.isfinite(watts) or watts < 0:
            raise ValueError("Invalid observed power.")
        if previous is not None:
            earlier, earlier_watts = previous
            interval = stamp - earlier
            if 0 < interval <= 2.5:
                duration_s += interval
                energy_j += interval * (earlier_watts + watts) / 2
        previous = (stamp, watts)
    return (energy_j if duration_s else None), duration_s


def index_row(task, result, manifest, raw, start, end, samples):
    checkpoint = json.loads(raw)
    if checkpoint["result"] != result:
        raise ValueError("Checkpoint changed after verification. Retry the export.")
    timing, check, telemetry = (result[k] for k in ("timing", "correctness", "telemetry"))
    covered, required, gap = activity_coverage(result)
    batch_us = sorted(b["event_elapsed_s"] * 1e6 / b["iterations"] for b in timing["batches"])
    work = result["workload"]
    row = {
        "task_id": task, "experiment_hash": manifest["experiment_hash"],
        "benchmark_sha256": manifest["sources"]["experiments/prefill-gemm/benchmark.py"],
        "code_revision": result["code_revision"], "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "timed_start_utc": iso(start), "timed_end_utc": iso(end), "checkpoint_completed_utc": iso(utc(checkpoint["completed_utc"])),
        **{k: work[k] for k in ("seed", "m", "k", "n", "dtype", "operand_bytes")},
        **{k: timing[k] for k in ("requested_seconds", "warmup_s", "warmup_iterations", "wall_elapsed_s", "cuda_event_total_s",
                                  "iterations", "batch_iterations", "flops_per_iteration", "wall_tflops", "cuda_event_tflops")},
        "timed_batches": len(batch_us), "batch_amortized_p10_us": percentile(batch_us, .1),
        "batch_amortized_p50_us": percentile(batch_us, .5), "batch_amortized_p90_us": percentile(batch_us, .9),
        "correctness_passed": check["passed"], "correctness_entries": check["entries"],
        "correctness_sampled_rows": len(check["sampled_rows"]), "correctness_sampled_columns": len(check["sampled_columns"]),
        "correctness_rtol": check["rtol"], "correctness_atol": check["atol"], "correctness_max_abs_error": check["max_absolute_error"],
        "primary_sampling_period_s": telemetry["requested_period_s"], "primary_required_util_samples": required,
        "primary_query_errors": len(telemetry["query_errors"]), "primary_crossing_window_end": telemetry["queries_crossing_window_end"],
        "primary_max_util_gap_s": gap, "primary_coverage_sufficient": covered,
        "primary_seed_meets_activity_rule": covered and telemetry["summary"]["gpu_busy_pct"]["mean"] >= 95,
    }
    for measure in PRIMARY:
        for stat in ("count", "mean", "peak_observed"):
            row[f"primary_{measure}_{stat}"] = telemetry["summary"][measure][stat]
    times = sorted(stamp for stamp, values in samples.items() if "util_pct" in values)
    edges = [start, *times, end]
    row.update(
        supplemental_samples=len(samples),
        supplemental_first_utc=iso(min(samples)) if samples else "",
        supplemental_last_utc=iso(max(samples)) if samples else "",
        supplemental_util_occupied_1s_bins=len({math.floor((stamp - start).total_seconds()) for stamp in times}),
        supplemental_wall_1s_bins=math.ceil(timing["wall_elapsed_s"]),
        supplemental_max_util_gap_s=max((b - a).total_seconds() for a, b in zip(edges, edges[1:])) if times else "",
    )
    for field in METRICS:
        values = [samples[stamp][field] for stamp in sorted(samples) if field in samples[stamp]]
        row[f"supplemental_{field}_count"] = len(values)
        row[f"supplemental_{field}_mean"] = statistics.fmean(values) if values else ""
        row[f"supplemental_{field}_peak"] = max(values) if values else ""
    primary_power = [((s["query_start_s"] + s["query_end_s"]) / 2, s["device_power_w"]) for s in telemetry["samples"]]
    supplemental_power = [((stamp - start).total_seconds(), values.get("power_w")) for stamp, values in samples.items()]
    for prefix, readings in (("primary", primary_power), ("supplemental", supplemental_power)):
        energy, duration = observed_energy(readings, timing["wall_elapsed_s"])
        row[f"{prefix}_observed_energy_estimate_j"] = energy
        row[f"{prefix}_observed_power_duration_s"] = duration
    return row


def export(results, output, telemetry_dir=None):
    tasks, records = read_results(results)
    if not records:
        raise ValueError("No completed GEMM checkpoints to export.")
    manifest = json.loads((results / "manifest.json").read_bytes())
    raw_dir = results / "tasks" if (results / "tasks").is_dir() else results
    windows = []
    for task, shape in tasks.items():
        if shape in records:
            result = records[shape]
            start = utc(result["timed_start_utc"])
            end = start + timedelta(seconds=result["timing"]["wall_elapsed_s"])
            windows.append((start, end, task, result))
    windows.sort(key=lambda w: w[0])
    if any(current[0] < prev[1] for prev, current in zip(windows, windows[1:])):
        raise ValueError("Timed windows overlap; supplementary samples cannot be assigned uniquely.")

    folder = output / "supplemental"
    stale = {p.stem for p in folder.glob("*.csv")} - {task for _, _, task, _ in windows}
    if stale:
        raise ValueError("The output directory contains sidecars for other tasks. Use a separate directory.")
    observed = load_samples(telemetry_dir, windows)
    index, sidecars = [], {}
    for start, end, task, result in windows:
        path = folder / f"{task}.csv"
        samples = existing_samples(path, start, end)
        for stamp, values in observed.items():
            if start <= stamp < end:
                merge_values(samples, stamp, values)
        if samples:
            rows = ({"timestamp_utc": iso(t), **samples[t]} for t in sorted(samples))
            sidecars[path] = csv_text(rows, ("timestamp_utc", *METRICS))
        raw = (raw_dir / f"{task}.json").read_bytes()
        index.append(index_row(task, result, manifest, raw, start, end, samples))
    for path, content in sidecars.items():
        save(path, content)
    save(output / "run-index.csv", csv_text(index, index[0].keys()))
    print(f"Exported {len(index)} run(s), {len(sidecars)} supplementary sidecar(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Published GEMM archive or local lab state.")
    parser.add_argument("--output", required=True, type=Path, help="Directory for run-index.csv and supplemental/*.csv.")
    parser.add_argument("--telemetry-dir", type=Path, help="Optional folder of timestamped gemm-*.jsonl monitor output.")
    args = parser.parse_args()
    try:
        export(args.results, args.output, args.telemetry_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        sys.exit(f"Evidence export failed: {exc}")
