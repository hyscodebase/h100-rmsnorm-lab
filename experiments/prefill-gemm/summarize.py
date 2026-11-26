"""Validate the public GEMM checkpoints and print their measured results."""

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import gemm_plan
import lab


def read_results(directory):
    plan = lab.read_json(ROOT / "plans/prefill-gemm.json")
    lab.validate_plan(plan)
    lab.require(plan["kind"] == "prefill_gemm" and len(plan["seeds"]) == 5,
                "The qualification rule requires five planned GEMM seeds per shape.")
    manifest = lab.read_json(directory / "manifest.json")
    sources = manifest["sources"]
    lab.require(manifest["version"] == 1 and set(sources) == {gemm_plan.SOURCE},
                "Unexpected source manifest.")
    digest = hashlib.sha256(json.dumps([plan, sources], sort_keys=True, allow_nan=False).encode()).hexdigest()
    lab.require(manifest["experiment_hash"] == digest, "Plan and source hashes differ from the manifest.")
    tasks = dict(lab.tasks(plan))
    raw = directory / "tasks" if (directory / "tasks").is_dir() else directory
    observed = {f.stem for f in raw.glob("*.json") if f.name != "manifest.json"}
    lab.require(observed <= tasks.keys(), f"Unplanned checkpoints: {sorted(observed - tasks.keys())}")
    records, revisions = {}, set()
    environment = None
    for task, shape in tasks.items():
        if task not in observed:
            continue
        cell = lab.read_json(raw / f"{task}.json")
        lab.require(cell["task_id"] == task and cell["experiment_hash"] == digest,
                    f"Checkpoint identity mismatch in {task}.")
        result = cell["result"]
        revision = result["code_revision"]
        if revision not in revisions:
            lab.require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision),
                        f"Missing full source revision in {task}.")
            blob = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "show",
                                   f"{revision}:{gemm_plan.SOURCE}"], capture_output=True)
            lab.require(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == sources[gemm_plan.SOURCE],
                        f"Cannot verify benchmark source at {revision}. Use a clone containing that revision.")
            revisions.add(revision)
        env = lab.validate_result(result, shape, plan)
        lab.require("H100" in env["gpu"] and environment in (None, env), f"Inconsistent H100 environment in {task}.")
        environment = env
        records[shape] = result
    return tasks, records


def activity_coverage(result):
    telemetry = result["telemetry"]
    timing = result["timing"]
    period = telemetry["requested_period_s"]
    lab.require(period == 1, "The planned device sampling period is one second.")
    midpoints = sorted((s["query_start_s"] + s["query_end_s"]) / 2 for s in telemetry["samples"]
                       if s["gpu_busy_pct"] is not None)
    edges = [0, *midpoints, timing["wall_elapsed_s"]]
    max_gap = max(end - start for start, end in zip(edges, edges[1:]))
    required_count = math.ceil(0.75 * timing["requested_seconds"] / period)
    return len(midpoints) >= required_count and max_gap <= 5, required_count, max_gap


def show(tasks, records):
    print(f"Completed checkpoints: {len(records)}/{len(tasks)}. "
          "Telemetry covers device-wide samples collected during each timed interval. "
          "The post-pilot coverage rule requires at least 75% of planned samples (rounded up) and gaps "
          "of at most 5 s between usable GPU readings, including both window edges.\n")
    print("| Seed | M × K × N | GEMMs | Wall TFLOP/s | Mean GPU busy (%) | Mean power (W) | "
          "Peak observed memory (MiB) | Samples (GPU / power / memory) | Activity coverage |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    by_shape = {}
    for (seed, m, k, n), result in records.items():
        timing = result["timing"]
        iterations = sum(b["iterations"] for b in timing["batches"])
        throughput = iterations * (2 * m * k * n) / timing["wall_elapsed_s"] / 1e12
        samples = result["telemetry"]["samples"]
        gpu, power, memory = ([s[field] for s in samples if s[field] is not None]
                              for field in ("gpu_busy_pct", "device_power_w", "device_memory_used_mib"))
        activity = statistics.fmean(gpu)
        watts = f"{statistics.fmean(power):.1f}" if power else "n/a"
        mib = f"{max(memory):,.0f}" if memory else "n/a"
        sufficient, required, gap = activity_coverage(result)
        coverage = "Sufficient" if sufficient else "Insufficient"
        print(f"| {seed} | {m} × {k} × {n} | {iterations:,} | {throughput:.2f} | {activity:.2f} | "
              f"{watts} | {mib} | {len(gpu)} / {len(power)} / {len(memory)} | "
              f"{coverage} ({required} required, max gap {gap:.2f} s) |")
        by_shape.setdefault((m, k, n), []).append((throughput, sufficient and activity >= 95))

    if len(records) < len(tasks):
        print(f"\nQualification and ranking pending until all {len(tasks)} planned checkpoints are available.")
        return
    print("\nA shape qualifies when mean GPU activity reaches 95% with sufficient coverage in at least four of five seeds. "
          "Qualifying shapes are ordered by median wall throughput across all five seeds.\n")
    print("| M × K × N | Seeds meeting activity and coverage criteria | Median wall TFLOP/s | Qualifies |")
    print("| --- | ---: | ---: | --- |")
    ranked = []
    for shape, runs in by_shape.items():
        active = sum(meets_activity_rule for _, meets_activity_rule in runs)
        median = statistics.median(throughput for throughput, _ in runs)
        ranked.append((active >= 4, median, shape, active))
    ranked.sort(key=lambda record: (not record[0], -record[1], record[2]))
    for qualifies, median, (m, k, n), active in ranked:
        print(f"| {m} × {k} × {n} | {active}/5 | {median:.2f} | {'Yes' if qualifies else 'No'} |")
    eligible = [row for row in ranked if row[0]]
    if len(eligible) >= 2:
        gap = (eligible[0][1] - eligible[1][1]) / eligible[0][1]
        outcome = "Near tie" if gap <= 0.03 else "Separated by more than 3%"
        print(f"\n{outcome}. The top two qualifying medians differ by {gap:.2%} of the leading median.")
    elif not eligible:
        print("\nNo shape reached the sustained activity threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results", help="published archive or lab state directory")
    try:
        show(*read_results(parser.parse_args().results))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        sys.exit(f"GEMM summary failed: {exc}")
