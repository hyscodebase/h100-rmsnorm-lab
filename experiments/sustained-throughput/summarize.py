"""Audit the frozen sustained-throughput plan and any saved raw checkpoints."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "plans/sustained-throughput.json"
RAW = ROOT / "results/sustained-throughput/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
SOURCES = (
    "kernels.py", "experiments/sustained-throughput/run.py",
    "experiments/sustained-throughput/telemetry.py",
    "experiments/sustained-throughput/PLAN.md", "plans/sustained-throughput.json",
    "experiments/sustained-throughput/step.py",
    "experiments/sustained-throughput/ssh_runner.sh",
    "experiments/sustained-throughput/summarize.py",
)
COMPUTE_SOURCES = SOURCES[:5]
BACKENDS = ("triton_4w_1r", "triton_8w_1r", "triton_4w_4r", "torch_compile")
GIB = 1024 ** 3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def plan_audit():
    p = json.loads(PLAN_PATH.read_text())
    need(p["issue"] == 64 and p["rows"] == 32768, "Issue/row count differs from frozen plan.")
    need(p["widths"] == [4103, 4104, 4105, 4112] and p["seeds"] == [50, 51, 52],
         "Frozen width or seed matrix changed.")
    need(p["candidate"] == "triton_4w_4r" and p["references"] == list(BACKENDS[:2]) + ["torch_compile"],
         "Frozen backend set changed.")
    need(p["ops_per_graph"] == 4 and p["latency_rounds"] == 401,
         "Latency graph or round count changed.")
    need(p["sustained_graph_calls_per_block"] == 32
         and p["sustained_event_windows_per_phase"] == 30
         and p["sustained_target_seconds_per_window"] == 1.0,
         "Sustained replay block/window protocol changed.")
    need(p["sustained_min_seconds"] == 30 and p["sustained_max_seconds"] == 60,
         "Sustained phase duration bounds changed.")
    need(p["minimum_active_telemetry_samples"] >= 120
         and p["telemetry_interval_seconds"] == 0.25
         and p["maximum_active_telemetry_gap_seconds"] == 1.5,
         "Active-window telemetry gate changed.")
    need(p["maximum_event_gap_between_windows_seconds"] == 0.005,
         "CUDA event adjacency gate changed.")
    output = p["rows"] * p["max_width"] * p["output_element_bytes"]
    inputs = p["rows"] * p["max_width"] * p["input_element_bytes"]
    reference = p["rows"] * p["max_width"] * p["reference_element_bytes"]
    live = inputs + reference + output * p["persistent_output_buffers_per_checkpoint"]
    latency_graph_bytes = inputs * 2 * p["ops_per_graph"]
    sustained_block_bytes = inputs * 2 * p["sustained_graph_calls_per_block"]
    need(p["max_width"] == max(p["widths"]), "Max width does not bound registered widths.")
    need((output, output * 4, inputs, reference, live) == (
        p["estimated_one_output_bytes"], p["estimated_persistent_outputs_bytes"],
        p["estimated_input_bytes"], p["estimated_fp32_reference_bytes"],
        p["estimated_live_buffers_bytes"]), "Persistent replay-buffer estimate mismatch.")
    need(latency_graph_bytes == p["estimated_latency_graph_nominal_bytes_at_max_width"],
         "Latency logical-byte estimate mismatch.")
    need(sustained_block_bytes == p["estimated_sustained_graph_block_nominal_bytes_at_max_width"],
         "Sustained block logical-byte estimate mismatch.")
    need(live == 1_886_388_224 and latency_graph_bytes == 2_155_872_256
         and sustained_block_bytes == 17_246_978_048, "Frozen byte math mismatch.")
    need(p["free_memory_guard_bytes"] == 16 * GIB
         and p["capture_reserved_delta_ceiling_bytes"] == 8 * GIB,
         "Memory guard differs from preregistered threshold.")
    need(p["expected_sustained_event_seconds_total"] == 12 * 4 * 30,
         "Expected total active CUDA-event time is not 24 minutes.")
    return p, {"persistent_live_bytes": live, "persistent_live_gib": live / GIB,
               "one_output_bytes": output, "latency_graph_logical_bytes": latency_graph_bytes,
               "sustained_block_logical_bytes": sustained_block_bytes,
               "latency_intervals_per_checkpoint": p["latency_rounds"] * len(BACKENDS),
               "latency_graph_calls_per_checkpoint": p["latency_rounds"] * len(BACKENDS) * p["ops_per_graph"],
               "sustained_target_seconds_total": p["expected_sustained_event_seconds_total"]}


def check_pin():
    need(SOURCE_PIN.is_file(), "Frozen source pin is missing.")
    pin = json.loads(SOURCE_PIN.read_text())
    actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    need(pin.get("source_sha256") == actual, "Current source differs from pinned SHA-256 hashes.")
    revision = pin.get("source_revision", "")
    need(len(revision) == 40, "Source pin does not name a full Git commit.")
    for name, digest in actual.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{name}"])
        need(hashlib.sha256(blob).hexdigest() == digest, f"Git source hash mismatch: {name}.")
    need(pin.get("research_assistant_model") == "GPT-6 Luna",
         "Research-assistant model pin is missing or changed.")
    return pin


def audit_checkpoint(path, plan):
    r = json.loads(path.read_text())
    need(r.get("status") == "complete", f"Checkpoint not complete: {path.name}.")
    need(path.stem == f"s{r['seed']}-w{r['width']}", f"Checkpoint identity mismatch: {path.name}.")
    need(r.get("source_revision"), f"Checkpoint has no source revision: {path.name}.")
    expected_sources = {key: hashlib.sha256((ROOT / key).read_bytes()).hexdigest()
                        for key in COMPUTE_SOURCES}
    need(r.get("source_sha256") == expected_sources, f"Checkpoint compute-source hash mismatch: {path.name}.")
    intervals = r.get("latency_intervals", [])
    need(len(intervals) == 401 * len(BACKENDS), f"Expected 1,604 latency intervals: {path.name}.")
    counts = {name: 0 for name in BACKENDS}
    by_backend = {name: [] for name in BACKENDS}
    for item in intervals:
        name = item["backend"]
        need(name in counts and item["ops_per_graph"] == 4, "Unexpected latency row/backend.")
        counts[name] += 1
        by_backend[name].append(item["graph_elapsed_us"])
    need(all(count == 401 for count in counts.values()), "Latency backend interval count mismatch.")
    medians = {name: statistics.median(by_backend[name]) / 4 for name in BACKENDS}
    phases = r.get("sustained_phases", [])
    need(len(phases) == 4 and {x.get("backend") for x in phases} == set(BACKENDS),
         "Expected one sustained phase per backend.")
    telemetry = {}
    for phase in phases:
        durations = phase["window_event_seconds"]
        gaps = phase["window_event_gaps_seconds"]
        need(len(durations) == 30 and len(gaps) == 29, "Sustained phase window cardinality mismatch.")
        need(all(math.isfinite(x) and 0 < x <= 2 for x in durations), "Invalid or overlong event window.")
        need(all(math.isfinite(x) and x <= 0.005 for x in gaps), "CUDA event-window gap exceeds 5 ms.")
        need(30 <= sum(durations) <= 60 and abs(sum(durations) - phase["event_seconds"]) < 1e-6,
             "Sustained event duration is outside 30–60s or not reproducible.")
        n = phase["replays_per_window"]
        calls = phase["graph_calls_per_block"]
        need(calls == 32 and n > 0, "Invalid sustained replay block count.")
        calls_total = 30 * n * calls
        raw_bytes = plan["rows"] * r["width"] * 2 * 2 * calls_total
        need(raw_bytes == phase["logical_input_plus_output_bytes"], "Recomputed logical bytes mismatch.")
        rate = raw_bytes / sum(durations) / 1e9
        need(abs(rate - phase["logical_input_plus_output_GB_per_s"]) <= max(1e-9, rate * 1e-8),
             "Recomputed logical throughput mismatch.")
        samples = phase.get("telemetry_samples", [])
        good = [s for s in samples if "error" not in s]
        times = [s["arrival_monotonic_ns"] for s in good]
        sample_gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
        need(len(good) >= 120 and sample_gaps and max(sample_gaps) <= 1.5,
             "Active-window telemetry coverage gate failed.")
        need(phase.get("telemetry_gate_passed") is True and phase.get("gate_passed") is True,
             "Recorded sustained phase gate did not pass.")
        telemetry[phase["backend"]] = {"poll_rows": len(good),
                                       "unique_gpu_busy_transitions": sum(
                                           a.get("gpu_busy_percent") != b.get("gpu_busy_percent")
                                           for a, b in zip(good, good[1:])),
                                       "gpu_busy_median_percent": statistics.median(
                                           [s["gpu_busy_percent"] for s in good if s.get("gpu_busy_percent") is not None]),
                                       "gpu_busy_peak_percent": max(
                                           [s["gpu_busy_percent"] for s in good if s.get("gpu_busy_percent") is not None], default=None),
                                       "power_median_w": statistics.median(
                                           [s["power_w"] for s in good if s.get("power_w") is not None]),
                                       "vram_peak_mib": max(
                                           [s["memory_used_mib"] for s in good if s.get("memory_used_mib") is not None], default=None),
                                       "max_query_gap_seconds": max(sample_gaps)}
    for section in ("fixtures", "timed_input", "post_latency_replay", "post_sustained_replay"):
        checks = r.get("correctness", {}).get(section, [])
        need(checks and all(c.get("strict_passed") and c.get("crosscheck_passed") for c in checks),
             f"Correctness section failed or absent: {section}.")
    need(r["memory"]["correctness_peak"]["peak_allocated_bytes"] > 0,
         "Measured correctness transient peak is missing.")
    need(r["memory"]["capture_reserved_delta_peak_bytes"] <= 8 * GIB,
         "Capture reserved-memory delta exceeded 8 GiB.")
    need(r["memory"]["before_allocations"]["free_bytes"] >= 16 * GIB,
         "Free-memory floor failed before allocations.")
    need(r["memory"]["after_persistent_allocation"]["free_bytes"] >= 16 * GIB,
         "Free-memory floor failed after persistent allocations.")
    return {"checkpoint": path.stem, "latency_median_us_per_op": medians,
            "phases": telemetry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true", help="Audit frozen plan and exact memory/replay arithmetic.")
    parser.add_argument("--audit", action="store_true", help="Audit source pin and all saved raw checkpoints.")
    args = parser.parse_args()
    plan, estimate = plan_audit()
    report = {"plan": "pass", "resource_estimate": estimate}
    if args.audit:
        pin = check_pin()
        files = sorted(CHECKPOINTS.glob("s*-w*.json"))
        results = [audit_checkpoint(path, plan) for path in files]
        expected = {f"s{s}-w{w}" for s in plan["seeds"] for w in plan["widths"]}
        names = {item["checkpoint"] for item in results}
        need(names <= expected, "Unexpected checkpoint in raw evidence.")
        report.update(source_revision=pin["source_revision"], source_pin="pass",
                      saved_checkpoints=len(names), expected_checkpoints=12,
                      checkpoint_results=results, all_complete=(names == expected))
        print(json.dumps(report, indent=2, allow_nan=False))
        need(names == expected, "Not all 12 checkpoints are available for final audit.")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
