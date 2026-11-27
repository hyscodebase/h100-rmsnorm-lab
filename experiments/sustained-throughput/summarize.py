"""Audit frozen sustained-throughput plans, receipts, and raw checkpoints."""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
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


def median_or_none(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def max_or_none(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return max(values) if values else None


def plan_audit():
    p = json.loads(PLAN_PATH.read_text())
    need(p["issue"] == 64 and p["rows"] == 32768, "Issue/row count differs from frozen plan.")
    need(p["widths"] == [4103, 4104, 4105, 4112] and p["seeds"] == [50, 51, 52],
         "Frozen width or seed matrix changed.")
    need(p["candidate"] == "triton_4w_4r" and p["references"] == [*BACKENDS[:2], "torch_compile"],
         "Frozen backend set changed.")
    need(p["ops_per_graph"] == 4 and p["latency_rounds"] == 401,
         "Frozen latency graph or round count changed.")
    need(p["sustained_graph_calls_per_block"] == 32
         and p["sustained_event_windows_per_phase"] == 30
         and p["sustained_target_seconds_per_window"] == 1.1
         and p["sustained_target_seconds"] == 33.0,
         "Sustained replay block/window protocol changed.")
    need(p["sustained_min_seconds"] == 30 and p["sustained_max_seconds"] == 60,
         "Sustained phase duration bounds changed.")
    need(p["minimum_active_telemetry_samples"] >= 25
         and p["telemetry_interval_seconds"] == 1.0
         and p["maximum_active_telemetry_gap_seconds"] == 1.5,
         "Active-window telemetry gate changed.")
    need(p["telemetry_required_field_coverage_fraction"] >= 0.95
         and set(p["telemetry_required_fields"]) == {
             "gpu_busy_percent", "memory_busy_percent", "memory_used_mib", "power_w", "temperature_c"},
         "Required telemetry field coverage changed.")
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
    need(p["expected_sustained_event_seconds_total"] == 12 * 4 * p["sustained_target_seconds"],
         "Expected active CUDA-event time does not match the headroom target.")
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


def audit_attempts(pin, checkpoint_records):
    receipts = {}
    per_unit_counts = defaultdict(int)
    for path in sorted(ATTEMPTS.glob("*.json")) if ATTEMPTS.exists() else []:
        item = json.loads(path.read_text())
        attempt_id = item.get("attempt_id")
        need(attempt_id and attempt_id not in receipts, f"Missing or duplicate attempt id: {path.name}.")
        need(path.stem == f"{item.get('unit')}-{attempt_id}", f"Attempt filename/identity mismatch: {path.name}.")
        unit = item.get("unit")
        expected_unit = f"s{item.get('seed')}-w{item.get('width')}"
        need(unit == expected_unit, f"Attempt checkpoint identity mismatch: {path.name}.")
        per_unit_counts[unit] += 1
        need(per_unit_counts[unit] <= 3, f"More than three attempt receipts for checkpoint: {unit}.")
        need(item.get("seed") in [50, 51, 52] and item.get("width") in [4103, 4104, 4105, 4112],
             f"Attempt contains unregistered checkpoint: {path.name}.")
        receipt_revision = item.get("source_revision", "")
        need(len(receipt_revision) == 40, f"Attempt has no full source revision: {path.name}.")
        need(item.get("outcome") in {"checkpoint_saved", "gate_failed", "transport_or_parse_failed", "paused_before_submit"},
             f"Unknown attempt outcome: {path.name}.")
        need(item.get("outcome") != "started", f"Unfinished attempt receipt: {path.name}.")
        need(item.get("finished_utc") and item.get("elapsed_seconds", -1) >= 0,
             f"Attempt receipt lacks completion metadata: {path.name}.")
        source_manifest = item.get("source_sha256")
        need(isinstance(source_manifest, dict) and set(source_manifest) == set(SOURCES)
             and all(isinstance(digest, str) and len(digest) == 64
                     for digest in source_manifest.values()),
             f"Attempt source hash manifest is malformed: {path.name}.")
        if receipt_revision == pin["source_revision"]:
            need(item.get("source_sha256") == pin["source_sha256"],
                 f"Current attempt source pin mismatch: {path.name}.")
        else:
            need(item.get("outcome") in {"transport_or_parse_failed", "paused_before_submit"},
                 f"Executed attempt must use the current source pin: {path.name}.")
            for source, digest in item["source_sha256"].items():
                blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{receipt_revision}:{source}"])
                need(hashlib.sha256(blob).hexdigest() == digest,
                     f"Historical transport attempt source mismatch at {receipt_revision}:{source}.")
        receipts[attempt_id] = item
    complete_units = set()
    successful_receipts = defaultdict(list)
    for attempt_id, item in receipts.items():
        if item.get("outcome") == "checkpoint_saved" and item.get("checkpoint_status") == "complete":
            successful_receipts[item["unit"]].append(attempt_id)
    for unit, record in checkpoint_records.items():
        attempt_id = record.get("attempt_id")
        need(attempt_id in receipts, f"Checkpoint has no attempt receipt: {unit}.")
        receipt = receipts[attempt_id]
        need(receipt.get("unit") == unit, f"Checkpoint/receipt unit mismatch: {unit}.")
        checkpoint_path = CHECKPOINTS / f"{unit}.json"
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        need(receipt.get("checkpoint_path") == str(checkpoint_path.relative_to(ROOT))
             and receipt.get("checkpoint_sha256") == digest,
             f"Attempt receipt checkpoint hash/path mismatch: {unit}.")
        if record.get("status") == "complete":
            need(receipt.get("outcome") == "checkpoint_saved"
                 and receipt.get("checkpoint_status") == "complete",
                 f"Complete checkpoint lacks a successful attempt receipt: {unit}.")
            successes = successful_receipts.get(unit, [])
            need(len(successes) == 1 and successes[0] == attempt_id,
                 f"Successful checkpoint must have exactly one receipt matching its attempt_id: {unit}.")
            complete_units.add(unit)
        else:
            need(receipt.get("outcome") == "gate_failed"
                 and receipt.get("checkpoint_status") == record.get("status"),
                 f"Failed checkpoint/attempt receipt status mismatch: {unit}.")
    need(all(count <= 3 for count in per_unit_counts.values()), "Attempt limit exceeded.")
    for unit in successful_receipts:
        need(unit in complete_units, f"Orphan successful attempt receipt for {unit}.")
    for attempt_id, receipt in receipts.items():
        if receipt.get("outcome") in {"checkpoint_saved", "gate_failed"}:
            need(receipt.get("unit") in checkpoint_records,
                 f"Checkpoint attempt receipt has no saved checkpoint: {attempt_id}.")
    return {"receipt_count": len(receipts), "saved_checkpoint_receipts": len(checkpoint_records),
            "successful_checkpoint_receipts": len(complete_units)}


def telemetry_summary(samples, phase, plan):
    good = [sample for sample in samples if "error" not in sample]
    phase_start = phase.get("phase_start_monotonic_ns")
    phase_end = phase.get("phase_end_monotonic_ns")
    need(isinstance(phase_start, int) and isinstance(phase_end, int) and phase_start < phase_end,
         "Phase monotonic boundaries are absent or invalid.")
    arrivals = [sample.get("arrival_monotonic_ns") for sample in samples]
    need(all(isinstance(t, int) and phase_start <= t <= phase_end for t in arrivals),
         "A retained telemetry row falls outside the recorded phase monotonic bounds.")
    times = [sample.get("arrival_monotonic_ns") for sample in good
             if isinstance(sample.get("arrival_monotonic_ns"), int)]
    gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
    fields = {}
    coverage_pass = True
    for field in plan["telemetry_required_fields"]:
        values = [s.get(field) for s in good if s.get(field) is not None]
        coverage = len(values) / len(samples) if samples else 0.0
        passed = coverage >= plan["telemetry_required_field_coverage_fraction"]
        coverage_pass &= passed
        fields[field] = {"count": len(values), "coverage_fraction": coverage,
                         "median": median_or_none(values), "peak": max_or_none(values),
                         "coverage_passed": passed}
    query_pass = (len(good) >= plan["minimum_active_telemetry_samples"] and gaps
                  and max(gaps) <= plan["maximum_active_telemetry_gap_seconds"])
    return {"poll_rows": len(samples), "valid_rows": len(good),
            "max_query_gap_seconds": max(gaps) if gaps else None,
            "unique_reading_transitions": sum(
                any(a.get(f) != b.get(f) for f in plan["telemetry_required_fields"])
                for a, b in zip(good, good[1:])),
            "fields": fields,
            "coverage_gate_passed": bool(coverage_pass),
            "query_gate_passed": bool(query_pass),
            "eligible": bool(coverage_pass and query_pass and phase.get("telemetry_gate_passed"))}


def audit_checkpoint(path, plan, pin):
    r = json.loads(path.read_text())
    need(path.stem == f"s{r['seed']}-w{r['width']}", f"Checkpoint identity mismatch: {path.name}.")
    need(r.get("source_revision") == pin["source_revision"], f"Checkpoint revision mismatch: {path.name}.")
    expected_sources = {key: pin["source_sha256"][key] for key in COMPUTE_SOURCES}
    need(r.get("source_sha256") == expected_sources, f"Checkpoint compute-source hash mismatch: {path.name}.")
    need(r.get("execution_source_sha256") == pin["source_sha256"], f"Checkpoint execution-source hash mismatch: {path.name}.")
    if r.get("status") != "complete":
        return {"checkpoint": path.stem, "seed": r["seed"], "width": r["width"],
                "status": r.get("status"), "attempt_id": r.get("attempt_id"),
                "failure": r.get("failure"), "raw_evidence_saved": True, "analysis_eligible": False}
    intervals = r.get("latency_intervals", [])
    need(len(intervals) == 401 * len(BACKENDS), f"Expected 1,604 latency intervals: {path.name}.")
    counts = {name: 0 for name in BACKENDS}
    by_backend = {name: [] for name in BACKENDS}
    interval_orders = {}
    expected_rng = random.Random(r["seed"] ^ (r["width"] << 8))
    expected_orders = []
    for _ in range(401):
        order = list(BACKENDS)
        expected_rng.shuffle(order)
        expected_orders.append(order)
    for index, item in enumerate(intervals):
        name = item["backend"]
        need(name in counts and item["ops_per_graph"] == 4, "Unexpected latency row/backend.")
        round_id, order_index = item.get("round"), item.get("order_index")
        need(isinstance(round_id, int) and 0 <= round_id < 401 and order_index in range(4),
             "Latency round/order metadata is invalid.")
        expected_index = round_id * len(BACKENDS) + order_index
        need(index == expected_index and expected_orders[round_id][order_index] == name,
             "Latency rows do not match the frozen randomized round/order schedule.")
        elapsed = item.get("graph_elapsed_us")
        per_op = item.get("us_per_op")
        need(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0,
             "Latency graph duration must be finite and positive.")
        need(isinstance(per_op, (int, float)) and math.isfinite(per_op)
             and math.isclose(per_op, elapsed / 4, rel_tol=1e-9, abs_tol=1e-12),
             "Latency per-op value does not match raw graph duration.")
        counts[name] += 1
        by_backend[name].append(elapsed)
        interval_orders.setdefault(round_id, []).append(name)
    need(all(count == 401 for count in counts.values()), "Latency backend interval count mismatch.")
    need(set(interval_orders) == set(range(401))
         and all(len(order) == 4 and set(order) == set(BACKENDS) for order in interval_orders.values()),
         "Latency round coverage/order is incomplete.")
    latency_medians = {name: statistics.median(by_backend[name]) / 4 for name in BACKENDS}
    phases = r.get("sustained_phases", [])
    need(len(phases) == 4 and {x.get("backend") for x in phases} == set(BACKENDS),
         "Expected one sustained phase per backend.")
    phase_results = {}
    for phase in phases:
        durations = phase["window_event_seconds"]
        gaps = phase["window_event_gaps_seconds"]
        replay_counts = phase.get("window_replay_counts", [])
        need(len(durations) == 30 and len(gaps) == 29 and len(replay_counts) == 30,
             "Sustained phase window/replay count cardinality mismatch.")
        need(all(isinstance(x, int) and x > 0 for x in replay_counts),
             "Invalid per-window replay count.")
        need(all(math.isfinite(x) and 0 < x <= 2 for x in durations), "Invalid or overlong event window.")
        need(all(math.isfinite(x) and x <= 0.005 for x in gaps), "CUDA event-window gap exceeds 5 ms.")
        need(30 <= sum(durations) <= 60 and abs(sum(durations) - phase["event_seconds"]) < 1e-6,
             "Sustained event duration is outside 30–60s or not reproducible.")
        calls = phase["graph_calls_per_block"]
        need(calls == 32 and phase.get("replays_per_window") == replay_counts[0],
             "Sustained replay block count differs from per-window records.")
        total_blocks = sum(replay_counts)
        total_calls = total_blocks * calls
        raw_bytes = plan["rows"] * r["width"] * plan["input_element_bytes"] * 2 * total_calls
        need(raw_bytes == phase["logical_input_plus_output_bytes"], "Recomputed logical bytes mismatch.")
        rate = raw_bytes / sum(durations) / 1e9
        need(abs(rate - phase["logical_input_plus_output_GB_per_s"]) <= max(1e-9, rate * 1e-8),
             "Recomputed logical throughput mismatch.")
        samples = phase.get("telemetry_samples", [])
        telemetry = telemetry_summary(samples, phase, plan)
        need(telemetry["eligible"], f"Telemetry fields/coverage make phase ineligible: {phase['backend']}.")
        need(phase.get("gate_passed") is True, "Recorded sustained phase gate did not pass.")
        phase_results[phase["backend"]] = {
            "logical_input_plus_output_GB_per_s": rate,
            "event_seconds": sum(durations), "total_graph_blocks": total_blocks,
            "total_graph_calls": total_calls, "window_replay_counts": replay_counts,
            "telemetry": telemetry}
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
    return {"checkpoint": path.stem, "seed": r["seed"], "width": r["width"], "status": "complete",
            "attempt_id": r.get("attempt_id"),
            "latency_median_us_per_op": latency_medians, "phases": phase_results}


def compare_checkpoints(results, plan):
    by_unit = {item["checkpoint"]: item for item in results if item.get("status") == "complete"}
    by_width = {}
    checkpoint_decisions = []
    threshold = plan["selection"]["minimum_gain_percent"]
    for seed in plan["seeds"]:
        for width in plan["widths"]:
            key = f"s{seed}-w{width}"
            if key not in by_unit:
                continue
            phases = by_unit[key]["phases"]
            rates = {backend: phases[backend]["logical_input_plus_output_GB_per_s"] for backend in BACKENDS}
            fastest_one_row_name = max(BACKENDS[:2], key=lambda name: rates[name])
            comparators = {"fastest_one_row": fastest_one_row_name, "torch_compile": "torch_compile"}
            gains = {label: 100.0 * (rates["triton_4w_4r"] / rates[backend] - 1.0)
                     for label, backend in comparators.items()}
            qualifies = all(gain >= threshold for gain in gains.values())
            checkpoint_decisions.append({"seed": seed, "width": width, "rates_GB_per_s": rates,
                                         "fastest_one_row_backend": fastest_one_row_name,
                                         "gain_percent": gains, "qualifies": qualifies})
    for width in plan["widths"]:
        rows = [item for item in checkpoint_decisions if item["width"] == width]
        if not rows:
            continue
        backends = {name: median_or_none([row["rates_GB_per_s"][name] for row in rows]) for name in BACKENDS}
        gain_one = [row["gain_percent"]["fastest_one_row"] for row in rows]
        gain_torch = [row["gain_percent"]["torch_compile"] for row in rows]
        by_width[str(width)] = {"seed_count": len(rows), "median_rate_GB_per_s": backends,
                                "median_gain_percent_vs_fastest_one_row": median_or_none(gain_one),
                                "median_gain_percent_vs_torch_compile": median_or_none(gain_torch),
                                "qualifying_seeds": sum(row["qualifies"] for row in rows),
                                "all_registered_seeds_qualify": len(rows) == len(plan["seeds"])
                                and all(row["qualifies"] for row in rows)}
    complete = (len(results) == len(plan["seeds"]) * len(plan["widths"])
                and all(item.get("status") == "complete" for item in results))
    all_qualify = complete and len(checkpoint_decisions) == 12 and all(r["qualifies"] for r in checkpoint_decisions)
    return {"checkpoint_comparisons": checkpoint_decisions, "per_width_across_seeds": by_width,
            "candidate_adoption_decision": {
                "eligible": complete,
                "adopt_four_row_candidate": all_qualify,
                "rule": f"candidate must be at least {threshold:.1f}% faster than fastest one-row Triton and torch_compile in every registered seed and width",
                "reason": "all 12 checkpoint comparisons pass" if all_qualify else
                          ("awaiting all 12 checkpoints" if not complete else "one or more comparisons miss the preregistered threshold")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true", help="Audit frozen plan and resource/replay arithmetic.")
    parser.add_argument("--audit", action="store_true", help="Audit all currently saved checkpoints and receipts; partial is allowed.")
    parser.add_argument("--final", action="store_true", help="Require and audit all 12 registered checkpoints.")
    args = parser.parse_args()
    plan, estimate = plan_audit()
    report = {"plan": "pass", "resource_estimate": estimate}
    if args.audit or args.final:
        pin = check_pin()
        records = {}
        files = sorted(CHECKPOINTS.glob("s*-w*.json")) if CHECKPOINTS.exists() else []
        expected = {f"s{s}-w{w}" for s in plan["seeds"] for w in plan["widths"]}
        for path in files:
            need(path.stem in expected, f"Unexpected checkpoint file: {path.name}.")
            need(path.stem not in records, f"Duplicate checkpoint unit: {path.stem}.")
            records[path.stem] = audit_checkpoint(path, plan, pin)
        receipt_audit = audit_attempts(pin, records)
        comparisons = compare_checkpoints(list(records.values()), plan)
        complete = (set(records) == expected
                    and all(item.get("status") == "complete" for item in records.values()))
        report.update(source_revision=pin["source_revision"], source_pin="pass",
                      saved_checkpoints=len(records), successful_checkpoints=sum(
                          item.get("status") == "complete" for item in records.values()),
                      expected_checkpoints=12, final_complete=complete, receipt_audit=receipt_audit,
                      checkpoint_results=list(records.values()), **comparisons)
        print(json.dumps(report, indent=2, allow_nan=False))
        need(not args.final or complete, "Final audit requires all 12 registered checkpoints.")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
