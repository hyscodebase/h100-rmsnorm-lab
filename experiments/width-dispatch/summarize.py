"""Audit the frozen width-dispatch plan, receipts, and raw checkpoints."""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "plans/width-dispatch.json"
RAW = ROOT / "results/width-dispatch/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
RUN_ID_PATH = ROOT / ".lab/width-dispatch/run-id.json"
PREDECESSOR = RAW / "archives/driver-preflight-d42481a"
EVIDENCE_COMMIT = "88b3930b4f0796a875e872e1cd4a965a3453541c"
RESUMED_DRIVER_RUNNER = "d0394d789ada7e284e4878cbb7b1f3e7afa7470d3a3f36f44a940f9ffea2d83b"
RESUME_UNIT = "s53-w4107"
CARRIED_UNITS = {"s53-w4102", "s53-w4106"}
SOURCES = (
    "kernels.py", "experiments/width-dispatch/run.py",
    "experiments/width-dispatch/telemetry.py",
    "experiments/width-dispatch/PLAN.md", "plans/width-dispatch.json",
    "lab.py",
    "experiments/width-dispatch/step.py",
    "experiments/width-dispatch/ssh_runner.sh",
    "experiments/width-dispatch/summarize.py",
    "experiments/width-dispatch/RESUME.md",
)
COMPUTE_SOURCES = SOURCES[:5]
BACKENDS = ("triton_4w_1r", "triton_8w_1r", "triton_4w_4r", "torch_compile")
GIB = 1024 ** 3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite_number(value):
    """Accept JSON numeric evidence without Python's bool-is-an-int shortcut."""
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return type(value) is int


def median_or_none(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def max_or_none(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return max(values) if values else None


def utc_epoch_ns(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000_000_000)


def plan_audit():
    p = json.loads(PLAN_PATH.read_text())
    need(p["issue"] == 66 and p["rows"] == 32768, "Issue/row count differs from frozen plan.")
    need(p["widths"] == [4102, 4106, 4107, 4108, 4109, 4111, 4113, 4116]
         and p["seeds"] == [53, 54, 55],
         "Frozen width or seed matrix changed.")
    need(p["fixtures"] == ["random", "zero_rows", "constant", "tiny", "large",
                           "alternating", "column_ramp", "outlier"],
         "Frozen correctness fixture set changed.")
    dispatch = p["selection"]["dispatch_pattern"]
    need(dispatch["odd_widths"] == [4107, 4109, 4111, 4113]
         and dispatch["even_widths"] == [4102, 4106, 4108, 4116]
         and set(dispatch["odd_widths"] + dispatch["even_widths"]) == set(p["widths"]),
         "Bounded dispatch-screen groups changed.")
    need(p["candidate"] == "triton_4w_4r" and p["references"] == [*BACKENDS[:2], "torch_compile"],
         "Frozen backend set changed.")
    need(p["selection"]["minimum_gain_percent"] == 5.0
         and p["selection"]["must_beat_each"] == ["fastest_one_row_triton", "torch_compile"]
         and p["selection"]["must_pass_all_seeds"] is True
         and p["selection"]["throughput_metric"] == "logical_input_plus_output_GB_per_s",
         "Frozen 5% selection rule or throughput metric changed.")
    need(p["ops_per_graph"] == 4 and p["latency_rounds"] == 401,
         "Frozen latency graph or round count changed.")
    need(p["eps"] == 1e-6 and p["dtype"] == "bfloat16"
         and p["strict_correctness"] == {"rtol": 0.005, "atol": 1e-6}
         and p["benchmark_crosscheck"] == {"rtol": 0.016, "atol": 1e-5},
         "Frozen epsilon, dtype or correctness tolerances changed.")
    need(p["sustained_graph_calls_per_block"] == 32
         and p["sustained_event_windows_per_phase"] == 30
         and p["sustained_target_seconds_per_window"] == 1.1
         and p["sustained_target_seconds"] == 33.0,
         "Sustained replay block/window protocol changed.")
    need(p["sustained_min_seconds"] == 30 and p["sustained_max_seconds"] == 60,
         "Sustained phase duration bounds changed.")
    need(p["minimum_active_telemetry_samples"] >= 25
         and p["telemetry_interval_seconds"] == 0.25
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
    need(live == 1_888_223_232 and latency_graph_bytes == 2_157_969_408
         and sustained_block_bytes == 17_263_755_264, "Frozen byte math mismatch.")
    need(p["standard_max_attempts_per_checkpoint"] == 3
         and p["max_manual_transport_retries_per_checkpoint"] == 0
         and p["max_manual_telemetry_retries_per_checkpoint"] == 0
         and p["max_attempts_per_checkpoint"] == 3
         and p["manual_transport_retry_unit"] is None
         and p["manual_telemetry_retry_unit"] is None,
         "Three-attempt stop policy changed.")
    need(p["remote_timeout_seconds"] == 300
         and p["remote_timeout_cleanup_reserve_seconds"] == 20
         and p["remote_total_timeout_seconds"] == 350,
         "Benchmark, timeout cleanup reserve or complete remote-operation deadline changed.")
    need(p["free_memory_guard_bytes"] == 16 * GIB
         and p["capture_reserved_delta_ceiling_bytes"] == 8 * GIB,
         "Memory guard differs from preregistered threshold.")
    need(p["expected_sustained_event_seconds_total"]
         == len(p["seeds"]) * len(p["widths"]) * len(BACKENDS) * p["sustained_target_seconds"],
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
    revision = pin.get("source_revision", "")
    source_hashes = pin.get("source_sha256", {})
    need(len(revision) == 40, "Source pin does not name a full Git commit.")
    need(set(source_hashes) == set(SOURCES)
         and all(isinstance(digest, str) and len(digest) == 64
                 for digest in source_hashes.values()),
         "Frozen source pin manifest is incomplete.")
    for name, digest in source_hashes.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{name}"])
        need(hashlib.sha256(blob).hexdigest() == digest, f"Git source hash mismatch: {name}.")
        current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        need(current == digest, f"Working source differs from the frozen source pin: {name}.")
    need(pin.get("research_assistant_model") == json.loads(PLAN_PATH.read_text())["research_assistant_model"],
         "Research-assistant model provenance is missing or changed.")
    need(pin.get("h100_execution_scope") in {"pilot", "full_matrix"},
         "H100 source pin lacks an explicit pilot/full-matrix execution scope.")
    has_active_evidence = (any(ATTEMPTS.glob("*.json")) if ATTEMPTS.exists() else False) \
        or (any(CHECKPOINTS.glob("*.json")) if CHECKPOINTS.exists() else False)
    run_id = pin.get("campaign_id")
    need(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id),
         "The continuation pin lacks its campaign ID.")
    if has_active_evidence and RUN_ID_PATH.is_file():
        state = json.loads(RUN_ID_PATH.read_text())
        need(state.get("source_revision") == revision
             and state.get("source_sha256") == source_hashes
             and state.get("run_id") == run_id, "Checkout run identity differs from its pin.")
    pin["active_run_id"] = run_id
    return pin


def audit_correctness(record, plan, allow_partial_sustained=False):
    correctness = record.get("correctness", {})
    fixtures = correctness.get("fixtures", [])
    expected_fixture_rows = {(fixture, backend) for fixture in plan["fixtures"]
                             for backend in BACKENDS}
    actual_fixture_rows = {(item.get("fixture"), item.get("backend")) for item in fixtures}
    need(len(fixtures) == len(expected_fixture_rows)
         and actual_fixture_rows == expected_fixture_rows,
         "Fixture correctness coverage is incomplete, duplicated or unexpected.")
    need(all(item.get("finite") is True and item.get("strict_passed") is True
             and item.get("crosscheck_passed") is True for item in fixtures),
         "Fixture correctness failed or contains non-finite output.")
    for section in ("timed_input", "post_latency_replay"):
        checks = correctness.get(section, [])
        need(isinstance(checks, list) and len(checks) == len(BACKENDS)
             and {item.get("backend") for item in checks} == set(BACKENDS),
             f"Correctness section has missing, duplicate or unexpected backend checks: {section}.")
        need(all(item.get("finite") is True and item.get("strict_passed") is True
                 and item.get("crosscheck_passed") is True for item in checks),
             f"Correctness section failed or contains non-finite output: {section}.")
    sustained = correctness.get("post_sustained_replay", [])
    if allow_partial_sustained:
        need(len(sustained) == 1 and sustained[0].get("backend") == "triton_4w_4r",
             "Telemetry-only recovery checkpoint lacks its first candidate output check.")
    else:
        need(len(sustained) == len(BACKENDS)
             and {item.get("backend") for item in sustained} == set(BACKENDS),
             "Post-sustained correctness is incomplete or duplicated.")
    need(all(item.get("finite") is True and item.get("strict_passed") is True
             and item.get("crosscheck_passed") is True for item in sustained),
         "Post-sustained correctness failed or contains non-finite output.")
    graph_checks = correctness.get("graph_write_checks", [])
    expected_graph_checks = {(backend, kind) for backend in BACKENDS
                             for kind in ("latency", "sustained")}
    actual_graph_checks = {(item.get("backend"), item.get("graph_kind"))
                           for item in graph_checks}
    need(len(graph_checks) == len(expected_graph_checks)
         and actual_graph_checks == expected_graph_checks,
         "Captured graph-write verification is incomplete or duplicated.")
    need(all(item.get("finite") is True and item.get("strict_passed") is True
             and item.get("crosscheck_passed") is True for item in graph_checks),
         "A captured graph did not write a complete finite expected output.")


def audit_attempts(pin, checkpoint_records, plan, raw=RAW):
    receipts = {}
    per_unit = defaultdict(list)
    allowed = {"checkpoint_saved", "gate_failed", "transport_or_parse_failed",
               "local_launch_failed", "paused_before_submit"}
    for path in sorted((raw / "attempts").glob("*.json")):
        item = json.loads(path.read_text())
        attempt_id = item.get("attempt_id")
        unit = item.get("unit")
        need(attempt_id and attempt_id not in receipts,
             f"Missing or duplicate attempt ID: {path.name}.")
        need(path.stem == f"{unit}-{attempt_id}",
             f"Attempt filename/identity mismatch: {path.name}.")
        need(unit == f"s{item.get('seed')}-w{item.get('width')}",
             f"Attempt checkpoint identity mismatch: {path.name}.")
        need(item.get("seed") in plan["seeds"] and item.get("width") in plan["widths"],
             f"Attempt contains an unregistered checkpoint: {path.name}.")
        need(item.get("source_revision") == pin["source_revision"]
             and item.get("source_sha256") == pin["source_sha256"],
             f"Attempt does not match the current source pin: {path.name}.")
        need(item.get("run_id") == pin.get("active_run_id"),
             f"Attempt does not match the active checkout run identity: {path.name}.")
        need(item.get("outcome") in allowed,
             f"Unknown or unfinished attempt outcome: {path.name}.")
        need(item.get("finished_utc"),
             f"Attempt receipt lacks a completion timestamp: {path.name}.")
        elapsed = item.get("elapsed_seconds")
        if elapsed is None:
            need(item.get("reconciled_after_interruption") is True
                 and item.get("local_elapsed_seconds_observed") is False,
                 f"Unknown attempt duration is not marked as interrupted: {path.name}.")
        else:
            need(finite_number(elapsed) and elapsed >= 0,
                 f"Attempt receipt has an invalid local duration: {path.name}.")
        remote_elapsed = item.get("remote_benchmark_wall_seconds")
        need(remote_elapsed is None or
             (finite_number(remote_elapsed) and remote_elapsed >= 0),
             f"Attempt receipt has an invalid remote benchmark duration: {path.name}.")
        need(not item.get("manual_transport_retry_override")
             and not item.get("manual_telemetry_retry_override"),
             f"Manual retry is not registered in this protocol: {path.name}.")
        receipts[attempt_id] = item
        per_unit[unit].append(item)

    for unit, items in per_unit.items():
        ordered = sorted(items, key=lambda item: item.get("started_utc", ""))
        submitted = [item for item in ordered
                     if item.get("outcome") not in {"paused_before_submit", "local_launch_failed"}]
        need(len(submitted) <= plan["max_attempts_per_checkpoint"],
             f"Attempt limit exceeded for checkpoint: {unit}.")
        outcomes = [item["outcome"] for item in ordered]
        if "gate_failed" in outcomes:
            need(outcomes[-1] == "gate_failed" and outcomes.count("gate_failed") == 1,
                 f"A retry followed a blocking gate failure: {unit}.")
        if "checkpoint_saved" in outcomes:
            need(outcomes[-1] == "checkpoint_saved" and outcomes.count("checkpoint_saved") == 1,
                 f"A retry followed a successful checkpoint: {unit}.")

    for attempt_id, item in receipts.items():
        if item["outcome"] not in {"checkpoint_saved", "gate_failed"}:
            continue
        unit = item["unit"]
        need(unit in checkpoint_records,
             f"Saved-checkpoint receipt has no checkpoint file: {attempt_id}.")
        record = checkpoint_records[unit]
        checkpoint_path = CHECKPOINTS / f"{unit}.json"
        digest = hashlib.sha256((raw / "checkpoints" / checkpoint_path.name).read_bytes()).hexdigest()
        producing_attempt_id = record.get("producing_attempt_id", attempt_id)
        need(record.get("attempt_id") == attempt_id
             and item.get("producing_attempt_id", attempt_id) == producing_attempt_id
             and item.get("cache_retrieved", False) is (producing_attempt_id != attempt_id)
             and item.get("checkpoint_path") == str(checkpoint_path.relative_to(ROOT))
             and item.get("checkpoint_sha256") == digest
             and item.get("checkpoint_status") == record.get("status"),
             f"Checkpoint/receipt ID, status, path or hash mismatch: {unit}.")
        if producing_attempt_id != attempt_id:
            producer = receipts.get(producing_attempt_id)
            need(producer is not None and producer.get("unit") == unit
                 and producer.get("outcome") == "transport_or_parse_failed"
                 and producer.get("started_utc", "") <= item.get("started_utc", ""),
                 f"Cached checkpoint does not link to an earlier unresolved producer attempt: {unit}.")

    successful_units = set()
    for unit, record in checkpoint_records.items():
        attempt_id = record.get("attempt_id")
        need(attempt_id in receipts,
             f"Checkpoint has no matching attempt receipt: {unit}.")
        producing_attempt_id = record.get("producing_attempt_id", attempt_id)
        need(producing_attempt_id in receipts,
             f"Checkpoint's producing attempt has no receipt: {unit}.")
        receipt = receipts[attempt_id]
        if record.get("status") == "complete":
            need(receipt.get("unit") == unit
                 and receipt.get("outcome") == "checkpoint_saved"
                 and receipt.get("checkpoint_status") == "complete",
                 f"Complete checkpoint lacks a matching successful receipt: {unit}.")
            successes = [item for item in per_unit.get(unit, [])
                         if item.get("outcome") == "checkpoint_saved"]
            need(len(successes) == 1 and successes[0].get("attempt_id") == attempt_id,
                 f"Complete checkpoint has an ambiguous successful receipt: {unit}.")
            successful_units.add(unit)
        else:
            need(receipt.get("unit") == unit
                 and receipt.get("outcome") == "gate_failed"
                 and receipt.get("checkpoint_status") == record.get("status"),
                 f"Failed checkpoint lacks its matching blocking receipt: {unit}.")
    return {"receipt_count": len(receipts),
            "saved_checkpoint_receipts": len(checkpoint_records),
            "successful_checkpoint_receipts": len(successful_units),
            "submitted_by_unit": {unit: sum(item["outcome"] not in {"paused_before_submit", "local_launch_failed"}
                                            for item in items) for unit, items in per_unit.items()}}

def telemetry_summary(samples, phase, plan):
    good = [sample for sample in samples if "error" not in sample]
    phase_start = phase.get("phase_start_monotonic_ns")
    phase_end = phase.get("phase_end_monotonic_ns")
    need(integer(phase_start) and integer(phase_end) and phase_start < phase_end,
         "Phase monotonic boundaries are absent or invalid.")
    arrivals = [sample.get("arrival_monotonic_ns") for sample in samples]
    need(all(integer(t) for t in arrivals),
         "A telemetry row lacks its reader-arrival monotonic timestamp.")
    phase_start_utc = phase.get("phase_start_utc")
    phase_end_utc = phase.get("phase_end_utc")
    need(isinstance(phase_start_utc, str) and isinstance(phase_end_utc, str),
         "Phase UTC boundaries are absent.")
    phase_start_query_ns = utc_epoch_ns(phase_start_utc)
    phase_end_query_ns = utc_epoch_ns(phase_end_utc)
    need(phase_start_query_ns < phase_end_query_ns, "Phase UTC boundaries are invalid.")
    times = [sample.get("query_unix_ns") for sample in good
             if integer(sample.get("query_unix_ns"))]
    need(len(times) == len(good)
         and all(phase_start_query_ns <= value <= phase_end_query_ns for value in times),
         "A valid telemetry query timestamp is missing or outside the measured phase.")
    gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
    query_ordered = all(b > a for a, b in zip(times, times[1:]))
    start_gap = (times[0] - phase_start_query_ns) / 1e9 if times else None
    end_gap = (phase_end_query_ns - times[-1]) / 1e9 if times else None
    measured_gaps = gaps + ([start_gap, end_gap] if times else [])
    fields = {}
    coverage_pass = True
    for field in plan["telemetry_required_fields"]:
        values = [s.get(field) for s in good if finite_number(s.get(field))]
        coverage = len(values) / len(samples) if samples else 0.0
        passed = coverage >= plan["telemetry_required_field_coverage_fraction"]
        coverage_pass &= passed
        fields[field] = {"count": len(values), "coverage_fraction": coverage,
                         "median": median_or_none(values), "peak": max_or_none(values),
                         "coverage_passed": passed}
    max_gap = max(measured_gaps) if measured_gaps else None
    query_pass = (len(good) >= plan["minimum_active_telemetry_samples"]
                  and len(times) == len(good)
                  and query_ordered
                  and max_gap is not None
                  and max_gap <= plan["maximum_active_telemetry_gap_seconds"])
    device_mapping_verified = (phase.get("telemetry_device_verified") is True
                               and bool(phase.get("telemetry_device_mapping")))
    sampler_error_free = (phase.get("telemetry_start_error") is None
                          and phase.get("telemetry_reader_error") is None
                          and all(sample.get("error") is None for sample in samples))
    derived_gate_passed = bool(sampler_error_free and coverage_pass and query_pass
                               and device_mapping_verified)
    need(phase.get("telemetry_gate_passed") is derived_gate_passed,
         "Saved telemetry gate flag does not match raw samples and sampler state.")
    need(phase.get("activity_telemetry_eligible") is derived_gate_passed,
         "Saved activity-telemetry eligibility does not match audited raw evidence.")
    return {"poll_rows": len(samples), "valid_rows": len(good),
            "max_query_gap_seconds": max_gap,
            "max_internal_query_gap_seconds": max(gaps) if gaps else None,
            "phase_start_query_gap_seconds": start_gap,
            "phase_end_query_gap_seconds": end_gap,
            "time_basis": "nvidia-smi query timestamp (UTC)",
            "query_timestamps_ordered": query_ordered,
            "unique_reading_transitions": sum(
                any(a.get(f) != b.get(f) for f in plan["telemetry_required_fields"])
                for a, b in zip(good, good[1:])),
            "fields": fields,
            "coverage_gate_passed": bool(coverage_pass),
            "query_gate_passed": bool(query_pass),
            "device_mapping_verified": device_mapping_verified,
            "sampler_error_free": sampler_error_free,
            "eligible": derived_gate_passed}


def telemetry_only_failure(record, plan):
    try:
        failure = record["failure"]
        phase = record["sustained_phases"][0]
        if (record.get("status") != "gate_failed"
                or failure.get("stage") != "sustained_sync:triton_4w_4r"
                or len(record.get("sustained_phases", [])) != 1
                or phase.get("backend") != "triton_4w_4r"
                or phase.get("telemetry_gate_passed") is not False
                or phase.get("compute_gate_passed") is not True
                or phase.get("gate_passed") is not True
                or not finite_number(phase.get("telemetry_max_gap_seconds"))
                or phase["telemetry_max_gap_seconds"] <= plan["maximum_active_telemetry_gap_seconds"]):
            return False
        audit_sustained_phase(phase, record, plan, allow_telemetry_failure=True)
        telemetry = telemetry_summary(phase.get("telemetry_samples", []), phase, plan)
        if (not telemetry["coverage_gate_passed"]
                or telemetry["eligible"]
                or telemetry["valid_rows"] < plan["minimum_active_telemetry_samples"]
                or phase.get("telemetry_sample_count") != telemetry["poll_rows"]
                or not telemetry["query_timestamps_ordered"]
                or telemetry["max_query_gap_seconds"]
                   <= plan["maximum_active_telemetry_gap_seconds"]
                or not telemetry["device_mapping_verified"]
                or not telemetry["sampler_error_free"]):
            return False
        audit_correctness(record, plan, allow_partial_sustained=True)
        audit_latency_intervals(record, plan)
        audit_capture_memory(record, plan)
        return True
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def audit_latency_intervals(record, plan):
    intervals = record.get("latency_intervals", [])
    rounds = plan["latency_rounds"]
    need(isinstance(intervals, list) and len(intervals) == rounds * len(BACKENDS),
         f"Expected {rounds * len(BACKENDS)} latency intervals.")
    counts = {name: 0 for name in BACKENDS}
    by_backend = {name: [] for name in BACKENDS}
    expected_rng = random.Random(record["seed"] ^ (record["width"] << 8))
    expected_orders = []
    for _ in range(rounds):
        order = list(BACKENDS)
        expected_rng.shuffle(order)
        expected_orders.append(order)
    interval_orders = defaultdict(list)
    for index, item in enumerate(intervals):
        need(isinstance(item, dict), "Latency row must be an object.")
        name = item.get("backend")
        need(name in counts and type(item.get("ops_per_graph")) is int
             and item["ops_per_graph"] == plan["ops_per_graph"],
             "Unexpected latency row/backend.")
        round_id, order_index = item.get("round"), item.get("order_index")
        need(integer(round_id) and 0 <= round_id < rounds
             and integer(order_index) and 0 <= order_index < len(BACKENDS),
             "Latency round/order metadata is invalid.")
        expected_index = round_id * len(BACKENDS) + order_index
        need(index == expected_index and expected_orders[round_id][order_index] == name,
             "Latency rows do not match the frozen randomized round/order schedule.")
        elapsed = item.get("graph_elapsed_us")
        per_op = item.get("us_per_op")
        need(finite_number(elapsed) and elapsed > 0,
             "Latency graph duration must be finite and positive.")
        need(finite_number(per_op)
             and math.isclose(per_op, elapsed / plan["ops_per_graph"], rel_tol=1e-9, abs_tol=1e-12),
             "Latency per-op value does not match raw graph duration.")
        counts[name] += 1
        by_backend[name].append(elapsed)
        interval_orders[round_id].append(name)
    need(all(count == rounds for count in counts.values()), "Latency backend interval count mismatch.")
    need(set(interval_orders) == set(range(rounds))
         and all(len(order) == len(BACKENDS) and set(order) == set(BACKENDS)
                 for order in interval_orders.values()),
         "Latency round coverage/order is incomplete.")
    return {name: statistics.median(by_backend[name]) / plan["ops_per_graph"]
            for name in BACKENDS}


def audit_capture_memory(record, plan):
    memory = record.get("memory", {})
    for key in ("before_allocations", "after_persistent_allocation"):
        snapshot = memory.get(key, {})
        free = snapshot.get("free_bytes")
        need(integer(free) and free >= plan["free_memory_guard_bytes"],
             f"Free-memory floor failed at {key}.")
    before_latency = memory.get("before_latency", {})
    latency_start = record.get("latency_start_monotonic_ns")
    need(integer(latency_start)
         and integer(before_latency.get("captured_monotonic_ns"))
         and before_latency["captured_monotonic_ns"] < latency_start
         and integer(before_latency.get("free_bytes"))
         and before_latency["free_bytes"] >= plan["free_memory_guard_bytes"],
         "Free-memory floor was not verified immediately before latency measurement.")
    baseline = memory.get("capture_baseline", {})
    baseline_reserved = baseline.get("reserved_bytes")
    need(integer(baseline_reserved) and baseline_reserved >= 0,
         "Capture baseline reserved-memory sample is missing or invalid.")
    snapshots = memory.get("per_backend_capture", [])
    need(isinstance(snapshots, list) and len(snapshots) == len(BACKENDS),
         "Capture-memory snapshots do not cover every backend.")
    deltas = []
    seen = set()
    for snapshot in snapshots:
        need(isinstance(snapshot, dict), "Capture-memory snapshot must be an object.")
        backend = snapshot.get("backend")
        peak = snapshot.get("peak_reserved_bytes")
        saved_delta = snapshot.get("capture_reserved_delta_bytes")
        need(backend in BACKENDS and backend not in seen
             and integer(peak) and peak >= 0
             and integer(saved_delta) and saved_delta >= 0,
             "Capture-memory snapshot has invalid backend or byte counts.")
        seen.add(backend)
        recomputed_delta = max(0, peak - baseline_reserved)
        need(saved_delta == recomputed_delta,
             f"Saved capture-memory delta does not match raw snapshots: {backend}.")
        deltas.append(recomputed_delta)
    peak_delta = memory.get("capture_reserved_delta_peak_bytes")
    need(seen == set(BACKENDS) and integer(peak_delta)
         and peak_delta == max(deltas)
         and memory.get("capture_reserved_delta_ceiling_bytes")
             == plan["capture_reserved_delta_ceiling_bytes"]
         and peak_delta <= plan["capture_reserved_delta_ceiling_bytes"],
         "Aggregate capture-memory delta does not match per-backend snapshots or ceiling.")
    correctness_peak = memory.get("correctness_peak", {})
    need(integer(correctness_peak.get("peak_allocated_bytes"))
         and correctness_peak["peak_allocated_bytes"] > 0,
         "Measured correctness transient peak is missing.")


def audit_sustained_phase(phase, record, plan, allow_telemetry_failure=False):
    need(isinstance(phase, dict), "Sustained phase must be an object.")
    durations = phase.get("window_event_seconds", [])
    gaps = phase.get("window_event_gaps_seconds", [])
    replay_counts = phase.get("window_replay_counts", [])
    expected_windows = plan["sustained_event_windows_per_phase"]
    need(isinstance(durations, list) and len(durations) == expected_windows
         and isinstance(gaps, list) and len(gaps) == expected_windows - 1
         and isinstance(replay_counts, list) and len(replay_counts) == expected_windows,
         "Sustained phase window/replay count cardinality mismatch.")
    need(all(finite_number(value) and 0 < value <= plan["maximum_event_window_seconds"]
             for value in durations), "Invalid or overlong event window.")
    need(all(finite_number(gap) and 0 <= gap <= plan["maximum_event_gap_between_windows_seconds"]
             for gap in gaps), "CUDA event-window gap is negative or exceeds the registered limit.")
    event_seconds = phase.get("event_seconds")
    need(plan["sustained_min_seconds"] <= sum(durations) <= plan["sustained_max_seconds"]
         and finite_number(event_seconds)
         and abs(sum(durations) - event_seconds) < 1e-6,
         "Sustained event duration is outside the bound or not reproducible.")
    calls = phase.get("graph_calls_per_block")
    pilot_replays = phase.get("pilot_replays")
    pilot_seconds = phase.get("pilot_event_seconds")
    need(type(calls) is int and calls == plan["sustained_graph_calls_per_block"]
         and integer(pilot_replays) and pilot_replays > 0
         and finite_number(pilot_seconds) and pilot_seconds > 0,
         "Sustained replay calibration is missing or invalid.")
    expected_replays = math.ceil(plan["sustained_target_seconds_per_window"]
                                 / (pilot_seconds / pilot_replays))
    need(integer(phase.get("replays_per_window"))
         and phase["replays_per_window"] == expected_replays
         and all(integer(value) and value == expected_replays for value in replay_counts),
         "Every sustained window must match the replay count recalibrated from its pilot.")
    total_calls = sum(replay_counts) * calls
    raw_bytes = (plan["rows"] * record["width"] * plan["input_element_bytes"]
                 * 2 * total_calls)
    need(integer(phase.get("logical_input_plus_output_bytes"))
         and phase["logical_input_plus_output_bytes"] == raw_bytes,
         "Recomputed logical bytes mismatch.")
    rate = raw_bytes / sum(durations) / 1e9
    saved_rate = phase.get("logical_input_plus_output_GB_per_s")
    need(finite_number(saved_rate)
         and abs(rate - saved_rate) <= max(1e-9, rate * 1e-8),
         "Recomputed logical throughput mismatch.")
    phase_start = phase.get("phase_start_monotonic_ns")
    memory_before = phase.get("memory_before_phase", {})
    need(integer(phase_start) and integer(memory_before.get("captured_monotonic_ns"))
         and memory_before["captured_monotonic_ns"] < phase_start
         and integer(memory_before.get("free_bytes"))
         and memory_before["free_bytes"] >= plan["free_memory_guard_bytes"],
         "Pre-phase memory sample is invalid or below the registered floor.")
    expected_cadence = f"--loop-ms={round(plan['telemetry_interval_seconds'] * 1000)}"
    need(expected_cadence in phase.get("telemetry_process", ""),
         "Recorded nvidia-smi cadence does not match the sampling interval.")
    telemetry = telemetry_summary(phase.get("telemetry_samples", []), phase, plan)
    need(phase.get("telemetry_max_gap_seconds") == telemetry["max_query_gap_seconds"],
         "Saved telemetry maximum gap does not match raw query timestamps.")
    compute_pass = (sum(durations) >= plan["sustained_min_seconds"]
                    and sum(durations) <= plan["sustained_max_seconds"]
                    and max(durations) <= plan["maximum_event_window_seconds"]
                    and all(0 <= gap <= plan["maximum_event_gap_between_windows_seconds"]
                            for gap in gaps))
    need(phase.get("compute_gate_passed") is compute_pass
         and phase.get("gate_passed") is compute_pass,
         "Saved sustained compute gate flags do not match the measured windows.")
    need(phase.get("activity_telemetry_eligible") is telemetry["eligible"],
         "Saved activity eligibility does not match raw telemetry.")
    if not allow_telemetry_failure:
        need(phase.get("telemetry_gate_passed") is telemetry["eligible"],
             "Saved telemetry gate does not match raw telemetry.")
    return {"rate": rate, "event_seconds": sum(durations),
            "total_graph_blocks": sum(replay_counts), "total_graph_calls": total_calls,
            "window_replay_counts": replay_counts, "telemetry": telemetry}


def verify_source_hashes_at(revision, expected):
    need(len(revision) == 40, "Checkpoint has no full source revision.")
    for source, digest in expected.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{source}"])
        need(hashlib.sha256(blob).hexdigest() == digest,
             f"Checkpoint source hash mismatch at {revision}:{source}.")


def audit_driver_probes(record):
    probes = record.get("driver_probes")
    sources = [p.get("source") for p in probes] if isinstance(probes, list) else []
    need(sources in (["nvidia-smi"], ["nvidia-smi", "/sys/module/nvidia/version"]),
         "Missing or unexpected driver provenance queries.")
    for probe in probes:
        need(finite_number(probe.get("elapsed_seconds")) and probe["elapsed_seconds"] >= 0,
             "Driver query lacks an observed duration.")
    smi = probes[0]
    status = smi.get("status")
    need(status in {"ok", "timeout", "os_error", "nonzero_exit", "invalid_output"},
         "Unknown nvidia-smi driver-query outcome.")
    if status == "timeout":
        need(smi.get("timeout_seconds") == 5, "Driver timeout lacks its deadline.")
    elif status == "os_error":
        need(integer(smi.get("errno")), "Driver launch error lacks errno.")
    else:
        need(integer(smi.get("exit_code")) and (smi["exit_code"] != 0) == (status == "nonzero_exit")
             and integer(smi.get("rows")) and smi["rows"] >= int(status == "ok"),
             "Driver query exit status or row count is inconsistent.")
    if status != "os_error":
        for stream in ("stdout", "stderr"):
            need(integer(smi.get(f"{stream}_bytes")) and smi[f"{stream}_bytes"] >= 0
                 and re.fullmatch(r"[0-9a-f]{64}", smi.get(f"{stream}_sha256", "")),
                 "Driver query lacks its output length or digest.")
    need((status == "ok") == (len(probes) == 1), "Driver fallback order is inconsistent.")
    final = probes[-1]
    if len(probes) == 2:
        need(final.get("status") in {"ok", "os_error", "invalid_output"},
             "Unknown kernel-module version query outcome.")
        if final["status"] == "os_error":
            need(integer(final.get("errno")), "Kernel-module read error lacks errno.")
    if final.get("status") == "ok":
        version = final.get("version")
        need(isinstance(version, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version)
             and (record.get("environment") is None or record["environment"].get("driver") == version),
             "Recorded environment and driver-query result disagree.")
        return
    need(record.get("status") == "gate_failed" and record.get("failure", {}).get("stage") == "preflight"
         and not record.get("latency_intervals") and not record.get("sustained_phases")
         and not any(record.get("correctness", {}).values()),
         "A checkpoint without driver provenance reached GPU measurements.")


def audit_checkpoint(path, plan, pin):
    r = json.loads(path.read_text())
    need(path.stem == f"s{r['seed']}-w{r['width']}", f"Checkpoint identity mismatch: {path.name}.")
    need(r.get("rows") == plan["rows"],
         f"Checkpoint row count does not match the registered workload: {path.name}.")
    need(r.get("run_id") == pin.get("active_run_id"),
         f"Checkpoint does not match the active checkout run identity: {path.name}.")
    revision = r.get("source_revision", "")
    expected_sources = {key: r.get("source_sha256", {}).get(key) for key in COMPUTE_SOURCES}
    expected_execution = r.get("execution_source_sha256", {})
    if "campaign_id" in pin and (r.get("status") == "complete" or "driver_probes" in r
                                 or "driver provenance" in r.get("failure", {}).get("message", "")):
        audit_driver_probes(r)
    if r.get("status") != "complete":
        need(r.get("status") in {"gate_failed", "execution_failed"}
             and isinstance(r.get("failure"), dict),
             f"Saved incomplete checkpoint has an unknown status or lacks failure details: {path.name}.")
        need(revision == pin["source_revision"],
             f"Failed checkpoint revision mismatch: {path.name}.")
        need(set(expected_execution) == set(pin["source_sha256"])
             and all(isinstance(digest, str) and len(digest) == 64
                     for digest in expected_execution.values()),
             f"Failed checkpoint execution-source manifest is incomplete: {path.name}.")
        need(expected_execution == pin["source_sha256"],
             f"Failed checkpoint execution-source pin mismatch: {path.name}.")
        need({key: expected_execution.get(key) for key in COMPUTE_SOURCES}
             == {key: pin["source_sha256"][key] for key in COMPUTE_SOURCES},
             f"Failed checkpoint compute-source pin mismatch: {path.name}.")
        verify_source_hashes_at(revision, expected_execution)
        need({key: expected_execution.get(key) for key in COMPUTE_SOURCES} == expected_sources,
             f"Failed checkpoint source manifests disagree: {path.name}.")
        classification = ("telemetry_only" if telemetry_only_failure(r, plan)
                          else "compute_or_execution_failure")
        phases = r.get("sustained_phases", [])
        telemetry_details = []
        for phase in phases:
            samples = phase.get("telemetry_samples")
            if isinstance(samples, list) and phase.get("phase_start_monotonic_ns") is not None:
                try:
                    telemetry = telemetry_summary(samples, phase, plan)
                    telemetry_details.append({"backend": phase.get("backend"),
                                              "eligible": telemetry["eligible"],
                                              "max_query_gap_seconds": telemetry["max_query_gap_seconds"],
                                              "summary": telemetry})
                except (ValueError, KeyError, TypeError):
                    telemetry_details.append({"backend": phase.get("backend"),
                                              "eligible": False,
                                              "summary_error": "Raw telemetry did not pass its audit."})
        return {"checkpoint": path.stem, "seed": r["seed"], "width": r["width"],
                "status": r["status"], "attempt_id": r.get("attempt_id"),
                "producing_attempt_id": r.get("producing_attempt_id", r.get("attempt_id")),
                "run_id": r.get("run_id"),
                "failure": r.get("failure"), "failure_classification": classification,
                "raw_evidence_saved": True, "analysis_eligible": False,
                "telemetry_activity_eligible": False, "telemetry_phases": telemetry_details}
    need(revision == pin["source_revision"], f"Checkpoint revision mismatch: {path.name}.")
    expected_sources = {key: pin["source_sha256"][key] for key in COMPUTE_SOURCES}
    need(r.get("source_sha256") == expected_sources, f"Checkpoint compute-source hash mismatch: {path.name}.")
    need(r.get("execution_source_sha256") == pin["source_sha256"], f"Checkpoint execution-source hash mismatch: {path.name}.")
    latency_medians = audit_latency_intervals(r, plan)
    phases = r.get("sustained_phases", [])
    need(len(phases) == 4 and {x.get("backend") for x in phases} == set(BACKENDS),
         "Expected one sustained phase per backend.")
    phase_results = {}
    for phase in phases:
        audited = audit_sustained_phase(phase, r, plan)
        rate = audited["rate"]
        durations = phase["window_event_seconds"]
        replay_counts = audited["window_replay_counts"]
        total_blocks = audited["total_graph_blocks"]
        total_calls = audited["total_graph_calls"]
        telemetry = audited["telemetry"]
        recorded_telemetry_eligibility = phase.get("activity_telemetry_eligible")
        need(type(recorded_telemetry_eligibility) is bool,
             "Recorded activity-telemetry eligibility is missing.")
        phase_results[phase["backend"]] = {
            "logical_input_plus_output_GB_per_s": rate,
            "event_seconds": sum(durations), "total_graph_blocks": total_blocks,
            "total_graph_calls": total_calls, "window_replay_counts": replay_counts,
            "activity_telemetry_eligible": telemetry["eligible"],
            "recorded_activity_telemetry_eligible": recorded_telemetry_eligibility,
            "activity_eligibility_matches_audit": recorded_telemetry_eligibility is telemetry["eligible"],
            "telemetry": telemetry}
    audit_correctness(r, plan)
    audit_capture_memory(r, plan)
    environment = r.get("environment")
    environment_fields = ("gpu", "compute_capability", "total_memory_bytes", "sm_count",
                          "driver", "cuda_runtime", "torch", "triton", "python")
    need(isinstance(environment, dict)
         and all(field in environment for field in environment_fields)
         and "H100" in str(environment.get("gpu"))
         and environment.get("compute_capability") == [9, 0]
         and all(isinstance(environment.get(field), str) and environment[field]
                 for field in ("gpu", "driver", "cuda_runtime", "torch", "triton", "python"))
         and integer(environment.get("total_memory_bytes"))
         and integer(environment.get("sm_count")),
         "Complete checkpoint lacks a verifiable H100/software environment.")
    return {"checkpoint": path.stem, "seed": r["seed"], "width": r["width"], "status": "complete",
            "attempt_id": r.get("attempt_id"),
            "producing_attempt_id": r.get("producing_attempt_id", r.get("attempt_id")),
            "run_id": r.get("run_id"),
            "environment": {field: environment[field] for field in environment_fields},
            "latency_median_us_per_op": latency_medians, "phases": phase_results}


def compare_checkpoints(results, plan):
    by_unit = {item["checkpoint"]: item for item in results if item.get("status") == "complete"}
    environments = {json.dumps(item["environment"], sort_keys=True, separators=(",", ":"))
                    for item in by_unit.values()}
    need(len(environments) <= 1,
         "Checkpoint matrix mixes H100 variants or software environments; refusing pooled comparisons.")
    execution_environment = (next(iter(by_unit.values()))["environment"] if by_unit else None)
    by_width = {}
    checkpoint_decisions = []
    threshold = 5.0
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
    pattern = plan["selection"]["dispatch_pattern"]
    odd_all_qualify = all(by_width.get(str(width), {}).get("all_registered_seeds_qualify", False)
                          for width in pattern["odd_widths"])
    even_any_qualify = any(by_width.get(str(width), {}).get("all_registered_seeds_qualify", False)
                           for width in pattern["even_widths"])
    bounded_pattern_supported = complete and odd_all_qualify and not even_any_qualify
    return {"checkpoint_comparisons": checkpoint_decisions, "per_width_across_seeds": by_width,
            "execution_environment": execution_environment,
            "bounded_dispatch_screen_decision": {
                "eligible": complete,
                "supports_registered_odd_even_split": bounded_pattern_supported,
                "all_registered_odd_widths_qualify": odd_all_qualify,
                "any_registered_even_width_qualifies": even_any_qualify,
                "scope": pattern["interpretation_scope"],
                "rule": f"all registered odd widths must be at least {threshold:.1f}% faster than fastest one-row Triton and torch_compile in every seed, and no registered even width may qualify",
                "reason": "bounded pattern reproduced in the complete matrix" if bounded_pattern_supported else
                          ("awaiting all registered checkpoints" if not complete else
                           "the preregistered bounded pattern was not reproduced")}}


def read_cohort(raw, plan, pin, expected):
    records = {}
    for path in sorted((raw / "checkpoints").glob("*.json")):
        need(path.stem in expected, f"Unexpected checkpoint file: {path.name}.")
        record = audit_checkpoint(path, plan, pin)
        record.update(source_revision=pin["source_revision"], raw_path=str(path.relative_to(ROOT)))
        records[path.stem] = record
    return records, audit_attempts(pin, records, plan, raw)


def verified_predecessor(plan, pin):
    """Re-audit the published prefix at its original source, including its stopped attempt."""
    rel = RAW.relative_to(ROOT)
    paths = [str(rel / name) for name in ("checkpoints", "attempts", "source-pin.json", "partial-audit.json")]
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only",
                                       EVIDENCE_COMMIT, "--", *paths], text=True).splitlines()
    published = {Path(name).relative_to(rel) for name in tracked}
    archived = {path.relative_to(PREDECESSOR) for path in PREDECESSOR.rglob("*.json")}
    need(published == archived and published, "Preflight archive differs from the published file set.")
    for name in sorted(published):
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{EVIDENCE_COMMIT}:{rel / name}"])
        need((PREDECESSOR / name).read_bytes() == blob, f"Archived evidence differs from Git: {name}.")
    previous = json.loads((PREDECESSOR / "source-pin.json").read_text())
    stopped = json.loads((PREDECESSOR / "checkpoints" / f"{RESUME_UNIT}.json").read_text())
    previous["active_run_id"] = stopped["run_id"]
    need(previous["active_run_id"] != pin["active_run_id"], "Continuation reuses the predecessor cache ID.")
    # Only this reviewed change to the benchmark's pre-allocation driver read is compatible.
    run_path = "experiments/width-dispatch/run.py"
    need(pin["source_sha256"][run_path] == RESUMED_DRIVER_RUNNER, "Continuation benchmark changed.")
    for name in (*COMPUTE_SOURCES, "lab.py", "experiments/width-dispatch/ssh_runner.sh"):
        if name != run_path:
            need(previous["source_sha256"][name] == pin["source_sha256"][name],
                 f"Cannot carry predecessor measurements across changed source: {name}.")
    records, receipts = read_cohort(PREDECESSOR, plan, previous, CARRIED_UNITS | {RESUME_UNIT})
    need(set(records) == CARRIED_UNITS | {RESUME_UNIT}
         and all(records[unit]["status"] == "complete" for unit in CARRIED_UNITS),
         "The two carried measurements must pass their original evidence gates.")
    failure = records.pop(RESUME_UNIT)
    need(failure["status"] == "gate_failed" and failure["failure"] == {
        "stage": "preflight", "type": "GateFailure",
        "message": "Cannot establish required NVIDIA driver provenance before measurement."}
         and stopped["latency_intervals"] == [] and stopped["sustained_phases"] == []
         and not any(stopped["correctness"].values())
         and all(stopped["memory"]["final"][key] == 0 for key in ("peak_allocated_bytes", "peak_reserved_bytes")),
         "Only the published driver failure before measurement can be reopened.")
    return records, {"evidence_commit": EVIDENCE_COMMIT, "source_revision": previous["source_revision"],
                     "run_id": previous["active_run_id"], "carried_units": sorted(records),
                     "archived_failure": failure, "receipt_audit": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true", help="Audit frozen plan and resource/replay arithmetic.")
    parser.add_argument("--audit", action="store_true", help="Audit all currently saved checkpoints and receipts; partial is allowed.")
    parser.add_argument("--final", action="store_true", help="Require and audit all registered checkpoints.")
    args = parser.parse_args()
    plan, estimate = plan_audit()
    report = {"plan": "pass", "resource_estimate": estimate}
    if args.audit or args.final:
        pin = check_pin()
        order = [f"s{s}-w{w}" for s in plan["seeds"] for w in plan["widths"]]
        expected = set(order)
        records, predecessor = verified_predecessor(plan, pin)
        current, receipt_audit = read_cohort(RAW, plan, pin, expected - records.keys())
        records.update(current)
        submitted = dict(predecessor["receipt_audit"]["submitted_by_unit"])
        first_pending = next((i for i, unit in enumerate(order)
                              if records.get(unit, {}).get("status") != "complete"), len(order))
        for unit, count in receipt_audit["submitted_by_unit"].items():
            need(unit not in CARRIED_UNITS and unit in order[:first_pending + 1],
                 f"Continuation repeated a completed unit or skipped pending work: {unit}.")
            submitted[unit] = submitted.get(unit, 0) + count
            need(submitted[unit] <= plan["max_attempts_per_checkpoint"],
                 f"Cross-campaign attempt limit exceeded: {unit}.")
        comparisons = compare_checkpoints(list(records.values()), plan)
        complete = (set(records) == expected
                    and all(item.get("status") == "complete" for item in records.values()))
        report.update(source_revision=pin["source_revision"], source_pin="pass",
                      saved_checkpoints=len(records), successful_checkpoints=sum(
                          item.get("status") == "complete" for item in records.values()),
                      expected_checkpoints=len(expected), final_complete=complete, receipt_audit=receipt_audit,
                      predecessor=predecessor, submitted_by_unit=submitted,
                      checkpoint_results=list(records.values()), **comparisons)
        print(json.dumps(report, indent=2, allow_nan=False))
        need(not args.final or complete, f"Final audit requires all {len(expected)} registered checkpoints.")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
