"""Check fixed-work energy windows, output bits and the per-seed decision."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import copy_free_run as copy_free
from copy_free_report import GUARDS, verify_supplemental_samples
import four_warp_energy_run as protocol
from four_warp_energy_run import boundary_padding_seconds
import run
from split_schedule_report import verify_attempt_clocks
from sustained_report import verify_measurement, verify_numerical, verify_recovery, verify_window
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "four-warp-energy-512-results"


def verify_reading(reading):
    if set(reading) != {"query_start_s", "query_end_s", "status", "millijoules"}:
        raise ValueError("A counter endpoint is missing its raw status, value or timestamps.")
    start, end = reading["query_start_s"], reading["query_end_s"]
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        raise ValueError("The counter query has invalid monotonic timestamps.")
    status, value = reading["status"], reading["millijoules"]
    if not (type(status) is int and status >= 0 or status == "missing_symbol"):
        raise ValueError("The counter query status is invalid.")
    if status == 0:
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("The cumulative reading is outside the NVML counter's range.")
    elif value is not None:
        raise ValueError("A failed counter query must keep the value missing.")


def verify_energy_window(window, bursts):
    verify_window(window, 0.0, rows=protocol.ROWS, fixed_bursts=bursts)
    before, after = window["energy_start"], window["energy_end"]
    verify_reading(before)
    verify_reading(after)
    if not before["query_end_s"] <= window["start_s"] <= window["end_s"] <= after["query_start_s"]:
        raise ValueError("Cumulative energy must bracket all measured work.")
    raw = protocol.energy_delta(before, after)
    rate = None if raw is None else raw * 1_000_000 / window["normalized_rows"]
    if (window["energy_mj"], window["joules_per_billion_rows"]) != (raw, rate):
        raise ValueError("The reported energy differs from the raw endpoints and completed rows.")


def verify_probe(probe, error):
    if probe is None:
        if error not in ("OSError", "AttributeError", "RuntimeError", "ValueError"):
            raise ValueError("An unavailable counter must archive its initialization error.")
        return
    if error is not None:
        raise ValueError("The counter probe cannot have an initialization error.")
    timestamp(probe["observer_origin_utc"])
    first, window = probe["first"], probe["window"]
    verify_reading(first)
    if (window is not None) != (first["status"] == 0):
        raise ValueError("Probe work differs from the first reading's availability.")
    if window is not None:
        if window["energy_start"] != first:
            raise ValueError("The probe's initial reading differs.")
        verify_energy_window(window, protocol.PROBE_BURSTS)


def verify_timing(seed, timing):
    verify_measurement(seed, timing, backends=protocol.BACKENDS, rows=protocol.ROWS,
                       warmup_s=protocol.WARMUP_S, fixed_bursts=protocol.MEASURED_BURSTS)
    preceding_query_end = 0
    for block in timing["blocks"]:
        verify_energy_window(block, protocol.MEASURED_BURSTS)
        if not preceding_query_end <= block["warmup"]["start_s"] < block["warmup"]["end_s"] <= block["energy_start"]["query_start_s"]:
            raise ValueError("Warm-up overlaps a cumulative energy interval.")
        preceding_query_end = block["energy_end"]["query_end_s"]
    if (timing["median_joules_per_billion_rows"] != protocol.median_energy(timing["blocks"])
            or timing["monotonic_counter"] != protocol.monotonic_counter(timing["blocks"])):
        raise ValueError("The energy medians or monotonic decision differ from the endpoints.")


def verify_device_record(record, seed, anchor, sources, revision, *, expected_counter_function):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["source_sha256"] != sources or record["code_revision"] != revision
            or record["environment"] != anchor["environment"] or record["asset"] != anchor["asset"]
            or record["counter_function"] != expected_counter_function):
        raise ValueError("The shape, source, energy counter or H100 differs.")
    four = protocol.four
    four.verify_codegen(record["codegen"], anchor)
    if four.identities(record["codegen"]) != four.identities(anchor["codegen"]):
        raise ValueError("A fixed-work kernel differs from the recorded four-warp comparison.")
    if record["graph_contract"] != copy_free.graph_contract(protocol.ROWS, backends=four.VERIFICATION_BACKENDS):
        raise ValueError("Unexpected benchmark graph work or restoration copies.")
    numerical = record["numerical"]
    digests = verify_numerical(numerical, four.VERIFICATION_BACKENDS, rows=protocol.ROWS)
    if any(set(guards) != GUARDS for guards in numerical["alias_guards"].values()):
        raise ValueError("A registered output alias guard is missing.")
    if record["numerically_exact"] != copy_free.numeric_gate(numerical, four.VERIFICATION_BACKENDS):
        raise ValueError("The numerical decision differs from the fixtures and graph outputs.")
    return digests


def verify_record(record, seed, anchor, sources, revision, *, timing_seeds=protocol.TIMING_SEEDS,
                  verify_timing_fn=verify_timing, verify_probe_fn=verify_probe,
                  qualifies_counter=protocol.counter_passed, expected_counter_function=protocol.COUNTER_FUNCTION):
    digests = verify_device_record(record, seed, anchor, sources, revision,
                                   expected_counter_function=expected_counter_function)
    probe = record["counter_probe"]
    verify_probe_fn(probe, record["counter_error"])
    eligible = record["numerically_exact"] and qualifies_counter(probe)
    timing = record["timing"]
    if record["gate_passed"] != eligible or (timing is not None) != (eligible and seed in timing_seeds):
        raise ValueError("Exactness, counter qualification or timing eligibility differs.")
    if timing:
        verify_timing_fn(seed, timing)
    if record["effect"] != protocol.comparison(timing):
        raise ValueError("The energy or throughput decision differs from the measured work.")
    return digests


def read_results(folder=RESULTS):
    records = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    first = records[protocol.NUMERIC_SEED]
    expected = protocol.SEEDS if first["gate_passed"] else (protocol.NUMERIC_SEED,)
    if records.keys() != set(expected):
        raise ValueError("The archive differs from the registered seeds.")
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The frozen source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor_bytes = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The frozen four-warp checkpoint differs.")
    anchor = json.loads(anchor_bytes)
    seen = set()
    for seed, record in sorted(records.items()):
        digests = verify_record(record, seed, anchor, sources, revision)
        if seen & digests:
            raise ValueError("Independent seeds share input digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, records, earlier_clock=folder / "clock-probe-before.json")
    verify_supplemental_samples(folder)
    timing_present = any(record["timing"] for record in records.values())
    verify_recovery(records, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision,
                    require_timing=timing_present)
    return records


def adopted(records):
    effects = [records.get(seed, {}).get("effect") for seed in protocol.TIMING_SEEDS]
    return all(effect and effect["passes_energy"] and effect["passes_throughput"] for effect in effects)


def energy_saving_sensitivity(timing, padding_watts):
    """Apply an assumed maximum power to unassigned time at the two boundaries."""
    gross = timing["median_joules_per_billion_rows"]
    adjusted = {
        backend: statistics.median(
            (block["energy_mj"] / 1000 - padding_watts * boundary_padding_seconds(block))
            * 1_000_000_000 / block["normalized_rows"]
            for block in timing["blocks"] if block["backend"] == backend
        ) for backend in protocol.BACKENDS
    }
    if min(adjusted.values()) <= 0:
        raise ValueError("The assumed boundary energy exceeds a measured window.")
    return (1 - gross[protocol.CANDIDATE] / adjusted[protocol.BASELINE],
            1 - adjusted[protocol.CANDIDATE] / gross[protocol.BASELINE])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    records = read_results(cli.parse_args().directory)
    for seed, record in sorted(records.items()):
        print(f"{seed}: exact={record['numerically_exact']} counter_ready={protocol.counter_passed(record['counter_probe'])}")
        if effect := record["effect"]:
            print(f"  energy saving: {100*effect['energy_saving']:+.4f}%")
            print(f"  throughput change: {100*effect['throughput_change']:+.4f}%")
            timing = record["timing"]
            maximum_padding = max(boundary_padding_seconds(block) for block in timing["blocks"])
            low, high = energy_saving_sensitivity(timing, 100)
            print(f"  largest boundary padding: {maximum_padding:.6f} s")
            print(f"  assumed 100 W padding sensitivity: {100*low:+.4f}% to {100*high:+.4f}%")
    print("Registered gross-window criteria met:", adopted(records))
