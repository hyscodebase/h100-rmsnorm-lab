"""Verify the recorded NVML field data and their relation to GPU work."""
import argparse
import math
from pathlib import Path

import energy_counter_boundaries_report as boundaries
import four_warp_energy_report as previous
import run
from sustained_report import verify_window
import timestamped_nvml_energy_run as protocol
from transfer_report import timestamp

RESULTS = run.HERE / "timestamped-nvml-energy-512-results"


def verify_reading(reading, origin):
    if reading["field_id"] != protocol.ENERGY_FIELD or not math.isfinite(origin) or origin < 0:
        raise ValueError("The field ID or monotonic origin differs.")
    before, after = reading["before_clock"], reading["after_clock"]
    for clock in (before, after):
        if set(clock) != {"monotonic_before_ns", "realtime_ns", "monotonic_after_ns"} or any(
            type(value) is not int or value <= 0 for value in clock.values()
        ) or clock["monotonic_before_ns"] > clock["monotonic_after_ns"]:
            raise ValueError("Incomplete or invalid realtime/monotonic calibration.")
    start, end = reading["query_start_ns"], reading["query_end_ns"]
    if not (type(start) is type(end) is int and before["monotonic_after_ns"] <= start <= end
            <= after["monotonic_before_ns"]):
        raise ValueError("The query falls outside its monotonic calibration brackets.")
    if (reading["query_start_s"], reading["query_end_s"]) != (
        start / protocol.NS_PER_S - origin, end / protocol.NS_PER_S - origin
    ) or reading["query_start_s"] < 0:
        raise ValueError("The relative query times differ from the monotonic clock.")
    api = reading["api_status"]
    if not (type(api) is int and api >= 0 or api == "missing_symbol"):
        raise ValueError("Invalid NVML API status.")
    field = reading["field_status"]
    value_type = reading["value_type"]
    metadata = (field, value_type, reading["sample_epoch_us"], reading["driver_latency_us"])
    if api == 0:
        if any(type(value) is not int for value in metadata) or field < 0:
            raise ValueError("A successful API query must retain its per-field results.")
    elif any(value is not None for value in metadata):
        raise ValueError("A failed API query must leave its per-field results missing.")
    status = protocol.reading_status(api, field, value_type)
    if reading["status"] != status:
        raise ValueError("A field's result differs from the recorded API status, field status or type.")
    value = reading["millijoules"]
    if status == 0:
        bits = 32 if value_type == 1 else 64
        if type(value) is not int or not 0 <= value < 2**bits:
            raise ValueError("The millijoule reading lies outside the unsigned NVML field type.")
    elif value is not None:
        raise ValueError("An unsuccessful field query must leave the counter value missing.")


def verify_energy_window(window, bursts, *, max_padding_ns=protocol.MAX_PADDING_NS):
    verify_window(window, 0.0, rows=protocol.ROWS, fixed_bursts=bursts)
    origin, before, after = window["monotonic_origin_s"], window["energy_start"], window["energy_end"]
    verify_reading(before, origin)
    verify_reading(after, origin)
    start, end = window["start_monotonic_ns"], window["end_monotonic_ns"]
    if not (type(start) is type(end) is int and before["after_clock"]["monotonic_after_ns"]
            <= start < end <= after["before_clock"]["monotonic_before_ns"]):
        raise ValueError("The calibration or query brackets overlap completed work.")
    if (window["start_s"], window["end_s"]) != (start / protocol.NS_PER_S - origin, end / protocol.NS_PER_S - origin):
        raise ValueError("The relative work bounds differ from the monotonic clock.")
    delta = protocol.energy.energy_delta(before, after)
    rate = None if delta is None else delta * 1_000_000 / window["normalized_rows"]
    if (window["energy_mj"], window["joules_per_billion_rows"]) != (delta, rate):
        raise ValueError("Reported energy differs from the raw field values and completed work.")
    if window["sample_bounds"] != protocol.sample_bounds(window, before, after, max_padding_ns=max_padding_ns):
        raise ValueError("The reported bounds differ from the raw field timestamps and clock calibration.")


def verify_probe(probe, error, *, max_padding_ns=protocol.MAX_PADDING_NS):
    if probe is None:
        previous.verify_probe(probe, error)
        return
    if error is not None:
        raise ValueError("A saved timestamped probe cannot have an initialization error.")
    timestamp(probe["observer_origin_utc"])
    first, window = probe["first"], probe["window"]
    verify_reading(first, probe["monotonic_origin_s"])
    if (window is not None) != (first["status"] == 0):
        raise ValueError("Probe work differs from the first reading's availability.")
    if window is not None:
        if window["energy_start"] != first or window["monotonic_origin_s"] != probe["monotonic_origin_s"]:
            raise ValueError("The probe's initial field query or clock origin differs.")
        verify_energy_window(window, protocol.energy.PROBE_BURSTS, max_padding_ns=max_padding_ns)


def verify_timing(seed, timing):
    if len({b["monotonic_origin_s"] for p in timing["pairs"] for b in p["blocks"]}) != 1:
        raise ValueError("All attempted periods must share the same monotonic origin.")
    boundaries.verify_timing(seed, timing, verify_energy_window=verify_energy_window,
                             qualifies_pair=protocol.pair_eligible)


def verify_record(record, seed, anchor, sources, revision):
    return previous.verify_record(record, seed, anchor, sources, revision, timing_seeds=protocol.TIMING_SEEDS,
                                  verify_timing_fn=verify_timing, verify_probe_fn=verify_probe,
                                  qualifies_counter=protocol.counter_passed,
                                  expected_counter_function=protocol.COUNTER_FUNCTION)


def read_results(folder=RESULTS):
    return boundaries.read_results(folder, campaign=protocol, verify=verify_record)


def probe_status(probe, error):
    if probe is None:
        return error
    if probe["window"] is None:
        return f"field status {probe['first']['status']}"
    return probe["window"]["sample_bounds"]["status"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    records = read_results(cli.parse_args().directory)
    for seed, record in sorted(records.items()):
        status = probe_status(record["counter_probe"], record["counter_error"])
        print(f"{seed}: exact={record['numerically_exact']} probe={status}")
        if timing := record["timing"]:
            print(f"  accepted pairs: {len(timing['accepted_pair_indices'])}/2, attempted: {len(timing['pairs'])}")
        if effect := record["effect"]:
            print(f"  energy saving: {100*effect['energy_saving']:+.4f}%")
            print(f"  throughput change: {100*effect['throughput_change']:+.4f}%")
    print("Registered energy criteria met:", boundaries.adopted(records, protocol.TIMING_SEEDS))
