"""Qualify equal-work energy samples whose reported times bracket GPU work."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import re
import time

import energy_counter_boundaries_run as boundaries
import four_warp_energy_run as energy
import run

ROWS, WIDTH = energy.ROWS, energy.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 201, (202, 203, 204)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
ANCHOR = energy.ANCHOR
COUNTER_FUNCTION = "nvmlDeviceGetFieldValues"
ENERGY_FIELD = 83
NS_PER_S = 1_000_000_000
MAX_PADDING_NS = int(boundaries.MAX_PADDING_S * NS_PER_S)
MAX_CLOCK_OFFSET_CHANGE_NS = 1_000_000
UNSIGNED_FIELDS = {1: "uiVal", 2: "ulVal", 3: "ullVal"}
SOURCES = (*boundaries.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    Path(__file__).resolve(), run.HERE / "COPY_FREE_TIMESTAMPED_NVML_ENERGY_512.md",
)))


class NvmlValue(C.Union):
    _fields_ = [("dVal", C.c_double), ("siVal", C.c_int), ("uiVal", C.c_uint), ("ulVal", C.c_ulong),
                ("ullVal", C.c_ulonglong), ("sllVal", C.c_longlong)]


class NvmlFieldValue(C.Structure):
    _fields_ = [("fieldId", C.c_uint), ("scopeId", C.c_uint), ("timestamp", C.c_longlong),
                ("latencyUsec", C.c_longlong), ("valueType", C.c_int), ("nvmlReturn", C.c_int), ("value", NvmlValue)]


def clock_sandwich():
    before = time.perf_counter_ns()
    realtime = time.time_ns()
    return {"monotonic_before_ns": before, "realtime_ns": realtime, "monotonic_after_ns": time.perf_counter_ns()}


def reading_status(api_status, field_status, value_type):
    if api_status != 0:
        return api_status
    if field_status != 0:
        return field_status
    return 0 if value_type in UNSIGNED_FIELDS else "unexpected_value_type"


def field_reading(sensor, origin):
    value = NvmlFieldValue(fieldId=ENERGY_FIELD)
    try:
        query = sensor.function(COUNTER_FUNCTION, C.c_void_p, C.c_int, C.POINTER(NvmlFieldValue))
    except AttributeError:
        query = None
    before = clock_sandwich()
    query_start = time.perf_counter_ns()
    api_status = "missing_symbol" if query is None else int(query(sensor.handle, 1, C.byref(value)))
    query_end = time.perf_counter_ns()
    after = clock_sandwich()
    field_status = value.nvmlReturn if api_status == 0 else None
    value_type = value.valueType if api_status == 0 else None
    status = reading_status(api_status, field_status, value_type)
    return {"field_id": ENERGY_FIELD, "api_status": api_status, "field_status": field_status,
            "value_type": value_type, "status": status,
            "millijoules": getattr(value.value, UNSIGNED_FIELDS[value_type]) if status == 0 else None,
            "sample_epoch_us": value.timestamp if api_status == 0 else None,
            "driver_latency_us": value.latencyUsec if api_status == 0 else None,
            "query_start_ns": query_start, "query_end_ns": query_end,
            "query_start_s": query_start / NS_PER_S - origin, "query_end_s": query_end / NS_PER_S - origin,
            "before_clock": before, "after_clock": after}


def sample_bounds(work, before, after):
    result = dict.fromkeys(("clock_offset_ns", "opening_sample_ns", "closing_sample_ns", "padding_upper_ns"))
    if before["status"] != 0 or after["status"] != 0:
        return {**result, "status": "field_unavailable"}
    clocks = [value[side] for value in (before, after) for side in ("before_clock", "after_clock")]
    if any(a["realtime_ns"] >= b["realtime_ns"] or a["monotonic_after_ns"] > b["monotonic_before_ns"]
           for a, b in zip(clocks, clocks[1:])):
        return {**result, "status": "inconsistent_clock_order"}
    lows = [clock["realtime_ns"] - clock["monotonic_after_ns"] for clock in clocks]
    highs = [clock["realtime_ns"] - clock["monotonic_before_ns"] for clock in clocks]
    low, high = min(lows), max(highs)
    result["clock_offset_ns"] = [low, high]
    if max(lows) - min(highs) > MAX_CLOCK_OFFSET_CHANGE_NS:
        return {**result, "status": "inconsistent_clock_offset"}
    if not 0 < before["sample_epoch_us"] < after["sample_epoch_us"]:
        return {**result, "status": "sample_time_not_advancing"}
    opening, closing = [
        [reading["sample_epoch_us"] * 1000 - high, reading["sample_epoch_us"] * 1000 + 999 - low]
        for reading in (before, after)
    ]
    result.update(opening_sample_ns=opening, closing_sample_ns=closing)
    if opening[0] > before["query_end_ns"] or closing[0] > after["query_end_ns"]:
        return {**result, "status": "sample_after_query"}
    start, end = work["start_monotonic_ns"], work["end_monotonic_ns"]
    if not opening[1] <= start <= end <= closing[0]:
        return {**result, "status": "sample_does_not_bracket_work"}
    result["padding_upper_ns"] = start - opening[0] + closing[1] - end
    if result["padding_upper_ns"] > MAX_PADDING_NS:
        return {**result, "status": "padding_exceeds_limit"}
    if energy.energy_delta(before, after) is None:
        return {**result, "status": "counter_not_advancing"}
    return {**result, "status": "qualified"}


def fixed_window(runner, graphs, sensor, origin, bursts, *, before=None):
    if before is None:
        before = field_reading(sensor, origin)
    work = energy.base.collect_window(runner, graphs, 0.0, origin, rows=ROWS, fixed_bursts=bursts, retain_clock=True)
    after = field_reading(sensor, origin)
    delta = energy.energy_delta(before, after)
    work.update(energy_start=before, energy_end=after, energy_mj=delta,
                joules_per_billion_rows=None if delta is None else delta * 1_000_000 / work["normalized_rows"],
                sample_bounds=sample_bounds(work, before, after))
    return work


def probe_counter(sensor, runner, graphs):
    utc, origin = run.original.utc_now(), time.perf_counter()
    first = field_reading(sensor, origin)
    window = (fixed_window(runner, graphs, sensor, origin, energy.PROBE_BURSTS, before=first)
              if first["status"] == 0 else None)
    return {"observer_origin_utc": utc, "monotonic_origin_s": origin, "first": first, "window": window}


def counter_passed(probe):
    return bool(probe and probe["window"] and probe["window"]["sample_bounds"]["status"] == "qualified")


def pair_eligible(blocks):
    return energy.monotonic_counter(blocks) and all(block["sample_bounds"]["status"] == "qualified" for block in blocks)


def measure(torch, seed, verified, sensor, cuda_uuid):
    return boundaries.measure(torch, seed, verified, sensor, cuda_uuid,
                              measure_window=fixed_window, qualifies_pair=pair_eligible)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--qualified", type=Path)
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide a frozen Git revision.")
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    result = energy.checkpoint(args.seed, args.asset_dir, qualification, revision, numeric_seed=NUMERIC_SEED,
                               timing_seeds=TIMING_SEEDS, source_paths=SOURCES, measure_work=measure,
                               probe_work=probe_counter, qualifies_counter=counter_passed,
                               counter_function=COUNTER_FUNCTION)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
