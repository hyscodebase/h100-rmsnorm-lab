"""Verify resumable H100 periods and bound energy savings under stated power assumptions."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import statistics

import energy_counter_boundaries_report as boundaries
import four_warp_energy_report as previous
import long_window_energy_run as protocol
import run
from split_schedule_report import clock_bounds
import timestamped_nvml_energy_report as field_report
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "long-window-energy-512-results"
POWER_ASSUMPTIONS_W = (700, 1000)
BASELINE, CANDIDATE = protocol.energy.BACKENDS
BACKENDS = (BASELINE, CANDIDATE)
COUNTER_ERRORS = {"OSError", "AttributeError", "RuntimeError", "ValueError"}


def verify_record(unit, record, anchor, sources, revision, seed_reference):
    unit.verify_identity(record)
    digests = previous.verify_device_record(
        record, unit.seed, anchor, sources, revision, expected_counter_function=protocol.COUNTER_FUNCTION,
    )
    start, end, elapsed = timestamp(record["started_utc"]), timestamp(record["finished_utc"]), record["wall_seconds"]
    if not start <= end or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid device checkpoint duration.")
    probe, error, window = record["counter_probe"], record["counter_error"], record["window"]
    if error is not None and error not in COUNTER_ERRORS:
        raise ValueError("Unknown NVML initialization error.")
    if unit.index is None:
        field_report.verify_probe(probe, error, max_padding_ns=None)
        if record["gate_passed"] != (record["numerically_exact"] and protocol.field.counter_passed(probe)):
            raise ValueError("The qualification differs from the numerical and field evidence.")
        if window is not None or record["seed_fingerprints_match"] is not None:
            raise ValueError("A qualification cannot have a timing period or seed reference.")
    else:
        if probe is not None:
            raise ValueError("A timing period must keep its own measured endpoints.")
        expected_match = None if seed_reference is None else (
            seed_reference["numerical"]["source_fingerprints"] == record["numerical"]["source_fingerprints"]
        )
        if record["seed_fingerprints_match"] is not expected_match:
            raise ValueError("The seed fingerprint decision differs from the saved inputs.")
        if (window is not None) != (record["numerically_exact"] and expected_match is not False and error is None):
            raise ValueError("The existence of a timing period differs from the registered gate.")
    post_fingerprints = record["post_input_fingerprints"]
    if window is None:
        if post_fingerprints is not None:
            raise ValueError("The final input digests require a completed work period.")
        return digests
    verify_window = partial(field_report.verify_energy_window, max_padding_ns=None)
    boundaries.verify_period(window, protocol.MEASURED_BURSTS, verify_window)
    if window["backend"] != unit.period["backend"] or window["auxiliary_after"]["query_end_s"] >= elapsed:
        raise ValueError("Unexpected measured backend or duration outside the device checkpoint.")
    epochs = [start, timestamp(window["auxiliary_before"]["utc"]), timestamp(window["warmup"]["started_utc"]),
              timestamp(window["warmup"]["finished_utc"]), timestamp(window["started_utc"]),
              timestamp(window["finished_utc"]), timestamp(window["auxiliary_after"]["utc"]), end]
    if epochs != sorted(epochs):
        raise ValueError("The period's wall-clock events are out of order.")
    # The mathematical decision retains invalid exactness results when a digest changes.
    if not isinstance(post_fingerprints, dict) or post_fingerprints.keys() != record["numerical"]["source_fingerprints"].keys():
        raise ValueError("The period is missing final input or weight digests.")
    return digests


def read_verified(directory):
    records = protocol.read_records(directory)
    protocol.next_work(records)
    if not records:
        return records
    first = records[f"s{protocol.NUMERIC_SEED}"]
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The registered source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor_bytes = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The local kernel anchor differs from the pinned source.")
    anchor = json.loads(anchor_bytes)
    seen = set()
    previous_finish = None
    for unit in protocol.ordered_work(records):
        if unit.name not in records:
            break
        record = records[unit.name]
        seed_reference = records[f"s{unit.seed}-p0"] if unit.index is not None and unit.index > 0 else None
        digests = verify_record(unit, record, anchor, sources, revision, seed_reference)
        if unit.index in (None, 0):
            if seen & digests:
                raise ValueError("Independent seeds share input digests.")
            seen.update(digests)
        if previous_finish is not None and timestamp(record["started_utc"]) < previous_finish:
            raise ValueError("Independent device runs overlap or are out of order.")
        previous_finish = timestamp(record["finished_utc"])
    return records


def work_energy_bounds(window, power_w):
    """Return lower and upper joules per billion rows under a padding-power assumption."""
    gross = window["joules_per_billion_rows"]
    padding_j_per_billion = power_w * window["sample_bounds"]["padding_upper_ns"] / window["normalized_rows"]
    return [max(0, gross - padding_j_per_billion), gross]


def comparison(pairs):
    windows = [record["window"] for pair in pairs for record in pair]
    rates, _ = protocol.energy.base.pace(windows, BACKENDS)
    gross = protocol.energy.median_energy(windows)
    sensitivity = {}
    for power_w in POWER_ASSUMPTIONS_W:
        bounds = {}
        for backend in BACKENDS:
            periods = [work_energy_bounds(w, power_w) for w in windows if w["backend"] == backend]
            bounds[backend] = [statistics.median(endpoint) for endpoint in zip(*periods, strict=True)]
        baseline_low, baseline_high = bounds[BASELINE]
        candidate_low, candidate_high = bounds[CANDIDATE]
        minimum = 1 - candidate_high / baseline_low if baseline_low > 0 else None
        sensitivity[str(power_w)] = {
            "energy_joules_per_billion_rows": bounds,
            "saving_lower": minimum,
            "saving_upper": 1 - candidate_low / baseline_high,
        }
    strict_low = sensitivity[str(max(POWER_ASSUMPTIONS_W))]["saving_lower"]
    return {"gross_energy_joules_per_billion_rows": gross,
            "gross_energy_saving": 1 - gross[CANDIDATE] / gross[BASELINE],
            "median_rows_per_second": rates, "throughput_change": rates[CANDIDATE] / rates[BASELINE] - 1,
            "padding_power_sensitivity": sensitivity,
            "passes_energy": strict_low is not None and strict_low >= 1 - protocol.energy.ENERGY_RATIO,
            "passes_throughput": rates[CANDIDATE] >= protocol.energy.THROUGHPUT_RATIO * rates[BASELINE]}


def summarize(records):
    summary = {"power_assumptions_w": list(POWER_ASSUMPTIONS_W), "seeds": {}}
    qualification = records.get(f"s{protocol.NUMERIC_SEED}")
    summary["qualification_passed"] = qualification["gate_passed"] if qualification is not None else None
    for seed in protocol.TIMING_SEEDS:
        pairs = [[records[name] for index in indices if (name := f"s{seed}-p{index}") in records]
                 for indices in ((0, 1), (2, 3), (4, 5))]
        accepted = [i for i, pair in enumerate(pairs) if protocol.pair_qualifies(pair)]
        exact = all(protocol.period_is_exact(record) for pair in pairs for record in pair)
        effect = comparison([pairs[index] for index in accepted]) if exact and len(accepted) == 2 else None
        summary["seeds"][str(seed)] = {
            "saved_periods": sum(map(len, pairs)), "eligible_pair_indices": accepted,
            "numerically_exact": exact if any(pairs) else None,
            "comparison": effect,
        }
    effects = [record["comparison"] for record in summary["seeds"].values()]
    summary["adopted"] = all(effect is not None and effect["passes_energy"] and effect["passes_throughput"]
                             for effect in effects)
    return summary


def verify_attempts(directory, records):
    first_probe = json.loads((directory / "clock-probe-before.json").read_bytes())
    last_probe = json.loads((directory / "clock-probe.json").read_bytes())
    early_bounds, late_bounds = clock_bounds(first_probe), clock_bounds(last_probe)
    lower, upper = min(early_bounds[0], late_bounds[0]), max(early_bounds[1], late_bounds[1])
    first = timestamp(first_probe["collected_utc"])
    last = timestamp(last_probe["collected_utc"])
    completed = set()
    for path in (directory / "attempts").glob("*.json"):
        attempt = json.loads(path.read_bytes())
        name, token = attempt["unit"], attempt["attempt_id"]
        record = records[name]
        if (path.stem, attempt["rows"], attempt["seed"], attempt["period"], attempt["code_revision"]) != (
            f"{name}-{token}", record["rows"], record["seed"], record["period"], record["code_revision"]
        ) or attempt["mode"] not in ("cache_lookup", "compute"):
            raise ValueError("An attempt record has the wrong name or provenance.")
        started = timestamp(attempt["started_utc"])
        if not first <= started <= last:
            raise ValueError("An experiment attempt falls outside the SSH clock probes.")
        outcome = attempt["outcome"]
        if outcome == "started":
            continue
        finished, elapsed = timestamp(attempt["finished_utc"]), attempt["elapsed_s"]
        if not started <= finished <= last or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("The attempt duration or clock brackets are invalid.")
        if outcome == "failed":
            continue
        if (outcome not in ("checkpoint_saved", "checkpoint_recovered") or name in completed
                or (timestamp(record["finished_utc"]) - finished).total_seconds() > upper):
            raise ValueError("Unexpected or duplicate checkpoint receipt.")
        if outcome == "checkpoint_saved" and (
            attempt["mode"] != "compute" or record["wall_seconds"] > elapsed
            or (timestamp(record["started_utc"]) - started).total_seconds() < lower
        ):
            raise ValueError("The device work falls outside the recorded local attempt.")
        completed.add(name)
    if completed != records.keys():
        raise ValueError("Every archive checkpoint requires a recorded completion or recovery.")


def verify_recovery(directory, records):
    recovery = json.loads((directory / "checkpoint-recovery.json").read_bytes())
    record = records[recovery["unit"]]
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if ((any(saved["window"] for saved in records.values()) and record["window"] is None)
            or recovery["code_revision"] != record["code_revision"] or recovery["result_origin"] != "cached"
            or recovery["canonical_json_bytes"] != len(canonical)
            or recovery["canonical_json_sha256"] != hashlib.sha256(canonical).hexdigest()):
        raise ValueError("Cache-only recovery does not match the archived work.")
    timestamp(recovery["read_utc"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    cli.add_argument("--partial", action="store_true", help="Verify the saved prefix of the experiment.")
    args = cli.parse_args()
    records = read_verified(args.directory)
    if not args.partial:
        if protocol.next_work(records) is not None:
            cli.error("Registered work is unfinished.")
        verify_attempts(args.directory, records)
        verify_recovery(args.directory, records)
    print(json.dumps(summarize(records), indent=2, allow_nan=False))
