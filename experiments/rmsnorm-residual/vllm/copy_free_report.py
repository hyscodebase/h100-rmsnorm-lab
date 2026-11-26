"""Verify copy-free residual RMSNorm evidence and recompute timing decisions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import copy_free_run as protocol
import run
from sixteen_warp_report import verify_attempts
from sustained_report import PAIR_FIELDS, admissible, verify_graph_records, verify_measurement, verify_numerical, verify_recovery
from transfer_report import CASES, timestamp, verify_cases, verify_revision

RESULTS = run.HERE / "copy-free-results"
GUARDS = {"activation_output_aliases_input", "residual_output_aliases_input", "outputs_alias_each_other"}
DEVICE_METRICS = {"util_pct", "used_mib", "total_mib", "power_w", "temperature_c", "sm_clock_mhz",
                  "memory_clock_mhz", "power_limit_w", "memory_busy_pct"}


def clock_bounds(record):
    """Bound remote minus local wall time using the recorded SSH exchanges."""
    collected_at_ns = round(timestamp(record["collected_utc"]).timestamp() * 1e9)
    intervals = []
    previous = 0
    for probe in record["probes"]:
        sent, remote, received, elapsed = (probe[key] for key in (
            "local_sent_unix_ns", "remote_unix_ns", "local_received_unix_ns", "local_elapsed_ns",
        ))
        if (any(type(value) is not int for value in (sent, remote, received, elapsed))
                or not previous < sent < received <= collected_at_ns or remote <= 0 or elapsed <= 0):
            raise ValueError("A timestamp exchange is malformed.")
        intervals.append((remote - received, remote - sent))
        previous = received
    if len(intervals) < 2:
        raise ValueError("The clock check needs repeated exchanges.")
    lower = max(value[0] for value in intervals)
    upper = min(value[1] for value in intervals)
    if lower > upper:
        raise ValueError("The clock exchanges admit no common offset.")
    return lower / 1e9, upper / 1e9


def verify_codegen(codegen, first, anchor, *, workload=protocol.DEFAULT):
    if protocol.binary_identity(codegen, candidate=workload.candidate) != protocol.binary_identity(
        first, candidate=workload.candidate,
    ):
        raise ValueError("A loaded kernel differs from qualification.")
    fused = codegen[workload.candidate]
    protocol.verify_anchor_identity(codegen, anchor, workload)
    add = codegen[protocol.SPLIT]["residual_add"]
    if ((add["num_warps"], add["tile_elements"], add["fp_fusion"]) != (4, 2048, False)
            or not add["ptx_opcodes"].get("ld.global.v4.b32")
            or not add["ptx_opcodes"].get("st.global.v4.b32")):
        raise ValueError("The add baseline is missing the expected vectorized layout.")
    warps = workload.warps
    layout = (warps, warps * 32, [1, 2, 4, 8, 16], warps // 4, 1, False)
    for kernel in (fused, codegen[protocol.SPLIT]["normalization"]):
        if (kernel["num_warps"], kernel["required_threads"], kernel["warp_shuffle_offsets"],
                kernel["emitted_shared_loads"], kernel["ptx_opcodes"].get("bar.sync"), kernel["fp_fusion"]
        ) != layout:
            raise ValueError("A normalization kernel differs from the matched collective.")
        if warps == 8 and not all(kernel["ptx_opcodes"].get(op) for op in ("ld.global.v4.b32", "st.global.v4.b32")):
            raise ValueError("An 8-warp normalization kernel lost its vectorized layout.")


def verify_boundary_diagnostic(numerical, anchor):
    diagnostic = numerical["prior_16warps"]
    if (diagnostic["backend"] != protocol.CANDIDATE
            or diagnostic["source_fingerprints"] != numerical["source_fingerprints"]
            or any(diagnostic["codegen"][key] != anchor[protocol.CANDIDATE][key] for key in protocol.IDENTITY_FIELDS)):
        raise ValueError("The boundary diagnostic differs from the pinned input or 16-warp binary.")
    verify_cases(diagnostic["fixtures"], CASES, protocol.BOUNDARY.rows * protocol.WIDTH)
    verify_graph_records(diagnostic["streams"], rows=protocol.BOUNDARY.rows)
    return {field: sum(sum(stream[field]) for stream in diagnostic["streams"]) for field in PAIR_FIELDS}


def verify_supplemental_samples(folder):
    archive = json.loads((folder / "device-samples.json").read_bytes())
    attempts = [json.loads(path.read_bytes()) for path in (folder / "attempts").glob("*.json")]
    finished = {attempt["attempt_id"]: attempt for attempt in attempts if "finished_utc" in attempt}
    for window in archive["observations"]:
        attempt = finished.pop(window["attempt_id"], None)
        if attempt is None or (window["rows"], window["seed"]) != (attempt["rows"], attempt["seed"]):
            raise ValueError("The supplemental device window differs from its attempt.")
        started, ended = (timestamp(attempt[key]) for key in ("started_utc", "finished_utc"))
        previous = None
        for sample in window["samples"]:
            if sample.keys() != DEVICE_METRICS | {"timestamp_utc"}:
                raise ValueError("The supplemental device fields differ.")
            observed = timestamp(sample["timestamp_utc"])
            if not started <= observed <= ended or (previous is not None and observed <= previous):
                raise ValueError("A supplemental device sample is outside the ordered attempt window.")
            for metric in DEVICE_METRICS:
                value = sample[metric]
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    or (metric.endswith("_pct") and value > 100)
                ):
                    raise ValueError("A supplemental device measurement is invalid.")
            previous = observed
    if finished:
        raise ValueError("A finished attempt is missing a supplemental device window.")


def read_results(folder=None, *, workload=protocol.DEFAULT):
    rows = workload.rows
    backends = workload.backends
    if folder is None:
        folder = RESULTS if workload == protocol.DEFAULT else run.HERE / f"copy-free-{rows}-results"
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("s*.json")}
    first = files[f"s{workload.numeric_seed}.json"]
    expected = workload.seeds if first["gate_passed"] else (workload.numeric_seed,)
    if files.keys() != {f"s{seed}.json" for seed in expected}:
        raise ValueError("The checkpoints differ from the registered seeds.")
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(workload.sources):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor_bytes = workload.anchor.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != sources[str(workload.anchor.relative_to(run.ROOT))]:
        raise ValueError("The prior checkpoint differs from the registered anchor.")
    anchor = json.loads(anchor_bytes)
    data, digests = {}, set()
    for seed in expected:
        result = files[f"s{seed}.json"]
        if (result["seed"], result["rows"], result["width"]) != (seed, rows, protocol.WIDTH):
            raise ValueError("The checkpoint seed or shape differs.")
        if (result["code_revision"] != revision or result["source_sha256"] != sources
                or result["asset"] != anchor["asset"] or result["environment"] != anchor["environment"]
                or result["graph_contract"] != protocol.graph_contract(rows, backends=backends)):
            raise ValueError("Source, stack, or graph contract differs.")
        verify_codegen(result["codegen"], first["codegen"], anchor["codegen"], workload=workload)
        numerical = result["numerical"]
        current = verify_numerical(numerical, backends, rows=rows)
        if digests & current or any(set(guards) != GUARDS for guards in numerical["alias_guards"].values()):
            raise ValueError("An independent input digest or alias guard is invalid.")
        digests.update(current)
        if workload == protocol.BOUNDARY and seed == workload.numeric_seed:
            verify_boundary_diagnostic(numerical, anchor["codegen"])
        passed = protocol.numeric_gate(numerical, backends)
        timing = result["timing"]
        if passed != result["gate_passed"] or bool(timing) != (passed and seed in workload.timing_seeds):
            raise ValueError("The recorded numerical eligibility differs.")
        if timing:
            verify_measurement(seed, timing, backends=backends, period_s=protocol.PERIOD_S, rows=rows)
        data[seed] = result
    probe = json.loads((folder / "clock-probe.json").read_bytes())
    verify_attempts(folder, data, remote_clock_bounds_s=clock_bounds(probe), rows=rows)
    verify_supplemental_samples(folder)
    if any(result["timing"] for result in data.values()):
        verify_recovery(data, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return data


def adopted(data, *, workload=protocol.DEFAULT):
    timings = [data.get(seed, {}).get("timing") for seed in workload.timing_seeds]
    return all(admissible(timing) and timing["comparisons"][protocol.SPLIT]["passes_5pct"] for timing in timings)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?")
    cli.add_argument("--rows", type=int, choices=protocol.WORKLOADS, default=protocol.ROWS)
    args = cli.parse_args()
    workload = protocol.WORKLOADS[args.rows]
    data = read_results(args.directory, workload=workload)
    for seed, result in sorted(data.items()):
        timing = result["timing"]
        if not timing:
            print(seed, "qualified" if result["gate_passed"] else "excluded")
            continue
        print(seed, "verified" if admissible(timing) else "invalidated")
        for backend in workload.backends:
            load = timing["telemetry"]["by_backend"][backend]
            gpu = load["metrics"]["gpu_busy_pct"]
            print(f"  {backend}: {timing['median_rows_per_second'][backend] / 1e6:.3f} M rows/s"
                  f"  GPU median={gpu['median']}% samples={gpu['count']} gap={load['maximum_gap_s']:.3f} s")
        for name, comparison in timing["comparisons"].items():
            print(f"  vs {name}: {100 * comparison['throughput_gain']:+.3f}% pass={comparison['passes_5pct']}")
    print("Fused implementation adopted:", adopted(data, workload=workload))
    if workload == protocol.BOUNDARY:
        anchor = json.loads(workload.anchor.read_bytes())
        difference = verify_boundary_diagnostic(data[workload.numeric_seed]["numerical"], anchor["codegen"])
        for field, count in difference.items():
            print(f"Prior 16 warps, {field}: {count}")
