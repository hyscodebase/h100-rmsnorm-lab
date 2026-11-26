"""Verify copy-free residual RMSNorm evidence and recompute timing decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import copy_free_run as protocol
import run
from sixteen_warp_report import verify_attempts
from sustained_report import admissible, verify_measurement, verify_numerical, verify_recovery
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "copy-free-results"
TRANSFER_RESULTS = run.HERE / "copy-free-255-results"
GUARDS = {"activation_output_aliases_input", "residual_output_aliases_input", "outputs_alias_each_other"}


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
    if protocol.binary_identity(codegen) != protocol.binary_identity(first):
        raise ValueError("A loaded kernel differs from qualification.")
    fused = codegen[protocol.CANDIDATE]
    protocol.verify_anchor_identity(codegen, anchor, workload)
    add = codegen[protocol.SPLIT]["residual_add"]
    if ((add["num_warps"], add["tile_elements"], add["fp_fusion"]) != (4, 2048, False)
            or not add["ptx_opcodes"].get("ld.global.v4.b32")
            or not add["ptx_opcodes"].get("st.global.v4.b32")):
        raise ValueError("The add baseline is missing the expected vectorized layout.")
    for kernel in (fused, codegen[protocol.SPLIT]["normalization"]):
        if (kernel["num_warps"], kernel["required_threads"], kernel["warp_shuffle_offsets"],
                kernel["emitted_shared_loads"], kernel["ptx_opcodes"].get("bar.sync"), kernel["fp_fusion"]
        ) != (16, 512, [1, 2, 4, 8, 16], 4, 1, False):
            raise ValueError("A normalization kernel differs from the matched collective.")


def read_results(folder=None, *, workload=protocol.DEFAULT):
    if folder is None:
        folder = RESULTS if workload == protocol.DEFAULT else TRANSFER_RESULTS
    rows = workload.rows
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
                or result["graph_contract"] != protocol.graph_contract(rows)):
            raise ValueError("Source, stack, or graph contract differs.")
        verify_codegen(result["codegen"], first["codegen"], anchor["codegen"], workload=workload)
        numerical = result["numerical"]
        current = verify_numerical(numerical, protocol.BACKENDS, rows=rows)
        if digests & current or any(set(guards) != GUARDS for guards in numerical["alias_guards"].values()):
            raise ValueError("An independent input digest or alias guard is invalid.")
        digests.update(current)
        passed = protocol.numeric_gate(numerical)
        timing = result["timing"]
        if passed != result["gate_passed"] or bool(timing) != (passed and seed in workload.timing_seeds):
            raise ValueError("The recorded numerical eligibility differs.")
        if timing:
            verify_measurement(seed, timing, backends=protocol.BACKENDS, period_s=protocol.PERIOD_S, rows=rows)
        data[seed] = result
    probe = json.loads((folder / "clock-probe.json").read_bytes())
    verify_attempts(folder, data, remote_clock_bounds_s=clock_bounds(probe), rows=rows)
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
        for backend in protocol.BACKENDS:
            load = timing["telemetry"]["by_backend"][backend]
            gpu = load["metrics"]["gpu_busy_pct"]
            print(f"  {backend}: {timing['median_rows_per_second'][backend] / 1e6:.3f} M rows/s"
                  f"  GPU median={gpu['median']}% samples={gpu['count']} gap={load['maximum_gap_s']:.3f} s")
        for name, comparison in timing["comparisons"].items():
            print(f"  vs {name}: {100 * comparison['throughput_gain']:+.3f}% pass={comparison['passes_5pct']}")
    print("Fused implementation adopted:", adopted(data, workload=workload))
