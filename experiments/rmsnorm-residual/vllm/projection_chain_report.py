"""Verify the archived 512-row BF16 projection chain and its registered effect."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from copy_free_report import GUARDS, verify_supplemental_samples
import projection_chain_run as protocol
import run
from split_schedule_report import verify_attempt_clocks
from sustained_report import verify_graph_records, verify_measurement, verify_recovery
from transfer_report import verify_revision

RESULTS = run.HERE / "projection-chain-512-results"
WIDE_RESULTS = run.HERE / "projection-tail-12288-results"


def verify_numerical(numerical, expected, *, workload=protocol.SQUARE):
    base = protocol.base
    rows, width, downstream = protocol.ROWS, protocol.WIDTH, workload.downstream_width
    shape_bytes, downstream_bytes = 2 * rows * width, 2 * rows * downstream
    weight_bytes = 2 * width * (1 + width + downstream)
    layout = {"disjoint_aligned_buffers": True, "unique_streams": base.STREAMS,
              "disjoint_allocations": 6 * base.STREAMS + 3,
              "input_and_output_bytes": base.STREAMS * run.OPS * (5 * shape_bytes + downstream_bytes) + weight_bytes,
              "pair_tensor_stride_bytes": shape_bytes, "pairs_per_stream": run.OPS,
              "writable_arrays_per_stream": 4}
    if downstream != width:
        layout["downstream_pair_tensor_stride_bytes"] = downstream_bytes
    if numerical["buffers"] != layout:
        raise ValueError("Projection buffer layout differs.")
    guards = numerical["alias_guards"]
    if set(guards) != set(protocol.BACKENDS) or any(set(item) != GUARDS for item in guards.values()):
        raise ValueError("An output alias guard is missing.")
    backends = (protocol.REFERENCE, *protocol.BACKENDS)
    checks = numerical["graph_checks"]
    audits = numerical["capture_audits"]
    if set(checks) != set(backends) or set(audits) != set(backends):
        raise ValueError("An oracle or candidate captured graph is missing.")
    for name in backends:
        captured = {"pairs": run.OPS, "residual_kernel_binary_sha256": expected.get(name)}
        if audits[name] != [captured] * base.STREAMS:
            raise ValueError("Captured residual kernel identities or call counts differ.")
        verify_graph_records(checks[name]["streams"], rows=rows, fields=protocol.OUTPUT_FIELDS,
                             field_widths={protocol.DOWNSTREAM_FIELD: downstream})
    fingerprints = numerical["source_fingerprints"]
    streams, weights = fingerprints["stream_input_sha256"], fingerprints["projection_weight_sha256"]
    if len(streams) != base.STREAMS or any(len(pair) != 2 for pair in streams) or len(weights) != 2:
        raise ValueError("A stream or projection-weight digest is missing.")
    digests = {fingerprints["weight_sha256"], *weights, *(digest for pair in streams for digest in pair)}
    if len(digests) != 2 * base.STREAMS + 3 or any(not re.fullmatch(r"[0-9a-f]{64}", d) for d in digests):
        raise ValueError("The independent input digests are invalid.")
    return digests


def verify_record(record, seed, anchor, sources, revision, *, workload=protocol.SQUARE):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["source_sha256"] != sources or record["code_revision"] != revision
            or record["environment"] != anchor["environment"] or record["asset"] != anchor["asset"]
            or record["gemm"] != protocol.gemm_contract(workload)):
        raise ValueError("The provenance or GEMM configuration differs.")
    numerical = record["numerical"]
    digests = verify_numerical(numerical, protocol.expected_kernels(anchor), workload=workload)
    passed = protocol.numeric_gate(numerical)
    timing = record["timing"]
    if record["gate_passed"] != passed or (timing is not None) != (passed and seed in workload.timing_seeds):
        raise ValueError("The recorded gate or timing eligibility differs.")
    if timing:
        verify_measurement(seed, timing, backends=protocol.BACKENDS, rows=protocol.ROWS, fields=protocol.OUTPUT_FIELDS,
                           field_widths={protocol.DOWNSTREAM_FIELD: workload.downstream_width})
    if record["effect"] != protocol.chain_effect(timing):
        raise ValueError("The recorded chain effect differs from the measured rates.")
    return digests


def read_results(folder=None, *, workload=protocol.SQUARE):
    if folder is None:
        folder = RESULTS if workload == protocol.SQUARE else WIDE_RESULTS
    results = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    preflight = results[workload.numeric_seed]
    expected = workload.seeds if preflight["gate_passed"] else (workload.numeric_seed,)
    if results.keys() != set(expected):
        raise ValueError("The archive differs from the registered seeds.")
    sources, revision = preflight["source_sha256"], preflight["code_revision"]
    if sources.keys() != set(workload.sources):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The tuning anchor changed.")
    anchor = json.loads(anchor)
    seen = set()
    for seed, result in sorted(results.items()):
        digests = verify_record(result, seed, anchor, sources, revision, workload=workload)
        if seen & digests:
            raise ValueError("Independent seeds share source digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, results, earlier_clock=folder / "clock-probe-before.json")
    verify_supplemental_samples(folder)
    if any(result["timing"] for result in results.values()):
        verify_recovery(results, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return results


def threshold_result(results, *, workload=protocol.SQUARE):
    effects = [results.get(seed, {}).get("effect") for seed in workload.timing_seeds]
    if any(effect is None for effect in effects):
        return "Incomplete exact timing"
    passed = sum(effect["passes_1pct"] for effect in effects)
    if passed == len(effects):
        return "At least 1% in every seed"
    if not passed:
        return "Below 1% in every seed"
    return "Varies across the 1% threshold"


def reproduced(results):
    return threshold_result(results) == "At least 1% in every seed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--downstream-width", type=int, choices=protocol.WORKLOADS, default=protocol.WIDTH)
    args = parser.parse_args()
    workload = protocol.WORKLOADS[args.downstream_width]
    results = read_results(args.directory, workload=workload)
    for seed, result in sorted(results.items()):
        print(f"{seed}: qualified={result['gate_passed']}")
        if result["timing"]:
            for name, rate in result["timing"]["median_rows_per_second"].items():
                print(f"  {name}: {rate / 1e6:.6f} M rows/s")
        if result["effect"]:
            print(f"  fused gain: {100 * result['effect']['fused_gain']:+.4f}%")
    if workload == protocol.SQUARE:
        print("Registered chain effect reproduced:", reproduced(results))
    else:
        print("Registered threshold result:", threshold_result(results, workload=workload))
