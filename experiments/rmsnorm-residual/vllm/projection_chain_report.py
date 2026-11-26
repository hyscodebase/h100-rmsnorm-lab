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


def verify_numerical(numerical, expected):
    base = protocol.base
    shape_bytes = 2 * protocol.ROWS * protocol.WIDTH
    layout = {"disjoint_aligned_buffers": True, "unique_streams": base.STREAMS,
              "disjoint_allocations": 6 * base.STREAMS + 3,
              "input_and_output_bytes": 6 * base.STREAMS * run.OPS * shape_bytes
              + 2 * protocol.WIDTH + 4 * protocol.WIDTH**2,
              "pair_tensor_stride_bytes": shape_bytes, "pairs_per_stream": run.OPS,
              "writable_arrays_per_stream": 4}
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
        verify_graph_records(checks[name]["streams"], rows=protocol.ROWS, fields=protocol.OUTPUT_FIELDS)
    fingerprints = numerical["source_fingerprints"]
    streams, weights = fingerprints["stream_input_sha256"], fingerprints["projection_weight_sha256"]
    if len(streams) != base.STREAMS or any(len(pair) != 2 for pair in streams) or len(weights) != 2:
        raise ValueError("A stream or projection-weight digest is missing.")
    digests = {fingerprints["weight_sha256"], *weights, *(digest for pair in streams for digest in pair)}
    if len(digests) != 2 * base.STREAMS + 3 or any(not re.fullmatch(r"[0-9a-f]{64}", d) for d in digests):
        raise ValueError("The independent input digests are invalid.")
    return digests


def verify_record(record, seed, anchor, sources, revision):
    if ((record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH)
            or record["source_sha256"] != sources or record["code_revision"] != revision
            or record["environment"] != anchor["environment"] or record["asset"] != anchor["asset"]
            or record["gemm"] != protocol.gemm_contract()):
        raise ValueError("The provenance or GEMM configuration differs.")
    numerical = record["numerical"]
    digests = verify_numerical(numerical, protocol.expected_kernels(anchor))
    passed = protocol.numeric_gate(numerical)
    timing = record["timing"]
    if record["gate_passed"] != passed or (timing is not None) != (passed and seed in protocol.TIMING_SEEDS):
        raise ValueError("The recorded gate or timing eligibility differs.")
    if timing:
        verify_measurement(seed, timing, backends=protocol.BACKENDS, rows=protocol.ROWS, fields=protocol.OUTPUT_FIELDS)
    if record["effect"] != protocol.chain_effect(timing):
        raise ValueError("The recorded chain effect differs from the measured rates.")
    return digests


def read_results(folder=RESULTS):
    results = {int(path.stem[1:]): json.loads(path.read_bytes()) for path in folder.glob("s[0-9]*.json")}
    preflight = results[protocol.NUMERIC_SEED]
    expected = protocol.SEEDS if preflight["gate_passed"] else (protocol.NUMERIC_SEED,)
    if results.keys() != set(expected):
        raise ValueError("The archive differs from the registered seeds.")
    sources, revision = preflight["source_sha256"], preflight["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    anchor = protocol.ANCHOR.read_bytes()
    if hashlib.sha256(anchor).hexdigest() != sources[str(protocol.ANCHOR.relative_to(run.ROOT))]:
        raise ValueError("The tuning anchor changed.")
    anchor = json.loads(anchor)
    seen = set()
    for seed, result in sorted(results.items()):
        digests = verify_record(result, seed, anchor, sources, revision)
        if seen & digests:
            raise ValueError("Independent seeds share source digests.")
        seen.update(digests)
    verify_attempt_clocks(folder, results, earlier_clock=folder / "clock-probe-before.json")
    verify_supplemental_samples(folder)
    if any(result["timing"] for result in results.values()):
        verify_recovery(results, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return results


def reproduced(results):
    effects = [results.get(seed, {}).get("effect") for seed in protocol.TIMING_SEEDS]
    return all(effect and effect["passes_1pct"] for effect in effects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    results = read_results(parser.parse_args().directory)
    for seed, result in sorted(results.items()):
        print(f"{seed}: qualified={result['gate_passed']}")
        if result["timing"]:
            for name, rate in result["timing"]["median_rows_per_second"].items():
                print(f"  {name}: {rate / 1e6:.6f} M rows/s")
        if result["effect"]:
            print(f"  fused gain: {100 * result['effect']['fused_gain']:+.4f}%")
    print("Registered chain effect reproduced:", reproduced(results))
