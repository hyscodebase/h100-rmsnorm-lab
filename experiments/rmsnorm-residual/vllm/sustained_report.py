"""Verify the sustained-load archive and recompute throughput and NVML coverage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import run
from sixteen_warp_report import verify_attempts, verify_pairwise
import sustained_run as protocol
from sustained_sensors import FIELDS, PERIOD_S as SENSOR_PERIOD, associate, summarize_load
from transfer_report import CASES, timestamp, verify_revision

PAIR_FIELDS = ("activation_bit_differences_by_pair", "residual_bit_differences_by_pair")


def verify_graph_records(records):
    if [item["stream"] for item in records] != list(range(protocol.STREAMS)):
        raise ValueError("Missing independent stream records.")
    for item in records:
        for field in PAIR_FIELDS:
            counts = item[field]
            if len(counts) != run.OPS or not all(type(n) is int and 0 <= n <= protocol.ROWS * protocol.WIDTH for n in counts):
                raise ValueError("Invalid per-pair bit difference counts.")


def verify_numerical(numerical, backends=protocol.BACKENDS):
    verify_pairwise(numerical["fixtures"], CASES, backends)
    if set(numerical["graph_checks"]) != set(backends):
        raise ValueError("A reference or candidate output check is missing.")
    for record in numerical["graph_checks"].values():
        verify_graph_records(record["streams"])
    shape_bytes = 2 * protocol.ROWS * protocol.WIDTH
    expected = {"disjoint_aligned_buffers": True, "unique_streams": protocol.STREAMS,
                "disjoint_allocations": protocol.STREAMS * 4 + 1,
                "input_and_output_bytes": protocol.STREAMS * 4 * run.OPS * shape_bytes + 2 * protocol.WIDTH,
                "pair_tensor_stride_bytes": shape_bytes, "pairs_per_stream": run.OPS}
    if numerical["buffers"] != expected:
        raise ValueError("The disjoint buffer layout differs from the plan.")
    fingerprints = numerical["source_fingerprints"]
    per_stream = fingerprints["stream_input_sha256"]
    if len(per_stream) != protocol.STREAMS or not all(len(pair) == 2 for pair in per_stream):
        raise ValueError("A stream is missing its source digests.")
    digests = {fingerprints["weight_sha256"], *(digest for pair in per_stream for digest in pair)}
    if len(digests) != 2 * protocol.STREAMS + 1 or not all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests):
        raise ValueError("The independent input digests are invalid.")
    return digests


def verify_window(window, minimum):
    start, end, bursts = window["start_s"], window["end_s"], window["bursts"]
    if (not bursts or end != bursts[-1]["end_s"] or not 0 <= start < end
            or window["wall_seconds"] != end - start or end - start < minimum
            or (len(bursts) > 1 and bursts[-2]["end_s"] - start >= minimum)
            or timestamp(window["started_utc"]) > timestamp(window["finished_utc"])):
        raise ValueError("The window differs from the first completed burst at the target duration.")
    previous = start
    for burst in bursts:
        if not (previous <= burst["start_s"] < burst["end_s"] <= end
                and math.isfinite(burst["cuda_ms"]) and burst["cuda_ms"] > 0):
            raise ValueError("The coordinated CUDA-event evidence is invalid.")
        previous = burst["end_s"]
    graphs = len(bursts) * protocol.STREAMS * protocol.BURST_REPLAYS
    if (window["attempted_graphs"] != graphs or window["completed_graphs"] != graphs
            or window["normalized_rows"] != graphs * run.OPS * protocol.ROWS
            or window["cuda_ms"] != sum(burst["cuda_ms"] for burst in bursts)):
        raise ValueError("A graph or normalized row count differs from the original bursts.")


def verify_telemetry(data, blocks, groups=protocol.BACKENDS, *, group_by="backend"):
    slots = data["slots"]
    if data["period_s"] != SENSOR_PERIOD or [s["slot"] for s in slots] != list(range(len(slots))):
        raise ValueError("The NVML observation schedule is missing slots.")
    for slot in slots:
        if slot["scheduled_s"] != slot["slot"] * SENSOR_PERIOD or slot["status"] not in ("sampled", "missed"):
            raise ValueError("Malformed NVML sample slot.")
        if slot["status"] == "sampled":
            if not slot["scheduled_s"] <= slot["query_start_s"] <= slot["query_end_s"]:
                raise ValueError("Malformed NVML sample time.")
            timestamp(slot["utc"])
            values = slot["values"]
            if values.keys() != set(FIELDS) or any(
                value is not None and (not math.isfinite(value) or value < 0 or field.endswith("_pct") and value > 100)
                for field, value in values.items()
            ):
                raise ValueError("Invalid NVML device metrics.")
    assigned = [dict(slot) for slot in slots]
    associate(assigned, blocks)
    if assigned != slots or summarize_load(blocks, assigned, groups, group_by=group_by) != data[f"by_{group_by}"]:
        raise ValueError("NVML sample assignment or qualification differs.")


def verify_measurement(seed, timing, *, backends=protocol.BACKENDS, period_s=protocol.PERIOD_S):
    order = protocol.balanced_order(seed, backends)
    blocks = timing["blocks"]
    sequence = [(round_no, i, backend) for round_no, group in enumerate(order) for i, backend in enumerate(group)]
    if timing["order"] != order or [(b["round"], b["position"], b["backend"]) for b in blocks] != sequence:
        raise ValueError("The recorded backend schedule differs from the balanced schedule.")
    timestamp(timing["observer_origin_utc"])
    previous = 0
    for number, block in enumerate(blocks):
        warmup = block["warmup"]
        verify_window(warmup, protocol.WARMUP_S)
        verify_window(block, period_s)
        if block["id"] != number or not previous <= warmup["start_s"] < warmup["end_s"] <= block["start_s"]:
            raise ValueError("Warm-up and measurement windows overlap.")
        verify_graph_records(block["post_graph_checks"])
        previous = block["end_s"]
    if timing["outputs_match_after_timing"] != all(protocol.outputs_match(b["post_graph_checks"]) for b in blocks):
        raise ValueError("Post-window output status differs.")
    verify_telemetry(timing["telemetry"], blocks, backends)
    if protocol.pace(blocks, backends) != (timing["median_rows_per_second"], timing["comparisons"]):
        raise ValueError("The recorded throughput decisions differ from the original bursts.")


def read_results(folder):
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("s*.json")}
    first = files[f"s{protocol.NUMERIC_SEEDS[0]}.json"]
    expected = protocol.SEEDS if first["gate_passed"] else protocol.NUMERIC_SEEDS
    if files.keys() != {f"s{seed}.json" for seed in expected}:
        raise ValueError("Checkpoint set differs from the registered sequence.")
    anchor = json.loads(protocol.ANCHOR.read_bytes())
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("Frozen source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    seen = set()
    data = {}
    for seed in expected:
        record = files[f"s{seed}.json"]
        if (record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, protocol.WIDTH):
            raise ValueError("Unexpected seed or shape.")
        if (record["source_sha256"] != sources or record["code_revision"] != revision
                or record["asset"] != anchor["asset"] or record["environment"] != anchor["environment"]):
            raise ValueError("Checkpoint provenance differs.")
        for backend in protocol.BACKENDS[1:]:
            for key in ("ptx_instruction_sha256", "binary_sha256"):
                if record["codegen"][backend][key] != anchor["codegen"][backend][key]:
                    raise ValueError("Kernel code generation differs from the latency experiment.")
        digests = verify_numerical(record["numerical"])
        if seen & digests:
            raise ValueError("The independent seeds reused input hashes.")
        seen.update(digests)
        eligible = protocol.numeric_gate(record["numerical"])
        if record["gate_passed"] != eligible or bool(record["timing"]) != (eligible and seed in protocol.TIMING_SEEDS):
            raise ValueError("The archived numerical or timing decision differs.")
        if record["timing"]:
            verify_measurement(seed, record["timing"])
        data[seed] = record
    verify_attempts(folder, data)
    verify_recovery(data, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return data


def verify_recovery(data, recovered, revision):
    cached = data[recovered["seed"]]
    canonical = json.dumps(cached, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (not cached["timing"] or recovered["rows"] != cached["rows"] or recovered["code_revision"] != revision
            or recovered["result_origin"] != "cached" or not recovered["matches_local_result"]
            or recovered["canonical_json_bytes"] != len(canonical)
            or recovered["canonical_json_sha256"] != hashlib.sha256(canonical).hexdigest()):
        raise ValueError("The recorded recovery differs from the archived timing checkpoint.")


def admissible(timing):
    return timing is not None and timing["inputs_unchanged_after_timing"] and timing["outputs_match_after_timing"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=run.HERE / "sustained-results")
    args = cli.parse_args()
    data = read_results(args.directory)
    for seed, record in sorted(data.items()):
        timing = record["timing"]
        if not timing:
            print(seed, "qualified" if record["gate_passed"] else "excluded", "accuracy")
            continue
        print(seed, "verified" if admissible(timing) else "invalidated", "timing")
        for backend in protocol.BACKENDS:
            rate = timing["median_rows_per_second"][backend]
            load = timing["telemetry"]["by_backend"][backend]
            activity = load["metrics"]["gpu_busy_pct"]
            print(f"  {backend} {rate / 1e6:.3f} M rows/s GPU median {activity['median']}%"
                  f" samples {activity['count']} high load {load['high_load']}")
        for name, stats in timing["comparisons"].items():
            print(f"  throughput vs {name}: {100 * stats['throughput_gain']:+.3f}% pass {stats['passes_5pct']}")
