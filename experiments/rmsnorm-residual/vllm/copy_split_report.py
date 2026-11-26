"""Verify coordinated CUDA intervals, exact outputs and NVML coverage."""
import argparse
import gzip
import json
import math
from pathlib import Path

import copy_split_run as protocol
import run
from sixteen_warp_report import verify_attempts
from sustained_report import (admissible, verify_graph_records, verify_numerical, verify_recovery,
                              verify_telemetry)
from transfer_report import timestamp, verify_revision


def verify_numeric(record):
    digests = verify_numerical(record)
    verify_graph_records(record["restored_before_operator"])
    if set(record["split_checks"]) != set(protocol.BACKENDS):
        raise ValueError("A split-graph reference is missing.")
    for check in record["split_checks"].values():
        verify_graph_records(check["streams"])
    return digests


def verify_window(window, mode, minimum):
    start, end, samples = window["start_s"], window["end_s"], window["samples"]
    if (not samples or end != samples[-1][1] or not 0 <= start < end or window["wall_seconds"] != end - start
            or end - start < minimum or (len(samples) > 1 and samples[-2][1] - start >= minimum)
            or timestamp(window["started_utc"]) > timestamp(window["finished_utc"])):
        raise ValueError("A measurement did not stop at the first qualifying completion.")
    previous = start
    for before, after, total, copy_ms, kernel_ms in samples:
        if not (previous <= before < after <= end and math.isfinite(total) and total > 0):
            raise ValueError("A CUDA event interval is invalid.")
        previous = after
        if mode == "split":
            if (not all(v is not None and math.isfinite(v) and v > 0 for v in (copy_ms, kernel_ms))
                    or not math.isclose(copy_ms + kernel_ms, total, rel_tol=0, abs_tol=.002)):
                raise ValueError("The split intervals do not partition the coordinated span.")
        elif copy_ms is not None or kernel_ms is not None:
            raise ValueError("A combined graph reports unmeasured stage intervals.")
    launches = protocol.STREAMS * len(samples) * (2 if mode == "split" else 1)
    pairs = protocol.PAIRS * len(samples)
    if (window["attempted_graphs"] != launches or window["completed_graphs"] != launches
            or window["completed_pairs"] != pairs or window["normalized_rows"] != pairs * protocol.ROWS
            or window["cuda_ms"] != sum(sample[2] for sample in samples)):
        raise ValueError("Recorded counts or device duration differ from original observations.")
    for field, index in (("restore_ms", 3), ("operator_ms", 4)):
        expected = sum(sample[index] for sample in samples) if mode == "split" else None
        if window[field] != expected:
            raise ValueError("The recorded stage duration differs from the original observations.")


def verify_timing(seed, timing):
    wanted = protocol.balanced_order(seed)
    sequence = [(r, i, mode, backend) for r, group in enumerate(wanted) for i, (mode, backend) in enumerate(group)]
    blocks = timing["blocks"]
    if (timing["order"] != wanted or tuple(timing["sample_columns"]) != protocol.SAMPLE_COLUMNS
            or [(b["round"], b["position"], b["mode"], b["backend"]) for b in blocks] != sequence):
        raise ValueError("The balanced measurement schedule differs.")
    previous = 0
    timestamp(timing["observer_origin_utc"])
    for number, block in enumerate(blocks):
        mode = block["mode"]
        warmup = block["warmup"]
        verify_window(warmup, mode, protocol.WARMUP_S)
        verify_window(block, mode, protocol.PERIOD_S)
        if (block["id"] != number or block["condition"] != f"{mode}:{block['backend']}"
                or not previous <= warmup["start_s"] < warmup["end_s"] <= block["start_s"]):
            raise ValueError("Window identities or boundaries overlap.")
        previous = block["end_s"]
        verify_graph_records(block["post_graph_checks"])
    if timing["outputs_match_after_timing"] != all(protocol.base.outputs_match(b["post_graph_checks"]) for b in blocks):
        raise ValueError("The archived output decision differs from the block checks.")
    if protocol.summarize(blocks) != (timing["rates"], timing["comparisons"]):
        raise ValueError("The throughput selection differs from original intervals.")
    verify_telemetry(timing["telemetry"], blocks, protocol.CONDITIONS, group_by="condition")


def read_results(folder):
    files = {path.name: json.loads(gzip.decompress(path.read_bytes())) for path in folder.glob("s*.json.gz")}
    first = files[f"s{protocol.NUMERIC_SEEDS[0]}.json.gz"]
    expected = protocol.SEEDS if first["gate_passed"] else protocol.NUMERIC_SEEDS
    if files.keys() != {f"s{seed}.json.gz" for seed in expected}:
        raise ValueError("The archived seed set differs from the registered sequence.")
    anchor = json.loads(protocol.ANCHOR.read_bytes())
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The source manifest differs from the pinned source set.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))
    seen, data = set(), {}
    for seed in expected:
        result = files[f"s{seed}.json.gz"]
        if (result["seed"], result["rows"], result["width"]) != (seed, protocol.ROWS, protocol.WIDTH):
            raise ValueError("Identity differs.")
        if (result["code_revision"] != revision or result["source_sha256"] != sources
                or result["asset"] != anchor["asset"] or result["environment"] != anchor["environment"]):
            raise ValueError("The H100 stack or immutable source differs.")
        for backend in protocol.BACKENDS[1:]:
            for field in ("ptx_instruction_sha256", "binary_sha256"):
                if result["codegen"][backend][field] != anchor["codegen"][backend][field]:
                    raise ValueError("An implementation differs from the sustained comparison.")
        digests = verify_numeric(result["numerical"])
        if seen & digests:
            raise ValueError("An independent seed reused input digests.")
        seen.update(digests)
        eligible = protocol.numeric_gate(result["numerical"])
        if result["gate_passed"] != eligible or bool(result["timing"]) != (eligible and seed in protocol.TIMING_SEEDS):
            raise ValueError("The archive disagrees with the numerical gate.")
        if result["timing"]:
            verify_timing(seed, result["timing"])
        data[seed] = result
    verify_attempts(folder, data)
    if any(result["timing"] for result in data.values()):
        verify_recovery(data, json.loads((folder / "checkpoint-recovery.json").read_bytes()), revision)
    return data


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=run.HERE / "copy-split-results")
    args = cli.parse_args()
    for seed, result in sorted(read_results(args.directory).items()):
        timing = result["timing"]
        print(seed, "verified" if admissible(timing) else "qualified" if result["gate_passed"] and not timing else "excluded")
        if not timing:
            continue
        for name in protocol.BACKENDS:
            joined = timing["rates"][f"combined:{name}"]
            split = timing["rates"][f"split:{name}"]
            print(f"  {name} operator={split['median_operator_rows_per_second']/1e6:.3f} M rows/s"
                  f" restore={100*split['restoration_fraction']:.2f}%"
                  f" combined wall={joined['median_wall_rows_per_second']/1e6:.3f}"
                  f" split wall={split['median_wall_rows_per_second']/1e6:.3f}")
        for scope, outcomes in timing["comparisons"].items():
            print(" ", scope, {name: f"{100*d['throughput_gain']:+.3f}%" for name, d in outcomes.items()})
