"""Verify and print the grouped-versus-vLLM checkpoint comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import grouped_run as protocol
import run


def read_results(folder):
    expected = set(protocol.SEEDS)
    paths = {int(p.stem[1:]): p for p in folder.glob("s*.json")}
    if set(paths) != expected:
        raise ValueError("The comparison requires the complete five-seed campaign.")
    records = {seed: json.loads(paths[seed].read_text()) for seed in protocol.SEEDS}
    first = records[protocol.REPLAY_SEED]
    root = run.ROOT
    revision, sources = first["code_revision"], first["source_sha256"]
    for path, checksum in sources.items():
        committed = subprocess.check_output(["git", "--no-replace-objects", "cat-file", "blob", f"{revision}:{path}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != checksum:
            raise ValueError(f"Frozen source hash differs at {path}.")
    for seed, result in records.items():
        if (result["seed"], result["rows"], result["width"]) != (seed, run.ROWS, run.WIDTH):
            raise ValueError("Checkpoint identity differs.")
        for field in ("code_revision", "source_sha256", "environment", "asset"):
            if result[field] != first[field]:
                raise ValueError(f"Checkpoint provenance differs at {field}.")
        if result["codegen"]["ptx_instruction_sha256"] != first["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The compiled instructions differ.")
        if len(result["fixtures"]) != 8 or not result["gate_passed"]:
            raise ValueError(f"Missing numerical proof for seed {seed}.")
        if not all(case["passed"] and case["bounded_rejected"] == 0
                   for case in result["fixtures"] + result["replay_checks"]):
            raise ValueError("A numerical gate failed.")
        if seed in protocol.TIMING_SEEDS:
            validate_timing(result)
        elif result["timing"] is not None or result["replay_checks"]:
            raise ValueError("The numerical preflight must not contain timing data.")
    old = first["old_coordinates"]
    if (sum(row["original_rejected"] for row in old), sum(row["still_different"] for row in old)) != (47, 0):
        raise ValueError("The 47 original rejection points must be reproduced and aligned.")
    coordinates = [point for row in old for point in row["coordinates"]]
    if len(coordinates) != 47 or any(p["vllm"] == p["triton_four_warp"] or p["vllm"] != p["triton_grouped"]
                                       for p in coordinates):
        raise ValueError("Original and grouped BF16 encodings do not prove the claimed agreement.")
    return records


def validate_timing(result):
    timing = result["timing"]
    if not timing or len(result["replay_checks"]) != run.OPS or timing["ops_per_graph"] != run.OPS:
        raise ValueError("The event comparison requires all replay checks.")
    if timing["round_order"] != run.original.schedule((protocol.REFERENCE, protocol.CANDIDATE), result["seed"]):
        raise ValueError("The shuffled backend order differs.")
    if not timing["inputs_restored_bitwise"]:
        raise ValueError("The graph inputs must be restored.")
    medians = []
    for name in (protocol.REFERENCE, protocol.CANDIDATE):
        data = timing["statistics"][name]
        intervals = data["graph_elapsed_ms"]
        if len(intervals) != run.ROUNDS or not all(math.isfinite(ms) and ms > 0 for ms in intervals):
            raise ValueError("Incomplete or invalid CUDA event intervals.")
        samples = [ms * 1000 / run.OPS for ms in intervals]
        q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
        if data["samples_us"] != samples or [data[k] for k in ("q1_us", "median_us", "q3_us")] != [q1, median, q3]:
            raise ValueError("Latency statistics differ from raw samples.")
        medians.append(median)
    reference, candidate = medians
    if timing["triton_over_vllm"] != candidate / reference or timing["passes_5pct"] != (candidate <= reference * .95):
        raise ValueError("The observed comparison differs from its decision rule.")
    return medians


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=run.HERE / "grouped-results")
    args = parser.parse_args()
    data = read_results(args.directory)
    trials = [data[seed] for seed in protocol.TIMING_SEEDS]
    for experiment in trials:
        vllm, triton = validate_timing(experiment)
        print(experiment["seed"], f"{vllm:.3f}", f"{triton:.3f}", f"{100 * (triton/vllm - 1):+.2f}%")
    print("All seeds met the 5% latency criterion:", all(r["timing"]["passes_5pct"] for r in trials))
