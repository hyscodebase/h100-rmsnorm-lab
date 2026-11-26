"""Verify the three-way H100 comparison from its frozen checkpoints."""
import argparse
import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import json

import run
import shuffle_run as protocol

BACKENDS = protocol.REFERENCE, protocol.GROUPED, protocol.CANDIDATE


def timing_medians(record):
    timing = record["timing"]
    if not timing or len(record["replay_checks"]) != run.OPS or timing["ops_per_graph"] != run.OPS:
        raise ValueError("The event comparison requires all 32 replay checks.")
    if timing["round_order"] != run.original.schedule(BACKENDS, record["seed"]) or not timing["inputs_restored_bitwise"]:
        raise ValueError("The backend randomization or input restoration differs.")
    result = {}
    for name in BACKENDS:
        data = timing["statistics"][name]
        events = data["graph_elapsed_ms"]
        if len(events) != run.ROUNDS or not all(math.isfinite(ms) and ms > 0 for ms in events):
            raise ValueError("Incomplete or invalid CUDA-event intervals.")
        microseconds = [ms * 1000 / run.OPS for ms in events]
        quartiles = statistics.quantiles(microseconds, n=4, method="inclusive")
        if data["samples_us"] != microseconds or [data[k] for k in ("q1_us", "median_us", "q3_us")] != quartiles:
            raise ValueError("The reported quartiles differ from the CUDA events.")
        result[name] = quartiles[1]
    ratio = result[protocol.CANDIDATE] / result[protocol.REFERENCE]
    if timing["triton_over_vllm"] != ratio or timing["passes_5pct"] != (ratio <= .95):
        raise ValueError("The measured ratio or reported decision differs.")
    return result


def read_results(folder):
    files = {p.name for p in folder.glob("s*.json")}
    if files != {f"s{seed}.json" for seed in protocol.SEEDS}:
        raise ValueError("The complete five checkpoint files are required.")
    data = {seed: json.loads((folder / f"s{seed}.json").read_bytes()) for seed in protocol.SEEDS}
    first = data[protocol.REPLAY_SEED]
    revision = first["code_revision"]
    for path, checksum in first["source_sha256"].items():
        blob = subprocess.check_output(["git", "--no-replace-objects", "cat-file", "blob", f"{revision}:{path}"],
                                       cwd=run.ROOT)
        if hashlib.sha256(blob).hexdigest() != checksum:
            raise ValueError(f"Frozen source hash differs for {path}.")
    coords = first["previous_coordinates"]
    if len(coords) != 47 or any(p["vllm"] != p["triton_grouped"] or p["vllm"] != p["triton_shuffle"] for p in coords):
        raise ValueError("The 47 registered coordinates must match.")
    for seed, r in data.items():
        if (r["seed"], r["rows"], r["width"]) != (seed, run.ROWS, run.WIDTH):
            raise ValueError("Checkpoint identity differs.")
        for key in ("code_revision", "source_sha256", "environment", "asset"):
            if r[key] != first[key]:
                raise ValueError(f"Checkpoint provenance differs at {key}.")
        code = r["codegen"]
        if (code["num_warps"], code["warp_shuffle_offsets"], code["emitted_shared_loads"]) != (8, [1, 2, 4, 8, 16], 2):
            raise ValueError("Launch or reduction instructions differ.")
        if code["ptx_instruction_sha256"] != first["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("PTX instructions changed across checkpoints.")
        if not r["gate_passed"] or len(r["fixtures"]) != 8:
            raise ValueError("Numerical verification must pass each seed.")
        for case in r["fixtures"] + r["replay_checks"]:
            if not case["passed"] or set(case["comparisons"]) != {protocol.GROUPED, protocol.CANDIDATE}:
                raise ValueError("Fixture or replay comparison failed.")
            for check in case["comparisons"].values():
                if not check["passed"] or check["bounded_rejected"] or check["elements"] != run.ROWS * run.WIDTH:
                    raise ValueError("Incomplete or failed pairwise evidence.")
        if seed in protocol.TIMING_SEEDS:
            timing_medians(r)
        elif r["timing"] is not None or r["replay_checks"]:
            raise ValueError("Preflight cannot provide timing data.")
    return data


def summary(data):
    return [timing_medians(data[seed]) for seed in protocol.TIMING_SEEDS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=run.HERE / "shuffle-results")
    args = parser.parse_args()
    measurements = read_results(args.directory)
    match = sum(case["comparisons"][protocol.CANDIDATE]["pairs_differ_bitwise"]
                for r in measurements.values() for case in r["fixtures"] + r["replay_checks"]) == 0
    for seed, times in zip(protocol.TIMING_SEEDS, summary(measurements)):
        vllm, grouped, shuffle = (times[name] for name in BACKENDS)
        print(seed, f"{vllm:.3f}", f"{grouped:.3f}", f"{shuffle:.3f}",
              f"vLLM minus shuffle: {100 * (1-shuffle/vllm):.2f}%")
    print("All compared shuffle outputs agree bitwise:", match)
    print("All timing seeds meet the 5% improvement gate:",
          all(measurements[s]["timing"]["passes_5pct"] for s in protocol.TIMING_SEEDS))
