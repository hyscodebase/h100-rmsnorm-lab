"""Check the 128-row replay and recompute its CUDA-event medians."""
import argparse
import json
from pathlib import Path

import run
import small_rows_run as protocol
from transfer_report import CASES, CAPTURES, counts, timing_medians, verify_cases, verify_revision

BACKENDS = (protocol.shuffle.REFERENCE, protocol.CANDIDATE)


def candidate_cases(records):
    return [case.get("comparisons", {}).get(protocol.CANDIDATE, case) for case in records]


def read_results(folder):
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("s*.json")}
    data = {seed: files[f"s{seed}.json"] for seed in protocol.NUMERIC_SEEDS}
    first = data[protocol.REPLAY_SEEDS[0]]
    anchor = protocol.load_archived(protocol.transfer.QUALIFICATION)
    sources = first["source_sha256"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The frozen source manifest differs.")
    verify_revision(run.ROOT, first["code_revision"], tuple(sorted(sources.items())))
    qualified = all(data[seed]["gate_passed"] for seed in protocol.NUMERIC_SEEDS)
    expected = protocol.SEEDS if qualified else protocol.NUMERIC_SEEDS
    if files.keys() != {f"s{seed}.json" for seed in expected}:
        raise ValueError("The archive must contain each eligible checkpoint exactly once.")
    data = {seed: files[f"s{seed}.json"] for seed in expected}
    for seed, record in data.items():
        if (record["seed"], record["rows"], record["width"]) != (seed, protocol.ROWS, run.WIDTH):
            raise ValueError("Checkpoint identity differs.")
        if (record["code_revision"] != first["code_revision"] or record["source_sha256"] != sources
                or record["asset"] != anchor["asset"] or record["environment"] != anchor["environment"]):
            raise ValueError("Checkpoint provenance differs.")
        code = record["codegen"]
        if (code["ptx_instruction_sha256"] != first["codegen"]["ptx_instruction_sha256"]
                or (code["num_warps"], code["warp_shuffle_offsets"], code["emitted_shared_loads"])
                != (8, [1, 1, 2, 2, 4, 4, 8, 8, 16, 16], 4)):
            raise ValueError("The compiled reduction differs.")
        fixtures, captures = record["fixtures"], record["replay_checks"]
        elements = protocol.ROWS * run.WIDTH
        verify_cases(candidate_cases(fixtures), CASES, elements)
        verify_cases(candidate_cases(captures), CAPTURES if captures else (), elements)
        exact, reproduced = protocol.check_replay(seed, fixtures, captures)
        timed = seed in protocol.TIMING_SEEDS and exact and reproduced
        if (bool(record["timing"]) != timed or record["gate_passed"] != (exact and reproduced)
                or record["archived_comparison_reproduced"] != (reproduced if seed in protocol.REPLAY_SEEDS else None)):
            raise ValueError("The recorded passage or timing differs from the numerical evidence.")
        if timed:
            timing_medians(record, BACKENDS)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=run.HERE / "small-rows-results")
    args = parser.parse_args()
    data = read_results(args.directory)
    print("seed normalized_elements bit_differences rejected archived_comparison vllm_us triton_us reduction")
    for seed, r in sorted(data.items()):
        numerical = {"fixtures": candidate_cases(r["fixtures"]), "replay_checks": candidate_cases(r["replay_checks"])}
        summary = (seed, *counts((numerical,)), r["archived_comparison_reproduced"])
        if r["timing"]:
            vllm, candidate = timing_medians(r, BACKENDS)
            print(*summary, f"{vllm:.3f} {candidate:.3f} {100 * (1 - candidate / vllm):.3f}%")
        else:
            print(*summary, "preflight" if r["gate_passed"] else "excluded")
