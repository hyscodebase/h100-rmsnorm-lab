"""Verify row-count checkpoints and print the latency and accuracy evidence."""
import argparse
import json
from pathlib import Path
import sys

import row_transfer as protocol
import run

sys.path.insert(0, str(run.ROOT))
from residual_plan import FIXTURES, timestamp, verify_revision

BACKENDS = protocol.shuffle.REFERENCE, protocol.shuffle.CANDIDATE
CASES = (*FIXTURES, "weight_rounding", "signed_zero")
CAPTURES = tuple(f"captured_{index}" for index in range(run.OPS))


def verify_cases(records, names, elements):
    if tuple(r["fixture"] for r in records) != names:
        raise ValueError("The numerical inputs differ from the registered protocol.")
    for record in records:
        if "reason" in record and not record["passed"]:
            continue
        rejected = record["bounded_rejected"]
        if (record["elements"] != elements or not 0 <= rejected <= record["strict_outside_tolerance"] <= elements
                or not 0 <= record["pairs_differ_bitwise"] <= elements
                or len(record["rejected_coords"]) != min(rejected, 16)
                or len(record["accepted_coords"]) != record["strict_outside_tolerance"] - rejected):
            raise ValueError("Incomplete numerical evidence.")
        passed = rejected == 0
        if record["fixture"] == "weight_rounding":
            passed = passed and record["weight_probe"] == [1.46875, 1.46875]
        if record["fixture"] == "signed_zero":
            passed = passed and record["normalized_zero_bits"] == [[32768, 0], [32768, 0]]
        if record["passed"] != passed:
            raise ValueError("The numerical decision differs from the recorded evidence.")
    return all(record["passed"] for record in records)


def timing_medians(record, backends=BACKENDS):
    timing = record["timing"]
    if (timing["ops_per_graph"] != run.OPS or not timing["inputs_restored_bitwise"]
            or timing["round_order"] != run.original.schedule(backends, record["seed"])
            or set(timing["statistics"]) != set(backends)):
        raise ValueError("The capture, restoration or randomization contract differs.")
    medians = []
    for backend in backends:
        stats = timing["statistics"][backend]
        if stats != run.original.latency(stats["graph_elapsed_ms"]):
            raise ValueError("The reported quartiles differ from the CUDA events.")
        medians.append(stats["median_us"])
    reference, candidate = medians
    if (timing["triton_over_vllm"] != candidate / reference
            or timing["passes_5pct"] != (candidate <= reference * .95)):
        raise ValueError("The reported comparison decision differs.")
    return reference, candidate


def verify_attempts(folder, records):
    completed = set()
    for path in sorted((folder / "attempts").glob("*.json")):
        attempt = json.loads(path.read_bytes())
        key = attempt["rows"], attempt["seed"]
        if (key not in records or attempt["code_revision"] != records[key]["code_revision"]
                or path.stem != f"r{key[0]}-s{key[1]}-{attempt['attempt_id']}"):
            raise ValueError("Attempt provenance differs.")
        if attempt["outcome"] != "checkpoint_saved":
            continue
        if key in completed or not (
            timestamp(attempt["started_utc"]) <= timestamp(records[key]["started_utc"])
            <= timestamp(records[key]["finished_utc"]) <= timestamp(attempt["finished_utc"])
        ):
            raise ValueError("Invalid successful attempt or duplicate completion.")
        completed.add(key)
    if completed != records.keys():
        raise ValueError("Every result needs one completed attempt.")


def read_results(folder):
    baseline = json.loads((run.HERE / "shuffle-results/s114.json").read_bytes())
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("r*-s*.json")}
    preflight = {rows: files[f"r{rows}-s{protocol.QUALIFICATION}.json"] for rows in protocol.ROWS}
    expected = [(rows, protocol.QUALIFICATION) for rows in protocol.ROWS]
    expected.extend((rows, seed) for rows, first in preflight.items()
                    if first["gate_passed"] for seed in protocol.TIMING_SEEDS)
    if files.keys() != {f"r{rows}-s{seed}.json" for rows, seed in expected}:
        raise ValueError("The complete eligible checkpoint set is required.")
    first = preflight[protocol.ROWS[0]]
    if first["source_sha256"].keys() != set(protocol.SOURCES):
        raise ValueError("The frozen source manifest differs.")
    verify_revision(run.ROOT, first["code_revision"], tuple(sorted(first["source_sha256"].items())))
    records = {}
    for rows, seed in expected:
        record = files[f"r{rows}-s{seed}.json"]
        if (record["rows"], record["width"], record["seed"]) != (rows, run.WIDTH, seed):
            raise ValueError("Checkpoint identity differs.")
        for field in ("code_revision", "source_sha256"):
            if record[field] != first[field]:
                raise ValueError(f"Checkpoint provenance differs at {field}.")
        for field in ("environment", "asset"):
            if record[field] != baseline[field]:
                raise ValueError(f"The 1,024-row reference differs at {field}.")
        if record["codegen"]["ptx_instruction_sha256"] != baseline["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The compiled kernel differs from the frozen 1,024-row kernel.")
        elements = rows * run.WIDTH
        fixtures = verify_cases(record["fixtures"], CASES, elements)
        capture_expected = seed in protocol.TIMING_SEEDS and fixtures
        captures = verify_cases(record["replay_checks"], CAPTURES if capture_expected else (), elements)
        eligible = fixtures and captures
        if record["gate_passed"] != eligible or bool(record["timing"]) != (eligible and capture_expected):
            raise ValueError("The timing exclusion differs from the numerical evidence.")
        if record["timing"]:
            timing_medians(record)
        records[rows, seed] = record
    verify_attempts(folder, records)
    return records


def counts(records):
    cases = [case for record in records for case in (record["fixtures"] + record["replay_checks"])]
    return tuple(sum(case.get(key, 0) for case in cases)
                 for key in ("elements", "pairs_differ_bitwise", "bounded_rejected"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=run.HERE / "transfer-results")
    args = parser.parse_args()
    measurements = read_results(args.directory)
    print("rows seed normalized_elements bit_differences rejected vllm_us triton_us reduction latency_gate")
    for (rows, seed), result in sorted(measurements.items()):
        evidence = counts((result,))
        if result["timing"]:
            vllm, triton = timing_medians(result)
            decision = "pass" if result["timing"]["passes_5pct"] else "fail"
            print(rows, seed, *evidence, f"{vllm:.3f} {triton:.3f} {100*(1-triton/vllm):.3f}% {decision}")
        else:
            print(rows, seed, *evidence, "preflight" if seed == protocol.QUALIFICATION else "excluded")
