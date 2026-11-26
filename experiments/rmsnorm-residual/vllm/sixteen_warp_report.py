"""Check frozen 16-warp evidence and recompute the three-way timing."""
import argparse
import hashlib
import json
from pathlib import Path

import run
import sixteen_warp_run as protocol
from transfer_report import CASES, CAPTURES, counts, timestamp, timing_medians, verify_cases, verify_revision


def verify_attempts(folder, data, *, remote_clock_bounds_s=(0.0, 0.0)):
    earliest_offset, latest_offset = remote_clock_bounds_s
    completed = set()
    for path in sorted((folder / "attempts").glob("*.json")):
        attempt = json.loads(path.read_bytes())
        row, seed, token = attempt["rows"], attempt["seed"], attempt["attempt_id"]
        if row != protocol.ROWS or seed not in data or path.stem != f"r{row}-s{seed}-{token}":
            raise ValueError("Unexpected attempt identity.")
        first = timestamp(attempt["started_utc"])
        phase = attempt["outcome"]
        if phase == "started":
            continue
        last = timestamp(attempt["finished_utc"])
        if phase not in ("checkpoint_saved", "checkpoint_recovered", "failed") or not first <= last:
            raise ValueError("Unexpected attempt outcome or timestamps.")
        if phase == "failed":
            verify_revision(run.ROOT, attempt["code_revision"], ())
            continue
        result = data[seed]
        started, finished = timestamp(result["started_utc"]), timestamp(result["finished_utc"])
        # The attempt log uses the local clock and the checkpoint uses the GPU host clock.
        earliest_offset = max(earliest_offset, (finished - last).total_seconds())
        if phase == "checkpoint_saved":
            latest_offset = min(latest_offset, (started - first).total_seconds())
        if (seed in completed or attempt["code_revision"] != result["code_revision"]
                or started > finished or earliest_offset > latest_offset):
            raise ValueError("Invalid or duplicate completed attempt.")
        completed.add(seed)
    if completed != data.keys():
        raise ValueError("Every checkpoint needs a recorded successful submission or recovery.")


def verify_pairwise(records, names, backends=protocol.BACKENDS):
    for case in records:
        if "reason" not in case and set(case["comparisons"]) != set(backends[1:]):
            raise ValueError("A candidate comparison is missing.")
    for backend in backends[1:]:
        fields = [case["comparisons"][backend] if "comparisons" in case else case for case in records]
        verify_cases(fields, names, protocol.ROWS * run.WIDTH)


def read_results(folder):
    files = {path.name: json.loads(path.read_bytes()) for path in folder.glob("s*.json")}
    first = files[f"s{protocol.NUMERIC_SEEDS[0]}.json"]
    expected = protocol.SEEDS if first["gate_passed"] else protocol.NUMERIC_SEEDS
    if files.keys() != {f"s{seed}.json" for seed in expected}:
        raise ValueError("The result set differs from the registered seeds.")
    anchor = json.loads(protocol.ANCHOR.read_bytes())
    sources, revision = first["source_sha256"], first["code_revision"]
    if sources.keys() != set(protocol.SOURCES):
        raise ValueError("The recorded source manifest differs.")
    verify_revision(run.ROOT, revision, tuple(sorted(sources.items())))

    data = {}
    for seed in expected:
        record = files[f"s{seed}.json"]
        if (record["rows"], record["width"], record["seed"]) != (protocol.ROWS, run.WIDTH, seed):
            raise ValueError("Result identity differs.")
        if (record["source_sha256"] != sources or record["code_revision"] != revision
                or record["asset"] != anchor["asset"] or record["environment"] != anchor["environment"]):
            raise ValueError("Result provenance differs.")
        eight, sixteen = (record["codegen"][name] for name in protocol.BACKENDS[1:])
        if eight["ptx_instruction_sha256"] != anchor["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The eight-warp compilation changed.")
        if (sixteen["num_warps"], sixteen["required_threads"], sixteen["warp_shuffle_offsets"],
                sixteen["emitted_shared_loads"], sixteen["ptx_opcodes"].get("bar.sync")) != (16, 512, [1, 2, 4, 8, 16], 4, 1):
            raise ValueError("The emitted 16-warp reduction differs.")
        if sixteen["ptx_instruction_sha256"] != first["codegen"][protocol.CANDIDATE]["ptx_instruction_sha256"]:
            raise ValueError("The candidate changed after qualification.")
        fixtures, captures = record["fixtures"], record["replay_checks"]
        verify_pairwise(fixtures, CASES)
        verify_pairwise(captures, CAPTURES if all(case["passed"] for case in fixtures) else ())
        eligible = protocol.numerically_eligible(fixtures, captures)
        timed = eligible and seed in protocol.TIMING_SEEDS
        if record["gate_passed"] != eligible or bool(record["timing"]) != timed:
            raise ValueError("Numerical or timing admission differs.")
        if timed:
            timing_medians(record, protocol.BACKENDS)
        if record["comparisons"] != protocol.comparisons(record["timing"]):
            raise ValueError("Timing comparisons differ from the recorded intervals.")
        data[seed] = record
    verify_attempts(folder, data)
    static = json.loads((folder / "compile-static.json").read_bytes())
    if static["binary_sha256"] != first["codegen"][protocol.CANDIDATE]["binary_sha256"]:
        raise ValueError("The offline compiler and H100 loaded different binaries.")
    recovered = json.loads((folder / "checkpoint-recovery.json").read_bytes())
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (recovered["seed"] != first["seed"] or recovered["rows"] != first["rows"]
            or recovered["code_revision"] != revision or recovered["result_origin"] != "cached"
            or not recovered["matches_local_result"]
            or recovered["canonical_json_sha256"] != hashlib.sha256(canonical).hexdigest()):
        raise ValueError("The recovered result differs from the archived preflight.")
    return data


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=run.HERE / "sixteen-warp-results")
    args = cli.parse_args()
    data = read_results(args.directory)
    print("seed elements_per_candidate differing_bits rejected vllm_us eight_warp_us sixteen_warp_us vs_vllm vs_eight")
    for seed, record in sorted(data.items()):
        evidence = {"fixtures": [r.get("comparisons", {}).get(protocol.CANDIDATE, r) for r in record["fixtures"]],
                    "replay_checks": [r.get("comparisons", {}).get(protocol.CANDIDATE, r) for r in record["replay_checks"]]}
        values = (seed, *counts((evidence,)))
        if record["timing"]:
            medians = [f"{m:.3f}" for m in timing_medians(record, protocol.BACKENDS)]
            effects = [f"{100 * record['comparisons'][name]['latency_reduction']:.3f}%" for name in protocol.BACKENDS[:-1]]
            print(*values, *medians, *effects)
        else:
            print(*values, "preflight" if record["gate_passed"] else "excluded")
