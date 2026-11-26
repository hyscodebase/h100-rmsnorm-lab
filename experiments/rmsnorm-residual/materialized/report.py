"""Verify frozen checkpoints and reproduce the precision-matched comparison."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import lab
import materialized_plan as protocol

require = lab.require


def read_records(directory):
    plan = lab.read_json(directory / "plan.json")
    require(plan == lab.read_json(ROOT / "plans/materialized-rmsnorm.json"), "Unexpected campaign plan.")
    lab.validate_plan(plan)
    manifest = lab.read_json(directory / "manifest.json")
    sources = manifest["sources"]
    require(manifest["version"] == 1 and set(sources) == set(protocol.SOURCES)
            and all(re.fullmatch("[0-9a-f]{64}", value) for value in sources.values()), "Unexpected source manifest.")
    digest = hashlib.sha256(json.dumps([plan, sources], sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(manifest["experiment_hash"] == digest, "Experiment manifest hash does not match plan and source hashes.")
    expected = list(lab.tasks(plan))
    present = {p.stem for p in (directory / "tasks").glob("*.json")}
    require(present == {task for task, _ in expected[:len(present)]}, "Unexpected checkpoints or a gap in execution order.")
    lab.progress(directory, plan, digest)
    return {shape[0]: lab.read_json(directory / "tasks" / f"{task}.json")["result"]
            for task, shape in expected if task in present}


def decision(records):
    if not records:
        return "Waiting for the numerical preflight."
    if not records[protocol.SEEDS[0]]["gate_passed"]:
        return "The numerical gate failed. No timing checkpoints are eligible."
    if set(records) != set(protocol.SEEDS):
        return "The numerical preflight passed. The decision awaits all three timing checkpoints."
    comparisons = [records[seed]["comparison"] for seed in protocol.SEEDS[1:]]
    if any(not comp or not comp["eligible"] for comp in comparisons):
        return "The latency decision is undetermined. A timing seed did not have both valid compiled reference styles."
    if all(comp["passes_5pct"] for comp in comparisons):
        return "The four-warp Triton implementation meets the 5% latency criterion in all three seeds."
    return "The four-warp Triton implementation does not meet the 5% latency criterion in all three seeds."


def show(records):
    print(decision(records))
    preflight = records.get(protocol.SEEDS[0])
    if preflight is None:
        return
    print("\n| Preflight backend | Outcome |\n| --- | --- |")
    for name, info in preflight["backends"].items():
        failure = info.get("correctness", {}).get("failure")
        outcome = "Verified" if info["status"] == "verified" else info["reason"]
        if failure:
            outcome += f" ({failure['fixture']})"
        print(f"| {name} | {outcome} |")
    print("\n| Seed | Median Triton µs | Fastest correct compiled | Median compiled µs | Latency reduction |")
    print("| ---: | ---: | --- | ---: | ---: |")
    for seed in protocol.SEEDS[1:]:
        if seed not in records:
            continue
        comp = records[seed]["comparison"]
        if comp is None:
            print(f"| {seed} | — | Ineligible | — | — |")
        else:
            print(f"| {seed} | {comp['candidate_median_us']:.3f} | {comp['fastest_correct_compiled']} | "
                  f"{comp['reference_median_us']:.3f} | {comp['latency_reduction_pct']:.2f}% |")


def attempts(state, records):
    plan = lab.read_json(state / "plan.json")
    manifest = lab.read_json(state / "manifest.json")
    tasks = dict(lab.tasks(plan))
    for path in sorted((state / "attempts").glob("*.json")):
        d = lab.read_json(path)
        if d["outcome"] == "started":
            continue
        key, identifier = d["task_id"], d["attempt_id"]
        allowed = {"attempt_id", "task_id", "experiment_hash", "started_utc", "finished_utc",
                   "controller_elapsed_s", "reproduce_argv", "outcome", "error_type", "exit_code"}
        require(key in tasks and re.fullmatch("[0-9a-f]{32}", identifier)
                and path.name == f"{key}-{identifier}.json" and set(d) <= allowed
                and d["experiment_hash"] == manifest["experiment_hash"]
                and d["reproduce_argv"] == ["python3", protocol.SOURCE, *protocol.arguments(tasks[key])]
                and d["outcome"] in {"checkpoint_saved", "failed", "interrupted"}
                and (d["outcome"] != "checkpoint_saved" or tasks[key][0] in records)
                and math.isfinite(d["controller_elapsed_s"]) and d["controller_elapsed_s"] >= 0
                and ("error_type" not in d or re.fullmatch("[A-Za-z]+", d["error_type"]))
                and ("exit_code" not in d or type(d["exit_code"]) is int), "Unexpected controller attempt record.")
        require(protocol.residual_plan.timestamp(d["started_utc"]) <= protocol.residual_plan.timestamp(d["finished_utc"]),
                "Unexpected controller attempt time.")
        yield path


def archive(state, destination, records):
    state, destination = state.resolve(), destination.resolve()
    require(not destination.is_relative_to(state), "Choose an archive destination outside the run state.")
    raw = [state / "manifest.json", state / "plan.json", *sorted((state / "tasks").glob("*.json")),
           *attempts(state, records)]
    copies = []
    for source in raw:
        data = source.read_bytes()
        require(not re.search(rb'/Users/|/root/|/home/|(?:GPU|MIG)-[0-9a-fA-F]{8}-|'
                              rb'PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{30,}', data),
                f"Review unfiltered metadata in {source.name} before publication.")
        target = destination / source.relative_to(state)
        require(not target.exists() or target.read_bytes() == data, f"Published record differs at {target.name}.")
        copies.append((data, target))
    for data, target in copies:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        require(target.read_bytes() == data, f"Published record differs at {target.name}.")
                finally:
                    temporary.unlink(missing_ok=True)
    read_records(destination)
    print(f"Verified {len(records)} checkpoints and {len(copies)} archived files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=HERE / "results")
    parser.add_argument("--archive-to", type=Path, help="Copy original checkpoints and terminal attempts to this directory.")
    args = parser.parse_args()
    try:
        records = read_records(args.directory)
        if args.archive_to:
            archive(args.directory, args.archive_to, records)
        show(records)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"Materialized report failed: {exc}")


if __name__ == "__main__":
    main()
