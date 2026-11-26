"""Reproduce the residual-kernel adoption decision from validated controller envelopes."""

import argparse
import hashlib
import json
import re
import statistics
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import residual_plan as protocol

TARGET = 1024


def read_results(directory):
    if directory.name == "tasks" and (directory.parent / "manifest.json").is_file():
        directory = directory.parent
    plan = json.loads((ROOT / "plans/residual-rmsnorm.json").read_text())
    protocol.validate_plan(plan)
    local_plan = directory / "plan.json"
    if local_plan.exists():
        protocol.require(json.loads(local_plan.read_text()) == plan, "The state has a different experiment plan.")
    manifest = json.loads((directory / "manifest.json").read_text())
    sources = manifest["sources"]
    protocol.require(manifest["version"] == 1 and set(sources) == set(protocol.SOURCES)
                     and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values()), "Unexpected source manifest.")
    digest = hashlib.sha256(json.dumps([plan, sources], sort_keys=True, allow_nan=False).encode()).hexdigest()
    protocol.require(manifest["experiment_hash"] == digest, "Experiment hash differs from the plan and source hashes.")
    raw = directory / "tasks" if (directory / "tasks").is_dir() else directory
    expected = dict(protocol.tasks(plan))
    observed = {p.stem for p in raw.glob("s*-r*-w*.json")}
    protocol.require(observed <= expected.keys(), "Unexpected shape/seed checkpoint.")
    pin = directory / "residual-revision.json"
    frozen = json.loads(pin.read_text()) if pin.is_file() else None
    hardware = None
    cells = {}
    for name, shape in expected.items():
        if name not in observed:
            continue
        path = raw / f"{name}.json"
        try:
            envelope = json.loads(path.read_text())
            protocol.require(envelope["task_id"] == name and envelope["experiment_hash"] == digest, "Envelope mismatch.")
            protocol.timestamp(envelope["completed_utc"])
            r = envelope["result"]
            protocol.verify_sources(r, ROOT, sources)
            protocol.require(frozen is None or frozen == r["code_revision"], "Pinned source revision differs.")
            protocol.validate_result(r, shape, plan)
            env = r["environment"]
            protocol.require("H100" in env["gpu"] and env["compute_capability"] == [9, 0]
                             and all(isinstance(env[k], str) and env[k].strip()
                                     for k in ("driver", "cuda_runtime", "torch", "triton", "python"))
                             and all(type(env[k]) is int and env[k] > 0 for k in ("sm_count", "total_memory_bytes")),
                             "Invalid H100 environment.")
            protocol.require(hardware is None or hardware == env, "Hardware or software varied across checkpoints.")
            hardware = env
            cells[shape] = (r, path.relative_to(directory))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    return plan, cells


def median_ms(info):
    if info["status"] != "measured":
        return None
    # The 21 intervals have one middle observation; convert its serialized
    # decimal to an exact rational for comparisons at the 5% boundary.
    return statistics.median(Decimal(str(ms)) for ms in info["timing"]["graph_elapsed_ms"])


def compare(result):
    backends = result["backends"]
    medians = {name: median_ms(info) for name, info in backends.items()}
    viable = [key for key in protocol.COMPILED if medians[key] is not None]
    best = min(viable, key=medians.get) if viable else None
    reference = medians[best] if best else None
    candidate = medians[protocol.CANDIDATE]
    eligible = all(v is not None for v in (candidate, reference, medians[protocol.EAGER]))
    passed = 100 * candidate <= 95 * reference if eligible else None
    reduction = 100 * (1 - candidate / reference) if eligible else None
    return medians, best, passed, reduction


def correctness_failure(info):
    if info.get("correctness", {}).get("passed") is False:
        return True
    return info.get("status") == "excluded" and "AssertionError" in info.get("error", {}).get("exception_types", [])


def verdict(plan, cells):
    failures = [key for key, (r, _) in cells.items() if correctness_failure(r["backends"][protocol.CANDIDATE])]
    if failures:
        where = ", ".join(f"seed {s}, {rows} rows" for s, rows in failures)
        return f"Candidate correctness failed at {where}. Its performance comparison is ineligible."
    target_passes = [compare(cells[s, TARGET][0])[2] for s in plan["seeds"] if (s, TARGET) in cells]
    if len(cells) < len(plan["seeds"]) * len(plan["rows"]):
        return (f"Partial evidence. {target_passes.count(True)} target seeds meet the gate so far. "
                "The adoption decision is pending the full set of numerical checks and checkpoints.")
    verified = all(r["backends"][n]["status"] == "measured" for r, _ in cells.values()
                   for n in (protocol.EAGER, protocol.CANDIDATE))
    if not verified or None in target_passes:
        return "Speedup against compiled PyTorch is undetermined. Required numerical verification or a target reference is unavailable."
    if all(target_passes) and len(target_passes) == 3:
        return "Recommend the four-warp Triton kernel at 1,024 × 4,096 on this H100. All three target seeds meet the 5% latency gate."
    return f"Retain the compiled path at 1,024 × 4,096. {target_passes.count(True)} of three target seeds meet the 5% latency gate."


def reason(info):
    failure = info.get("correctness", {}).get("failure")
    if failure:
        return f"{failure['fixture']} / {failure['reason']}"
    error = info["error"]
    detail = error["detail"].replace("|", "\\|").replace("\n", " ")
    return f"{info['excluded_at']} / {' → '.join(error['exception_types'])} {detail}".strip()


def show(plan, cells):
    print(f"Completed {len(cells)}/{len(plan['rows']) * len(plan['seeds'])} shape/seed checkpoints. "
          "Times below are median microseconds per two-output call.\n")
    print(verdict(plan, cells))
    print("\n| Rows | Seed | Eager µs | Triton µs | Fastest correct compiled mode | Compiled µs | Latency reduction | Result |")
    print("|---:|---:|---:|---:|---|---:|---:|---|")
    for rows in plan["rows"]:
        for seed in plan["seeds"]:
            if (seed, rows) not in cells:
                continue
            r, path = cells[seed, rows]
            medians, best, passed, reduction = compare(r)
            def us(key):
                value = medians.get(key)
                return "—" if value is None else f"{value * 1000 / plan['ops']:.4f}"
            if passed is None:
                status = "Ineligible"
            elif rows == TARGET:
                status = "Target meets 5%" if passed else "Target below 5%"
            else:
                status = "Control win" if reduction > 0 else ("Control loss" if reduction < 0 else "Control tie")
            delta = "—" if reduction is None else f"{reduction:+.3f}%"
            label = best or "—"
            if r["comparison"]["native_only_compiled_reference"]:
                label += " (native only)"
            print(f"| {rows} | [{seed}]({path.as_posix()}) | {us(protocol.EAGER)} | {us(protocol.CANDIDATE)} | "
                  f"{label} | {us(best)} | {delta} | {status} |")
    excluded = [(seed, rows, name, reason(info)) for (seed, rows), (r, _) in cells.items()
                for name, info in r["backends"].items() if info["status"] == "excluded"]
    if excluded:
        print("\n| Seed | Rows | Excluded backend | Reason |\n|---:|---:|---|---|")
        for seed, rows, name, message in excluded:
            print(f"| {seed} | {rows} | {name} | {message} |")
    print("\nSeed links contain all four compiled attempts and every measured CUDA-event interval. "
          "Controls report wins or losses against their own fastest correct compiled reference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="lab state, its tasks directory, or a flat archive with manifest.json")
    try:
        show(*read_results(parser.parse_args().directory))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        sys.exit(f"Residual summary failed: {exc}")
