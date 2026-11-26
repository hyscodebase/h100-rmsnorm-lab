"""Publish an incremental, atomic local snapshot of the observer campaign on macOS or Linux."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import analyze

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab
import observer_plan


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def keys(record, expected):
    need(isinstance(record, dict) and set(record) == set(expected.split()), "Unexpected field in controller evidence.")


def read(path):
    need(path.is_file() and not path.is_symlink(), f"Expected a regular file at {path.name}.")
    def unique(pairs):
        obj = dict(pairs)
        need(len(pairs) == len(obj), "Duplicate JSON field.")
        return obj
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique), hashlib.sha256(raw).hexdigest()


def save(path, record):
    raw = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def utc(value):
    need(isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|\+00:00)", value),
         "Expected a UTC timestamp.")
    return analyze.utc(value)


def check_result(record, shape, revision):
    preflight = shape == ("preflight",)
    fields = "preflight passed physical_uuid_matched checks" if preflight else "seed arm timed_start_utc workload correctness timing observers"
    keys(record, "code_revision environment " + fields)
    need(record["code_revision"] == revision, "Checkpoint source revision differs from the pinned revision.")
    env = record["environment"]
    keys(env, " ".join(analyze.ENV_KEYS) + " observer_versions")
    versions = env["observer_versions"]
    keys(versions, "binding_version nvml_version driver_version nvidia_smi_version")
    # Hardware and software fields have narrow public forms; a host, UUID, file
    # path, arbitrary free text or an unexpected field must never reach the export.
    need(re.fullmatch(r"NVIDIA H100(?: [A-Za-z0-9-]+){0,4}", env["gpu"]) and len(env["gpu"]) <= 40
         and env["compute_capability"] == [9, 0]
         and all(type(env[k]) is int and env[k] > 0 for k in ("sm_count", "total_memory_bytes")),
         "Unexpected public H100 description.")
    need(all(re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,4}(?:\+cu\d+)?", env[k])
             for k in ("driver", "python", "torch", "cuda_runtime")), "Unexpected environment version.")
    need(versions["binding_version"] in (None, "nvml-ctypes-v1") and
         all(v is None or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,4}", v)
             for key, v in versions.items() if key != "binding_version"), "Unexpected observer version.")
    if preflight:
        keys(record["checks"], "cli nvml")
        for check in record["checks"].values():
            keys(check, "passed values")
            need(set(check["values"]) == analyze.METRICS, "Unexpected preflight metric.")
            for field, v in check["values"].items():
                need(v is None or (analyze.finite(v) and v >= 0 and (not field.endswith("_pct") or v <= 100)),
                     "Unexpected preflight reading.")
    else:
        utc(record["timed_start_utc"])
        keys(record["workload"], "operation dtype m k n seed operand_bytes bf16_reduced_precision_reduction reuse")
        keys(record["correctness"], "passed reference sampled_rows sampled_columns entries rtol atol max_absolute_error")
        keys(record["timing"], "requested_seconds warmup_s warmup_iterations batch_iterations wall_elapsed_s "
             "cuda_event_total_s iterations flops_per_iteration flops_formula wall_tflops cuda_event_tflops batches")
        need(record["timing"]["flops_formula"] == "2*M*N*K; one multiply-add counts as two operations",
             "Unexpected throughput formula.")
        for batch in record["timing"]["batches"]:
            keys(batch, "iterations event_elapsed_s")
    return observer_plan.validate_result(record, shape)


def terminal(record, path, digest, tasks, completed):
    fields = "attempt_id task_id experiment_hash started_utc reproduce_argv outcome"
    task, identifier, outcome = record["task_id"], record["attempt_id"], record["outcome"]
    need(task in tasks and record["experiment_hash"] == digest and re.fullmatch(r"[0-9a-f]{32}", identifier)
         and path.name == f"{task}-{identifier}.json", "Attempt identity mismatch.")
    permitted = list(tasks)[:len(completed)+1]
    need(task in permitted, "Attempt falls outside the completed schedule prefix and next pending unit.")
    expected = ["python3", observer_plan.SOURCE, *observer_plan.arguments(tasks[task])]
    need(record["reproduce_argv"] == expected, "Unexpected attempt arguments.")
    start = utc(record["started_utc"])
    if outcome == "started":
        keys(record, fields)
        return False
    fields += " finished_utc controller_elapsed_s"
    elapsed = record["controller_elapsed_s"]
    need(utc(record["finished_utc"]) >= start and analyze.finite(elapsed) and elapsed >= 0, "Invalid attempt timing.")
    if outcome == "checkpoint_saved":
        need(task in completed, "Attempt finished after the snapshot began. Retry the export.")
    elif outcome in ("interrupted", "failed"):
        fields += " error_type"
        error = record["error_type"]
        need(error in {"CalledProcessError", "TimeoutExpired", "RuntimeError", "ValueError", "JSONDecodeError",
                       "OSError", "FileNotFoundError", "PermissionError", "KeyError", "TypeError", "KeyboardInterrupt"}
             and (error == "KeyboardInterrupt") == (outcome == "interrupted"), "Unexpected controller exception.")
        if error == "CalledProcessError":
            fields += " exit_code"
            code = record["exit_code"]
            need(type(code) is int and code != 0 and -255 <= code <= 255, "Invalid process exit code.")
    else:
        raise ValueError("Unknown terminal outcome.")
    keys(record, fields)
    return True


def build(state, stage):
    plan, _ = read(state / "plan.json")
    need(plan == lab.read_json(ROOT / "plans/observer-coverage.json"), "State differs from the public observer plan.")
    lab.validate_plan(plan)
    manifest, _ = read(state / "manifest.json")
    keys(manifest, "version experiment_hash sources")
    sources = manifest["sources"]
    need(type(manifest["version"]) is int and manifest["version"] == 1 and set(sources) == set(observer_plan.SOURCES)
         and all(re.fullmatch("[0-9a-f]{64}", value) for value in sources.values()), "Invalid observer source manifest.")
    digest = hashlib.sha256(json.dumps([plan, sources], sort_keys=True, allow_nan=False).encode()).hexdigest()
    need(manifest["experiment_hash"] == digest, "Source manifest hash mismatch.")
    revision, _ = read(state / "observer-revision.json")
    need(re.fullmatch("[0-9a-f]{40}|[0-9a-f]{64}", revision) and
         all(lab.local_gemm_revision(ROOT, value, source, revision) == revision for source, value in sources.items()),
         "Cannot verify all three source blobs at the pinned Git commit.")
    save(stage / "plan.json", plan)
    save(stage / "manifest.json", manifest)
    save(stage / "observer-revision.json", revision)
    tasks = dict(observer_plan.tasks(plan))
    raw, attempts = state / "tasks", state / "attempts"
    need(raw.is_dir() and not raw.is_symlink() and not attempts.is_symlink(), "Expected regular state directories.")
    completed = {p.stem for p in raw.glob("*.json")}
    need(completed == set(list(tasks)[:len(completed)]), "Checkpoints must form a prefix of the registered task order.")
    environment = None
    for task in list(tasks)[:len(completed)]:
        envelope, checksum = read(raw / f"{task}.json")
        keys(envelope, "task_id experiment_hash completed_utc result")
        need(envelope["task_id"] == task and envelope["experiment_hash"] == digest, "Checkpoint identity mismatch.")
        utc(envelope["completed_utc"])
        env = check_result(envelope["result"], tasks[task], revision)
        need(environment in (None, env), "GPU environment changed since preflight.")
        environment = env
        result_hash = save(stage / f"{task}.json", envelope["result"])
        save(stage / "provenance" / f"{task}.json", {"task_id": task, "experiment_hash": digest,
             "completed_utc": envelope["completed_utc"], "controller_envelope_sha256": checksum,
             "exported_result_sha256": result_hash})
    analyze.load(stage)  # Cross-check sequence, nonoverlapping windows and environment across compute arms.
    terminal_count = 0
    for path in sorted(attempts.glob("*.json")):
        record, _ = read(path)
        if terminal(record, path, digest, tasks, completed):
            save(stage / "attempts" / path.name, record)
            terminal_count += 1
    need(completed or terminal_count, "No completed checkpoints or terminal attempts yet.")
    return len(completed), terminal_count


def exchange(left, right):
    # POSIX rename alone cannot replace a nonempty directory. These platforms
    # support exchanging two same-filesystem directory names atomically.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        status = libc.renamex_np(os.fsencode(left), os.fsencode(right), 2)  # RENAME_SWAP
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        status = libc.renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2)  # RENAME_EXCHANGE
    else:
        raise ValueError("Atomic directory exchange requires macOS or Linux; choose a new output directory.")
    if status:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def snapshot(state, output):
    state, output = state.resolve(strict=True), output.absolute()
    need(not output.is_symlink() and not state.is_relative_to(output.resolve()) and not output.resolve().is_relative_to(state),
         "Archive destination must be separate from the source state.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Cooperating publishers serialize on an existing directory, without adding a lockfile to the public archive.
    parent_fd = os.open(output.parent, os.O_RDONLY)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".observer-archive-") as tmp:
            stage = Path(tmp) / "snapshot"
            count, attempts = build(state, stage)
            if not output.exists():
                stage.rename(output)
            else:
                need(output.is_dir() and not output.is_symlink(), "Output is not a regular directory.")
                for previous in output.rglob("*"):
                    relative = previous.relative_to(output)
                    current = stage / relative
                    need(not previous.is_symlink() and
                         (previous.is_dir() and current.is_dir() or
                          previous.is_file() and current.is_file() and previous.read_bytes() == current.read_bytes()),
                         f"Archive conflict at {relative}; existing output was preserved.")
                if {p.relative_to(stage) for p in stage.rglob("*")} != {p.relative_to(output) for p in output.rglob("*")}:
                    exchange(stage, output)
    finally:
        os.close(parent_fd)
    return count, attempts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        count, attempts = snapshot(args.state_dir, args.output)
        print(f"Verified {count} checkpoints (including preflight when complete), {attempts} terminal attempts.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"Observer archive failed: {exc}")
