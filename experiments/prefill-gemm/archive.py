"""Create an immutable local snapshot of verified prefill checkpoints and terminal attempts."""

import argparse
import builtins
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from export_evidence import export
from summarize import ROOT, gemm_plan, lab, read_results


def regular_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file at {path}.")
    return path.read_bytes()


def unique_json(raw, name):
    def without_duplicates(pairs):
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError(f"Duplicate JSON key in {name}.")
        return values
    return json.loads(raw, object_pairs_hook=without_duplicates)


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|\+00:00)", value):
        raise ValueError("Expected a UTC timestamp in the controller record.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def checkpoint_bytes(path):
    # Copy original bytes so the published checksum identifies the actual run,
    # but reject extra metadata, including within individual samples and errors.
    raw = regular_bytes(path)
    cell = unique_json(raw, path.name)
    r = cell["result"]
    t = r["telemetry"]
    metric_keys = set(gemm_plan.TELEMETRY)
    schema = [
        (cell, "task_id experiment_hash result completed_utc"),
        (r, "timed_start_utc code_revision environment workload correctness timing telemetry"),
        (r["environment"], "gpu compute_capability total_memory_bytes sm_count driver cuda_runtime torch python"),
        (r["workload"], "operation dtype m k n seed operand_bytes bf16_reduced_precision_reduction reuse"),
        (r["correctness"], "passed reference sampled_rows sampled_columns entries rtol atol max_absolute_error"),
        (r["timing"], "requested_seconds warmup_s warmup_iterations batch_iterations wall_elapsed_s cuda_event_total_s "
                      "iterations flops_per_iteration flops_formula wall_tflops cuda_event_tflops batches"),
        (t, "scope requested_period_s samples summary queries_crossing_window_end query_errors"),
    ]
    for item, names in schema:
        if not isinstance(item, dict) or set(item) != set(names.split()):
            raise ValueError(f"Unexpected checkpoint fields in {path.name}.")
    # PyTorch versions can include an arbitrary local build label. Keep only
    # public release formats rather than copying a distributor's private text.
    public_version = r"[0-9]+(?:\.[0-9]+){1,3}"
    patterns = {
        "gpu": r"NVIDIA H100(?:[ -](?:SXM[45]?|PCIe|NVL|[0-9]{1,3}GB|HBM[23][eE]?))*",
        "driver": public_version, "cuda_runtime": public_version,
        "python": public_version + r"(?:(?:a|b|rc)[0-9]+)?",
        "torch": public_version + r"(?:(?:a|b|rc)[0-9]+)?(?:\.(?:dev|post)[0-9]+)?(?:\+cu[0-9]+)?",
    }
    for key, pattern in patterns.items():
        value = r["environment"][key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"Unexpected public environment field {key} in {path.name}.")
    if (r["timing"]["flops_formula"] != "2*M*N*K; one multiply-add counts as two operations"
            or t["scope"] != "Device-wide nvidia-smi readings; mean and maximum over observed in-window samples"
            or type(t["requested_period_s"]) is not int or t["requested_period_s"] != 1
            or type(t["queries_crossing_window_end"]) is not int or t["queries_crossing_window_end"] < 0
            or not isinstance(t["query_errors"], list)):
        raise ValueError(f"Unexpected benchmark protocol metadata in {path.name}.")
    for items, keys in ((r["timing"]["batches"], {"iterations", "event_elapsed_s"}),
                        (t["samples"], {"query_start_s", "query_end_s", *metric_keys}),
                        (t["summary"].values(), {"count", "mean", "peak_observed"}),
                        (t["query_errors"], {"query_start_s", "error_type"})):
        if any(set(item) != keys for item in items):
            raise ValueError(f"Unexpected measurement fields in {path.name}.")
    error_types = {"ValueError", "OSError", "FileNotFoundError", "PermissionError", "TimeoutExpired",
                   "CalledProcessError", "SubprocessError"}
    if (set(t["summary"]) != metric_keys or any(e["error_type"] not in error_types or
            type(e["query_start_s"]) not in (int, float) or not math.isfinite(e["query_start_s"])
            or e["query_start_s"] < 0 for e in t["query_errors"])):
        raise ValueError(f"Unexpected telemetry metadata in {path.name}.")
    timestamp(cell["completed_utc"])
    timestamp(r["timed_start_utc"])
    return raw


def terminal_attempt(path, plan, tasks, digest, completed):
    data = unique_json(regular_bytes(path), path.name)
    task, identifier = data["task_id"], data["attempt_id"]
    if (task not in tasks or not isinstance(identifier, str) or not re.fullmatch("[0-9a-f]{32}", identifier)
            or path.name != f"{task}-{identifier}.json" or data["experiment_hash"] != digest):
        raise ValueError(f"Attempt identity mismatch in {path.name}.")
    expected = ["python3", gemm_plan.SOURCE, *gemm_plan.arguments(tasks[task], plan)]
    if data["reproduce_argv"] != expected:
        raise ValueError(f"Unexpected reproduction command in {path.name}.")
    fields = {"attempt_id", "task_id", "experiment_hash", "started_utc", "reproduce_argv", "outcome"}
    start = timestamp(data["started_utc"])
    outcome = data["outcome"]
    if outcome == "started" and set(data) == fields:
        return None
    fields.update(("finished_utc", "controller_elapsed_s"))
    duration = data["controller_elapsed_s"]
    if (timestamp(data["finished_utc"]) < start or type(duration) not in (float, int)
            or not math.isfinite(duration) or duration < 0):
        raise ValueError(f"Invalid attempt duration in {path.name}.")
    if outcome == "checkpoint_saved":
        if task not in completed:
            raise ValueError(f"Saved attempt lacks a matching checkpoint for {task}. Retry if the state advanced.")
    elif outcome in ("failed", "interrupted"):
        fields.add("error_type")
        error = data["error_type"]
        klass = getattr(builtins, error, None) if isinstance(error, str) else None
        known_error = isinstance(klass, type) and issubclass(klass, BaseException)
        if (not known_error and error not in ("CalledProcessError", "TimeoutExpired", "SubprocessError", "JSONDecodeError")
                or (outcome == "interrupted") != (error == "KeyboardInterrupt")):
            raise ValueError(f"Unexpected error type in {path.name}.")
        if error == "CalledProcessError":
            fields.add("exit_code")
            code = data["exit_code"]
            if type(code) is not int or code == 0 or not -255 <= code <= 255:
                raise ValueError(f"Invalid process exit code in {path.name}.")
    else:
        raise ValueError(f"Unexpected attempt outcome in {path.name}.")
    if set(data) != fields:
        raise ValueError(f"Unexpected attempt fields in {path.name}.")
    return json.dumps(data, indent=2, allow_nan=False) + "\n"


def snapshot(state, output, usage=None):
    state = state.resolve(strict=True)
    output = output.absolute()
    if output.is_symlink() or output.resolve().is_relative_to(state) or state.is_relative_to(output.resolve()):
        raise ValueError("Use a separate output directory outside the source state.")
    if usage is not None and not usage.is_dir():
        raise ValueError("The supplemental usage directory does not exist.")
    plan = unique_json(regular_bytes(state / "plan.json"), "plan.json")
    expected_plan = lab.read_json(ROOT / "plans/prefill-gemm.json")
    if plan != expected_plan:
        raise ValueError("The state does not match the public prefill campaign plan.")
    manifest = regular_bytes(state / "manifest.json")
    metadata = unique_json(manifest, "manifest.json")
    if (not isinstance(metadata, dict) or set(metadata) != {"version", "experiment_hash", "sources"}
            or type(metadata["version"]) is not int or metadata["version"] != 1
            or not isinstance(metadata["sources"], dict) or set(metadata["sources"]) != {gemm_plan.SOURCE}
            or not all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
                       for v in (metadata["experiment_hash"], metadata["sources"][gemm_plan.SOURCE]))):
        raise ValueError("Unexpected source manifest fields.")
    source = state / "tasks"
    attempts = state / "attempts"
    if not source.is_dir() or source.is_symlink() or attempts.is_symlink():
        raise ValueError("Expected a local state with a regular task directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gemm-archive-") as temporary:
        stage = Path(temporary) / "snapshot"
        results, evidence = stage / "results", stage / "evidence"
        results.mkdir(parents=True)
        (results / "manifest.json").write_bytes(manifest)
        for path in sorted(source.glob("*.json")):
            (results / path.name).write_bytes(checkpoint_bytes(path))
        # Validate the captured bytes, including each recorded Git blob, before publishing.
        tasks, records = read_results(results)
        if not records and metadata != lab.manifest(plan):
            raise ValueError("A failed-only source manifest requires the matching benchmark checkout.")
        completed = {name for name, shape in tasks.items() if shape in records}
        count = 0
        for path in sorted(attempts.glob("*.json")):
            text = terminal_attempt(path, plan, tasks, metadata["experiment_hash"], completed)
            if text is not None:
                target = evidence / "attempts" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                count += 1
        if records:
            export(results, evidence, usage)
        elif count == 0:
            raise ValueError("No completed checkpoints or terminal attempts to archive.")
        if output.exists():
            staged_files = {p.relative_to(stage) for p in stage.rglob("*") if p.is_file()}
            existing = list(output.rglob("*")) if output.is_dir() else []
            existing_files = {p.relative_to(output) for p in existing if p.is_file()}
            if (any(p.is_symlink() for p in existing) or staged_files != existing_files
                    or any((stage / p).read_bytes() != (output / p).read_bytes() for p in staged_files)):
                raise ValueError("A different snapshot already exists at the destination. Choose a new directory.")
        else:
            stage.rename(output)
    print(f"Verified {len(records)} checkpoints and {count} terminal attempts at {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path, help="prefill state created by lab.py")
    parser.add_argument("--output", required=True, type=Path, help="new snapshot directory, or an identical prior snapshot")
    parser.add_argument("--usage-dir", type=Path, help="optional private telemetry; only in-window metric columns are exported")
    args = parser.parse_args()
    try:
        snapshot(args.state_dir, args.output, args.usage_dir)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        sys.exit(f"Archive failed: {exc}")
