"""Run or recover exactly one frozen sustained-throughput H100 checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab

STATE = ROOT / ".lab/sustained-throughput"
RAW = ROOT / "results/sustained-throughput/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
RUNNER = ROOT / "experiments/sustained-throughput/ssh_runner.sh"
PLAN = json.loads((ROOT / "plans/sustained-throughput.json").read_text())
COMPUTE_SOURCES = (
    "kernels.py", "experiments/sustained-throughput/run.py",
    "experiments/sustained-throughput/telemetry.py",
    "experiments/sustained-throughput/PLAN.md", "plans/sustained-throughput.json",
)
EXECUTION_SOURCES = (*COMPUTE_SOURCES,
    "experiments/sustained-throughput/step.py",
    "experiments/sustained-throughput/ssh_runner.sh",
    "experiments/sustained-throughput/summarize.py")
PAUSE_MARKERS = (STATE / "paused", ROOT / ".lab/vllm/paused")


def telemetry_only_failure(record):
    """Recognize only the saved first-candidate telemetry-gap failure."""
    try:
        phase, failure = record["sustained_phases"][0], record["failure"]
        correctness = record["correctness"]
        if (record.get("status") != "gate_failed"
                or len(record.get("sustained_phases", [])) != 1
                or failure.get("stage") != "sustained_sync:triton_4w_4r"
                or phase.get("backend") != "triton_4w_4r"
                or phase.get("telemetry_gate_passed") is not False
                or phase.get("telemetry_sample_count", 0) < PLAN["minimum_active_telemetry_samples"]
                or phase.get("telemetry_max_gap_seconds", 0) <= PLAN["maximum_active_telemetry_gap_seconds"]):
            return False
        coverage = phase.get("telemetry_field_coverage", {})
        if any(coverage.get(field, 0) < PLAN["telemetry_required_field_coverage_fraction"]
               for field in PLAN["telemetry_required_fields"]):
            return False
        durations = phase["window_event_seconds"]
        gaps = phase["window_event_gaps_seconds"]
        if (len(durations) != PLAN["sustained_event_windows_per_phase"]
                or len(gaps) != len(durations) - 1
                or not PLAN["sustained_min_seconds"] <= phase["event_seconds"] <= PLAN["sustained_max_seconds"]
                or max(durations) > PLAN["maximum_event_window_seconds"]
                or max(gaps) > PLAN["maximum_event_gap_between_windows_seconds"]):
            return False
        if len(record.get("latency_intervals", [])) != PLAN["latency_rounds"] * 4:
            return False
        checks = [
            ("fixtures", len(PLAN["fixtures"]) * 4),
            ("timed_input", 4), ("post_latency_replay", 4), ("post_sustained_replay", 1),
        ]
        for section, expected_count in checks:
            rows = correctness.get(section, [])
            if len(rows) != expected_count or not all(
                    item.get("strict_passed") and item.get("crosscheck_passed") for item in rows):
                return False
        memory = record["memory"]
        if (memory["before_allocations"]["free_bytes"] < PLAN["free_memory_guard_bytes"]
                or memory["after_persistent_allocation"]["free_bytes"] < PLAN["free_memory_guard_bytes"]
                or memory["capture_reserved_delta_peak_bytes"] > PLAN["capture_reserved_delta_ceiling_bytes"]):
            return False
        return True
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def validate_telemetry_retry_archive(record, receipt_path, receipt):
    attempt_id = record.get("attempt_id")
    archive = RAW / "archives" / "s50-w4103" / f"attempt-{attempt_id}"
    manifest_path = archive / "manifest.json"
    archived_checkpoint = archive / "checkpoint.json"
    archived_attempt = archive / "attempt.json"
    if not manifest_path.is_file() or not archived_checkpoint.is_file() or not archived_attempt.is_file():
        raise SystemExit("Telemetry-only retry archive is incomplete; stop before submission.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    checkpoint_sha = hashlib.sha256(archived_checkpoint.read_bytes()).hexdigest()
    attempt_sha = hashlib.sha256(archived_attempt.read_bytes()).hexdigest()
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if (manifest.get("attempt_id") != attempt_id
            or manifest.get("original_checkpoint_sha256") != checkpoint_sha
            or manifest.get("original_attempt_receipt_sha256") != attempt_sha
            or attempt_sha != receipt_sha
            or manifest.get("gate_failure_classification") != "telemetry_only"
            or receipt.get("checkpoint_sha256") != checkpoint_sha):
        raise SystemExit("Telemetry-only retry archive checksum/link mismatch; stop before submission.")
    return str(manifest_path.relative_to(ROOT)), hashlib.sha256(manifest_bytes).hexdigest(), checkpoint_sha


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with temp.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in EXECUTION_SOURCES}


def verify_source_pin():
    if not SOURCE_PIN.is_file():
        raise SystemExit("Frozen source pin is missing; do not submit H100 work.")
    pin = json.loads(SOURCE_PIN.read_text())
    current = hashes()
    if pin.get("source_sha256") != current:
        raise SystemExit("Working source differs from the frozen source pin.")
    revision = pin.get("source_revision", "")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SystemExit("Source pin lacks a full commit hash.")
    for name, digest in current.items():
        try:
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{name}"])
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"Pinned source is absent from frozen commit: {name}") from error
        if hashlib.sha256(blob).hexdigest() != digest:
            raise SystemExit(f"Frozen commit hash mismatch: {name}")
    return revision, current, pin


def work_units():
    return [(seed, width) for seed in PLAN["seeds"] for width in PLAN["widths"]]


def is_paused():
    return any(path.exists() for path in PAUSE_MARKERS)


def show_status():
    saved = {}
    if CHECKPOINTS.exists():
        for path in CHECKPOINTS.glob("s*-w*.json"):
            try:
                saved[path.stem] = json.loads(path.read_text()).get("status")
            except (OSError, json.JSONDecodeError):
                saved[path.stem] = "unreadable"
    units = work_units()
    completed = {unit for unit, status in saved.items() if status == "complete"}
    failed = sorted(unit for unit, status in saved.items() if status != "complete")
    next_unit = (None if failed else next((f"s{seed}-w{width}" for seed, width in units
                                           if f"s{seed}-w{width}" not in completed), None))
    markers = [str(path.relative_to(ROOT)) for path in PAUSE_MARKERS if path.exists()]
    print(f"Completed: {len(completed)}/{len(units)} checkpoints.")
    print(f"Saved failed/incomplete: {len(failed)}")
    print(f"Blocking checkpoint(s): {', '.join(failed) if failed else 'none'}")
    next_label = f"blocked by {', '.join(failed)}" if failed else (next_unit or "none")
    print(f"Next: {next_label}")
    print(f"Paused: {', '.join(markers) if markers else 'no'}")


def run_one(lock, retry_transport_failures=False, retry_telemetry_only=False):
    if is_paused():
        raise SystemExit("Paused at checkpoint boundary; no H100 work submitted.")
    if retry_transport_failures and retry_telemetry_only:
        raise SystemExit("Choose only one manual retry gate.")
    revision, source_digests, pin = verify_source_pin()
    if pin.get("h100_execution_authorized") is not True:
        raise SystemExit("Source pin is plan-only; H100 execution is not authorized.")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    saved_records = {}
    for path in CHECKPOINTS.glob("s*-w*.json"):
        try:
            saved_records[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot inspect saved checkpoint {path.name}; stop before submission.") from error
    failed_checkpoints = sorted(unit for unit, item in saved_records.items()
                                if item.get("status") != "complete")
    if retry_telemetry_only:
        name = PLAN["manual_telemetry_retry_unit"]
        if (failed_checkpoints != [name] or set(saved_records) != {name}
                or not telemetry_only_failure(saved_records[name])):
            raise SystemExit("Telemetry-only recovery requires the sole saved failure to match the registered telemetry-gate artifact.")
        unit = (PLAN["seeds"][0], PLAN["widths"][0])
    elif failed_checkpoints:
        raise SystemExit("A saved checkpoint failed; inspect its evidence before continuing: "
                         + ", ".join(failed_checkpoints))
    else:
        unit = next(((seed, width) for seed, width in work_units()
                     if f"s{seed}-w{width}" not in saved_records), None)
    if unit is None:
        print("All 12 registered checkpoints are saved.")
        return
    seed, width = unit
    name = f"s{seed}-w{width}"
    receipts = []
    for path in sorted(ATTEMPTS.glob(f"{name}-*.json")):
        try:
            receipts.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot inspect prior attempt receipt {path.name}; stop before submission.") from error
    receipts.sort(key=lambda item: item.get("started_utc", ""))
    standard_limit = PLAN["standard_max_attempts_per_checkpoint"]
    total_limit = PLAN["max_attempts_per_checkpoint"]
    manual_transport_retry = False
    manual_telemetry_retry = False
    retry_archive = None
    if retry_telemetry_only:
        if len(receipts) != standard_limit + PLAN["max_manual_transport_retries_per_checkpoint"]:
            raise SystemExit("Telemetry-only recovery requires exactly three transport receipts and the saved fourth telemetry-gate attempt.")
        if (any(item.get("outcome") != "transport_or_parse_failed" for item in receipts[:standard_limit])
                or receipts[-1].get("outcome") != "gate_failed"
                or receipts[-1].get("attempt_id") != saved_records[name].get("attempt_id")
                or not receipts[-1].get("manual_transport_retry_override")
                or len(receipts) + 1 > total_limit):
            raise SystemExit("Prior attempt history does not match the registered telemetry-only recovery condition.")
        retry_archive = validate_telemetry_retry_archive(
            saved_records[name], ATTEMPTS / f"{name}-{receipts[-1]['attempt_id']}.json", receipts[-1])
        manual_telemetry_retry = True
        prior_ids = [item.get("attempt_id") for item in receipts]
    elif len(receipts) >= standard_limit:
        prior_ids = [item.get("attempt_id") for item in receipts]
        if (not retry_transport_failures
                or len(receipts) != standard_limit
                or len(receipts) + 1 > total_limit
                or name != PLAN["manual_transport_retry_unit"]
                or any(item.get("outcome") != "transport_or_parse_failed" for item in receipts)
                or (CHECKPOINTS / f"{name}.json").exists()):
            raise SystemExit(
                f"{name} reached the standard attempt limit. A fourth attempt is allowed only with "
                "--retry-transport-failures after exactly three transport failures and no checkpoint JSON."
            )
        manual_transport_retry = True
    elif retry_transport_failures:
        raise SystemExit("--retry-transport-failures is valid only after exactly three transport failures and no checkpoint JSON.")
    elif retry_telemetry_only:
        raise SystemExit("--retry-telemetry-only requires the registered saved telemetry-only failure.")
    else:
        prior_ids = []
    attempt_id = uuid.uuid4().hex
    attempt_path = ATTEMPTS / f"{name}-{attempt_id}.json"
    attempt = {"unit": name, "seed": seed, "width": width,
               "source_revision": revision, "source_sha256": source_digests,
               "attempt_id": attempt_id, "started_utc": utc_now(), "outcome": "started",
               "manual_transport_retry_override": manual_transport_retry,
               "checkpoint_absent_at_manual_retry": manual_transport_retry,
               "previous_transport_attempt_ids": prior_ids if manual_transport_retry else [],
               "manual_telemetry_retry_override": manual_telemetry_retry,
               "previous_telemetry_retry_attempt_ids": prior_ids if manual_telemetry_retry else [],
               "preserved_checkpoint_attempt_id": receipts[-1].get("attempt_id") if manual_telemetry_retry else None,
               "preserved_checkpoint_sha256": retry_archive[2] if retry_archive else None,
               "recovery_archive_manifest_path": retry_archive[0] if retry_archive else None,
               "recovery_archive_manifest_sha256": retry_archive[1] if retry_archive else None,
               "checkpoint_present_at_telemetry_retry": manual_telemetry_retry,
               "invocation_flag": "--retry-transport-failures" if manual_transport_retry else
                                  ("--retry-telemetry-only" if manual_telemetry_retry else None)}
    atomic_json(attempt_path, attempt)
    started = time.monotonic()
    if is_paused():
        attempt.update(outcome="paused_before_submit", finished_utc=utc_now(),
                       elapsed_seconds=time.monotonic() - started)
        atomic_json(attempt_path, attempt)
        raise SystemExit("Pause marker appeared before H100 submission.")
    command = [str(RUNNER), "bench", str(seed), str(width)]
    record = None
    try:
        output = subprocess.check_output(
            command, cwd=ROOT, env={**os.environ, "BENCH_REVISION": revision},
            pass_fds=(lock.fileno(),), timeout=PLAN["remote_timeout_seconds"] + 60)
        if re.search(rb"/Users/|/root/|/home/|PRIVATE KEY|(?:GPU|MIG)-[0-9a-fA-F]{8}-", output):
            raise ValueError("Remote output contains an unexpected private path or identifier.")
        record = json.loads(output)
        expected_sources = {name: source_digests[name] for name in COMPUTE_SOURCES}
        if (record.get("source_revision") != revision
                or record.get("source_sha256") != expected_sources
                or (record.get("seed"), record.get("rows"), record.get("width"))
                   != (seed, PLAN["rows"], width)):
            raise ValueError("Remote source pin or checkpoint identity mismatch.")
        record["execution_source_sha256"] = source_digests
        record["research_assistant_model"] = pin["research_assistant_model"]
        record["attempt_id"] = attempt_id
        checkpoint_path = CHECKPOINTS / f"{name}.json"
        atomic_json(checkpoint_path, record)
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        success = record.get("status") == "complete"
        attempt.update(outcome="checkpoint_saved" if success else "gate_failed",
                       finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                       checkpoint_status=record.get("status"), checkpoint_path=str(checkpoint_path.relative_to(ROOT)),
                       checkpoint_sha256=checkpoint_sha256, failure=record.get("failure"))
        atomic_json(attempt_path, attempt)
        print(f"Saved {name}: status={record.get('status')}, "
              f"latency intervals={len(record.get('latency_intervals', []))}, "
              f"sustained phases={len(record.get('sustained_phases', []))}.", flush=True)
        if not success:
            raise SystemExit(f"{name} gate or execution failed; later checkpoints remain stopped.")
    except BaseException as error:
        if attempt_path.exists():
            latest = json.loads(attempt_path.read_text())
            if latest.get("outcome") == "started":
                latest.update(outcome="transport_or_parse_failed", error_type=type(error).__name__,
                              finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
                if isinstance(error, subprocess.CalledProcessError):
                    latest["exit_code"] = error.returncode
                atomic_json(attempt_path, latest)
        raise


if __name__ == "__main__":
    STATE.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--pause", action="store_true")
    actions.add_argument("--resume", action="store_true")
    actions.add_argument("--retry-transport-failures", action="store_true",
                        help="Allow one fourth attempt only after exactly three transport-failed attempts and no checkpoint JSON.")
    actions.add_argument("--retry-telemetry-only", action="store_true",
                         help="Allow one fifth attempt only for the archived s50-w4103 telemetry-gap-only checkpoint.")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.pause:
        PAUSE_MARKERS[0].touch()
        print("Paused before the next H100 checkpoint.")
    elif args.resume:
        PAUSE_MARKERS[0].unlink(missing_ok=True)
        print("Cleared this experiment's marker; the global vLLM marker still applies.")
    else:
        with lab.worker_lock(STATE) as lock:
            run_one(lock, retry_transport_failures=args.retry_transport_failures,
                    retry_telemetry_only=args.retry_telemetry_only)
