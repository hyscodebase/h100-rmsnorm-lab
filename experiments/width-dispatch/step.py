"""Run exactly one frozen width-dispatch H100 checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab

STATE = ROOT / ".lab/width-dispatch"
RAW = ROOT / "results/width-dispatch/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
RUN_ID_PATH = STATE / "run-id.json"
RUNNER = ROOT / "experiments/width-dispatch/ssh_runner.sh"
PLAN = json.loads((ROOT / "plans/width-dispatch.json").read_text())
COMPUTE_SOURCES = (
    "kernels.py", "experiments/width-dispatch/run.py",
    "experiments/width-dispatch/telemetry.py",
    "experiments/width-dispatch/PLAN.md", "plans/width-dispatch.json",
)
EXECUTION_SOURCES = (*COMPUTE_SOURCES,
    "lab.py",
    "experiments/width-dispatch/step.py",
    "experiments/width-dispatch/ssh_runner.sh",
    "experiments/width-dispatch/summarize.py")
PAUSE_MARKERS = (STATE / "paused", ROOT / ".lab/vllm/paused")
PRIVATE_PATH = re.compile(r"/(?:Users|home|root)/[^\s\"'<>]+")
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b")
PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    lab.save_json(path, value)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in EXECUTION_SOURCES}


def current_run_id(revision, source_digests, create=False):
    """Keep remote cache reuse local to this checkout's frozen source campaign."""
    binding = {"source_revision": revision, "source_sha256": source_digests}
    active_evidence = any(path.is_file() for directory in (ATTEMPTS, CHECKPOINTS)
                          for path in directory.glob("*.json"))
    if RUN_ID_PATH.is_file():
        try:
            saved = json.loads(RUN_ID_PATH.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit("Run identity state is unreadable; stop before H100 submission.") from error
        run_id = saved.get("run_id")
        if (saved.get("source_revision") == revision
                and saved.get("source_sha256") == source_digests
                and isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id)):
            return run_id
        if active_evidence:
            raise SystemExit("Run identity is bound to a different source pin while active evidence exists.")
    elif active_evidence:
        raise SystemExit("Active evidence lacks its checkout run identity; restore the run-id file before resuming.")
    if not create:
        return None
    run_id = uuid.uuid4().hex
    atomic_json(RUN_ID_PATH, {**binding, "run_id": run_id, "created_utc": utc_now()})
    return run_id


def validate_runner_prerequisites():
    if not RUNNER.is_file() or not os.access(RUNNER, os.X_OK):
        raise SystemExit("Local H100 runner is missing or not executable; no attempt was submitted.")
    if any(shutil.which(command) is None for command in ("ssh", "tar", "bash", "python3")):
        raise SystemExit("A local SSH, tar, Bash or Python executable is unavailable; no attempt was submitted.")
    for key in ("rmsnorm.host", "rmsnorm.setup"):
        result = subprocess.run(["git", "-C", str(ROOT), "config", "--local", "--get", key],
                                text=True, capture_output=True, check=False)
        value = result.stdout.strip()
        if result.returncode != 0 or not value or "\n" in value:
            raise SystemExit(f"Local Git setting {key} is missing or invalid; no attempt was submitted.")


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


def redact_remote_value(value):
    if isinstance(value, str):
        value = PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", value)
        value = PRIVATE_PATH.sub("[REDACTED PATH]", value)
        return GPU_UUID.sub("[REDACTED GPU ID]", value)
    if isinstance(value, list):
        return [redact_remote_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_remote_value(item) for key, item in value.items()}
    return value


def read_saved_checkpoints():
    saved = {}
    for path in CHECKPOINTS.glob("*.json") if CHECKPOINTS.exists() else []:
        try:
            saved[path.stem] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot inspect saved checkpoint {path.name}; stop before submission.") from error
    return saved


def reconcile_dangling_attempts(saved_records):
    """Close interrupted local attempts before any retry or checkpoint skip."""
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(ATTEMPTS.glob("*.json")):
        try:
            attempt = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot inspect prior attempt receipt {path.name}; stop before submission.") from error
        if attempt.get("outcome") != "started":
            continue
        unit = attempt.get("unit")
        expected_units = {f"s{seed}-w{width}" for seed, width in work_units()}
        if unit not in expected_units or path.stem != f"{unit}-{attempt.get('attempt_id')}":
            raise SystemExit(f"Cannot safely reconcile malformed started receipt: {path.name}.")
        checkpoint_path = CHECKPOINTS / f"{unit}.json"
        if checkpoint_path.is_file():
            try:
                record = json.loads(checkpoint_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise SystemExit(f"Cannot reconcile checkpoint {checkpoint_path.name}.") from error
            if record.get("attempt_id") != attempt.get("attempt_id"):
                raise SystemExit(f"Started attempt and saved checkpoint identity conflict: {unit}.")
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            status = record.get("status")
            attempt.update(
                outcome="checkpoint_saved" if status == "complete" else "gate_failed",
                finished_utc=utc_now(),
                remote_checkpoint_finished_utc=record.get("finished_utc"),
                elapsed_seconds=None,
                local_elapsed_seconds_observed=False,
                remote_benchmark_wall_seconds=record.get("wall_seconds"),
                producing_attempt_id=record.get("producing_attempt_id", record.get("attempt_id")),
                cache_retrieved=(record.get("producing_attempt_id", record.get("attempt_id"))
                                 != attempt.get("attempt_id")),
                checkpoint_status=status,
                checkpoint_path=str(checkpoint_path.relative_to(ROOT)),
                checkpoint_sha256=digest,
                failure=record.get("failure"),
                reconciled_after_interruption=True)
            saved_records[unit] = record
        else:
            attempt.update(outcome="transport_or_parse_failed", finished_utc=utc_now(),
                           elapsed_seconds=None, local_elapsed_seconds_observed=False,
                           error_type="InterruptedError",
                           error_message="Reconciled an unfinished local attempt before retry.",
                           reconciled_after_interruption=True)
        atomic_json(path, attempt)


def validate_saved_state(revision, source_digests, saved_records, run_id):
    expected_units = {f"s{seed}-w{width}" for seed, width in work_units()}
    unexpected = sorted(set(saved_records) - expected_units)
    if unexpected:
        raise SystemExit("Unexpected checkpoint file(s) block this source pin: " + ", ".join(unexpected))
    expected_compute = {name: source_digests[name] for name in COMPUTE_SOURCES}
    receipts_by_unit = {}
    receipt_ids = set()
    allowed_outcomes = {"checkpoint_saved", "gate_failed", "transport_or_parse_failed",
                        "local_launch_failed", "paused_before_submit"}
    for path in sorted(ATTEMPTS.glob("*.json")) if ATTEMPTS.exists() else []:
        try:
            item = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot inspect attempt receipt {path.name}; stop before submission.") from error
        attempt_id = item.get("attempt_id")
        unit = item.get("unit")
        if (not attempt_id or attempt_id in receipt_ids
                or path.stem != f"{unit}-{attempt_id}"):
            raise SystemExit(f"Attempt filename or ID is invalid: {path.name}.")
        receipt_ids.add(attempt_id)
        if (item.get("outcome") not in allowed_outcomes
                or item.get("source_revision") != revision
                or item.get("source_sha256") != source_digests
                or item.get("run_id") != run_id
                or item.get("outcome") == "started"):
            raise SystemExit(f"Attempt receipt does not match the active source pin: {path.name}.")
        if item.get("unit") != f"s{item.get('seed')}-w{item.get('width')}":
            raise SystemExit(f"Attempt checkpoint identity mismatch: {path.name}.")
        if item.get("seed") not in PLAN["seeds"] or item.get("width") not in PLAN["widths"]:
            raise SystemExit(f"Attempt refers to an unregistered checkpoint: {path.name}.")
        receipts_by_unit.setdefault(unit, []).append(item)

    for unit, attempts in receipts_by_unit.items():
        ordered = sorted(attempts, key=lambda item: item.get("started_utc", ""))
        submitted = [item for item in ordered
                     if item.get("outcome") not in {"paused_before_submit", "local_launch_failed"}]
        if len(submitted) > PLAN["max_attempts_per_checkpoint"]:
            raise SystemExit(f"Attempt limit exceeded for {unit}.")
        outcomes = [item.get("outcome") for item in ordered]
        if "gate_failed" in outcomes and outcomes[-1] != "gate_failed":
            raise SystemExit(f"An attempt followed a blocking gate failure for {unit}.")
        if "checkpoint_saved" in outcomes and outcomes[-1] != "checkpoint_saved":
            raise SystemExit(f"An attempt followed a saved successful checkpoint for {unit}.")
        for item in ordered:
            if (item.get("outcome") in {"checkpoint_saved", "gate_failed"}
                    and unit not in saved_records):
                raise SystemExit(f"Saved-checkpoint receipt has no matching checkpoint: {unit}.")

    for unit, record in saved_records.items():
        seed, width = record.get("seed"), record.get("width")
        if (unit != f"s{seed}-w{width}" or seed not in PLAN["seeds"]
                or width not in PLAN["widths"] or record.get("rows") != PLAN["rows"]):
            raise SystemExit(f"Checkpoint identity differs from the registered plan: {unit}.")
        if (record.get("source_revision") != revision
                or record.get("source_sha256") != expected_compute
                or record.get("execution_source_sha256") != source_digests
                or record.get("run_id") != run_id):
            raise SystemExit(f"Saved checkpoint source pin mismatch: {unit}.")
        matches = [item for item in receipts_by_unit.get(unit, [])
                   if item.get("attempt_id") == record.get("attempt_id")]
        if len(matches) != 1:
            raise SystemExit(f"Saved checkpoint does not have one matching attempt receipt: {unit}.")
        receipt = matches[0]
        checkpoint_path = CHECKPOINTS / f"{unit}.json"
        checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if (receipt.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
                or receipt.get("checkpoint_sha256") != checkpoint_digest
                or receipt.get("checkpoint_status") != record.get("status")):
            raise SystemExit(f"Checkpoint and receipt status/path/hash do not match: {unit}.")
        producing_attempt_id = record.get("producing_attempt_id", record.get("attempt_id"))
        if producing_attempt_id != record.get("attempt_id"):
            producers = [item for item in receipts_by_unit.get(unit, [])
                         if item.get("attempt_id") == producing_attempt_id]
            if (len(producers) != 1
                    or producers[0].get("outcome") != "transport_or_parse_failed"
                    or producers[0].get("started_utc", "") > receipt.get("started_utc", "")):
                raise SystemExit(f"Cached checkpoint has no earlier unresolved producer receipt: {unit}.")
        if record.get("status") == "complete":
            if receipt.get("outcome") != "checkpoint_saved":
                raise SystemExit(f"Complete checkpoint lacks a successful receipt: {unit}.")
            successes = [item for item in receipts_by_unit[unit]
                         if item.get("outcome") == "checkpoint_saved"]
            if len(successes) != 1:
                raise SystemExit(f"Checkpoint has an ambiguous successful receipt: {unit}.")
        else:
            if record.get("status") not in {"gate_failed", "execution_failed"}:
                raise SystemExit(f"Checkpoint has an unknown status: {unit}.")
            if receipt.get("outcome") != "gate_failed":
                raise SystemExit(f"Failed checkpoint lacks a blocking failure receipt: {unit}.")


def refresh_partial_audit():
    command = [sys.executable, str(ROOT / "experiments/width-dispatch/summarize.py"), "--audit"]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Checkpoint is saved, but its audit could not run ({type(error).__name__}); stop before later work.") from None
    if result.returncode != 0:
        raise SystemExit("Checkpoint is saved, but its source/receipt/evidence audit failed; stop before later work.")
    try:
        audit = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("Checkpoint is saved, but the partial audit did not return valid JSON.") from error
    audit["generated_utc"] = utc_now()
    atomic_json(RAW / "partial-audit.json", audit)


def show_status():
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / ".worker.lock").touch(exist_ok=True)
    active = lab.worker_active(STATE)
    records = read_saved_checkpoints()
    audit_command = [sys.executable, str(ROOT / "experiments/width-dispatch/summarize.py"), "--audit"]
    audit, audit_error, in_flight_unit = None, None, None
    started_receipts = []
    if active:
        for path in sorted(ATTEMPTS.glob("*.json")) if ATTEMPTS.exists() else []:
            try:
                item = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if item.get("outcome") == "started":
                started_receipts.append(item)
        if len(started_receipts) == 1:
            try:
                pin = json.loads(SOURCE_PIN.read_text())
                current = hashes()
                receipt = started_receipts[0]
                cached = json.loads((RAW / "partial-audit.json").read_text())
                cached_units = {item.get("checkpoint") for item in cached.get("checkpoint_results", [])}
                if (pin.get("source_sha256") == current
                        and receipt.get("source_revision") == pin.get("source_revision")
                        and receipt.get("source_sha256") == pin.get("source_sha256")
                        and cached.get("source_revision") == pin.get("source_revision")
                        and cached_units == set(records)):
                    audit = cached
                    in_flight_unit = receipt.get("unit")
                else:
                    audit_error = "active receipt or cached partial audit does not match the current source pin"
            except (OSError, json.JSONDecodeError, KeyError) as error:
                audit_error = f"cannot validate in-flight checkpoint state ({type(error).__name__})"
        elif started_receipts:
            audit_error = "multiple unfinished attempt receipts exist while the worker lock is active"
    if audit is None:
        try:
            audit_process = subprocess.run(audit_command, cwd=ROOT, text=True,
                                           capture_output=True, check=False, timeout=120)
            audit = json.loads(audit_process.stdout) if audit_process.returncode == 0 else None
            if audit is None and audit_error is None:
                audit_error = (audit_process.stderr.strip().splitlines()[-1]
                               if audit_process.stderr.strip()
                               else "partial evidence audit failed")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            audit, audit_error = None, f"partial evidence audit unavailable ({type(error).__name__})"
    audited_records = (audit or {}).get("checkpoint_results", [])
    saved = {item.get("checkpoint"): item.get("status") for item in audited_records}
    units = work_units()
    completed = {unit for unit, status in saved.items() if status == "complete"}
    failed = sorted(unit for unit, status in saved.items() if status != "complete")
    blockers = []
    if audit is None:
        blockers.append("source/receipt/evidence audit failed")
    if failed:
        blockers.append("saved failed checkpoint(s): " + ", ".join(failed))
    if audit is not None:
        pin = json.loads(SOURCE_PIN.read_text()) if SOURCE_PIN.is_file() else {}
        scope = pin.get("h100_execution_scope")
        if scope not in {"pilot", "full_matrix"}:
            blockers.append("source pin lacks an explicit pilot/full-matrix execution scope")
        attempts_by_unit = {}
        for path in sorted(ATTEMPTS.glob("*.json")) if ATTEMPTS.exists() else []:
            try:
                item = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                blockers.append(f"unreadable attempt receipt: {path.name}")
                continue
            attempts_by_unit.setdefault(item.get("unit"), []).append(item)
        for unit, items in attempts_by_unit.items():
            submitted = [item for item in items
                         if item.get("outcome") not in {"paused_before_submit", "local_launch_failed"}]
            if unit not in saved and len(submitted) >= PLAN["standard_max_attempts_per_checkpoint"]:
                blockers.append(f"retry limit exhausted for {unit}")
        if scope == "pilot" and completed:
            blockers.append("pilot checkpoint allowance used; full-matrix review is required")
    next_unit = (None if blockers else next((f"s{seed}-w{width}" for seed, width in units
                                             if f"s{seed}-w{width}" not in completed), None))
    markers = [str(path.relative_to(ROOT)) for path in PAUSE_MARKERS if path.exists()]
    print(f"Accepted completed: {len(completed)}/{len(units)} checkpoints.")
    print(f"Raw checkpoint files saved: {len(records)}")
    print(f"Saved failed/incomplete: {len(failed)}")
    blocker_label = "; ".join(blockers) if blockers else "none"
    if audit_error and audit is None:
        blocker_label += f" ({audit_error})"
    print(f"Submission blocker(s): {blocker_label}")
    next_label = (f"in progress: {in_flight_unit}" if in_flight_unit else
                  f"blocked by {blocker_label}" if blockers else (next_unit or "none"))
    print(f"Next: {next_label}")
    worker_state = ("pausing" if active and is_paused() else "running" if active
                    else "paused" if is_paused() else "idle")
    print(f"Worker: {worker_state}")
    if in_flight_unit:
        print(f"Current attempt: {in_flight_unit}; saved progress comes from the last partial audit.")
    print(f"Pause markers: {', '.join(markers) if markers else 'none'}")
    phases = [phase for record in records.values() for phase in record.get("sustained_phases", [])]
    event_seconds = sum(float(phase.get("event_seconds", 0.0)) for phase in phases
                        if type(phase.get("event_seconds")) in (int, float))
    samples = [sample for phase in phases for sample in phase.get("telemetry_samples", [])]
    def values(field):
        return [sample[field] for sample in samples
                if type(sample.get(field)) in (int, float)]
    print(f"Saved sustained CUDA-event time: {event_seconds / 60:.2f} / 52.80 min")
    print(f"Saved telemetry rows: {len(samples)}; eligible phases: "
          f"{sum(phase.get('activity_telemetry_eligible') is True for phase in phases)}/{len(phases)}")
    if values("gpu_busy_percent"):
        print(f"GPU busy median/peak: {statistics.median(values('gpu_busy_percent')):.1f}% / "
              f"{max(values('gpu_busy_percent')):.1f}%")
    if values("memory_used_mib"):
        print(f"VRAM used peak: {max(values('memory_used_mib')):.0f} MiB")
    if values("power_w"):
        print(f"Power median/peak: {statistics.median(values('power_w')):.1f} / "
              f"{max(values('power_w')):.1f} W")
    if values("temperature_c"):
        print(f"GPU temperature peak: {max(values('temperature_c')):.1f} C")


def run_one(lock):
    if is_paused():
        raise SystemExit("Paused at checkpoint boundary; no H100 work submitted.")
    revision, source_digests, pin = verify_source_pin()
    if pin.get("h100_execution_authorized") is not True:
        raise SystemExit("Source pin is plan-only; H100 execution is not authorized.")
    execution_scope = pin.get("h100_execution_scope")
    if execution_scope not in {"pilot", "full_matrix"}:
        raise SystemExit("Source pin must explicitly authorize a pilot or the full checkpoint matrix.")
    run_id = current_run_id(revision, source_digests, create=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    saved_records = read_saved_checkpoints()
    reconcile_dangling_attempts(saved_records)
    validate_saved_state(revision, source_digests, saved_records, run_id)
    refresh_partial_audit()
    failed_checkpoints = sorted(unit for unit, item in saved_records.items()
                                if item.get("status") != "complete")
    if failed_checkpoints:
        raise SystemExit("A saved checkpoint failed; inspect its evidence before continuing: "
                         + ", ".join(failed_checkpoints))
    else:
        unit = next(((seed, width) for seed, width in work_units()
                     if f"s{seed}-w{width}" not in saved_records), None)
    if execution_scope == "pilot" and saved_records and unit is not None:
        raise SystemExit("Pilot checkpoint is already saved; review and explicitly authorize the full matrix before continuing.")
    if unit is None:
        print(f"All {len(work_units())} registered checkpoints are saved.")
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
    submitted_attempts = [item for item in receipts
                          if item.get("outcome") not in {"paused_before_submit", "local_launch_failed"}]
    if len(submitted_attempts) >= standard_limit:
        raise SystemExit(f"{name} reached the three-attempt limit; inspect the saved receipts before manual recovery.")
    validate_runner_prerequisites()
    attempt_id = uuid.uuid4().hex
    attempt_path = ATTEMPTS / f"{name}-{attempt_id}.json"
    attempt = {"unit": name, "seed": seed, "width": width,
               "source_revision": revision, "source_sha256": source_digests,
               "run_id": run_id,
               "attempt_id": attempt_id, "started_utc": utc_now(), "outcome": "started",
               "manual_recovery": False, "invocation_flag": None}
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
            command, cwd=ROOT, env={**os.environ, "BENCH_REVISION": revision,
                                    "BENCH_ATTEMPT_ID": attempt_id,
                                    "BENCH_RUN_ID": run_id,
                                    "BENCH_EXECUTION_SOURCE_SHA256": json.dumps(
                                        source_digests, sort_keys=True, separators=(",", ":"))},
            pass_fds=(lock.fileno(),), timeout=PLAN["remote_total_timeout_seconds"] + 10)
        record = redact_remote_value(json.loads(output))
        if not isinstance(record, dict):
            raise ValueError("Remote checkpoint output is not a JSON object.")
        expected_sources = {name: source_digests[name] for name in COMPUTE_SOURCES}
        if (record.get("source_revision") != revision
                or record.get("source_sha256") != expected_sources
                or record.get("execution_source_sha256") != source_digests
                or record.get("run_id") != run_id
                or (record.get("seed"), record.get("rows"), record.get("width"))
                   != (seed, PLAN["rows"], width)):
            raise ValueError("Remote source pin or checkpoint identity mismatch.")
        producing_attempt_id = record.get("attempt_id")
        prior_attempts = {item.get("attempt_id"): item for item in receipts}
        if (not isinstance(producing_attempt_id, str)
                or not re.fullmatch(r"[0-9a-f]{32}", producing_attempt_id)):
            raise ValueError("Remote checkpoint lacks a valid producing attempt ID.")
        if producing_attempt_id != attempt_id:
            producer = prior_attempts.get(producing_attempt_id)
            if (producer is None or producer.get("outcome") != "transport_or_parse_failed"
                    or producer.get("started_utc", "") > attempt["started_utc"]):
                raise ValueError("Cached result producer does not match an earlier unresolved transport attempt.")
        record["producing_attempt_id"] = producing_attempt_id
        record["cache_retrieved"] = producing_attempt_id != attempt_id
        record["attempt_id"] = attempt_id
        record["research_assistant_model"] = pin["research_assistant_model"]
        checkpoint_path = CHECKPOINTS / f"{name}.json"
        atomic_json(checkpoint_path, record)
        checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        success = record.get("status") == "complete"
        attempt.update(outcome="checkpoint_saved" if success else "gate_failed",
                       finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                       local_elapsed_seconds_observed=True,
                       remote_checkpoint_finished_utc=record.get("finished_utc"),
                       remote_benchmark_wall_seconds=record.get("wall_seconds"),
                       producing_attempt_id=producing_attempt_id,
                       cache_retrieved=record["cache_retrieved"],
                       checkpoint_status=record.get("status"), checkpoint_path=str(checkpoint_path.relative_to(ROOT)),
                       checkpoint_sha256=checkpoint_sha256, failure=record.get("failure"))
        atomic_json(attempt_path, attempt)
        refresh_partial_audit()
        print(f"Saved {name}: status={record.get('status')}, "
              f"latency intervals={len(record.get('latency_intervals', []))}, "
              f"sustained phases={len(record.get('sustained_phases', []))}.", flush=True)
        if not success:
            raise SystemExit(f"{name} gate or execution failed; later checkpoints remain stopped.")
    except BaseException as error:
        if attempt_path.exists():
            latest = json.loads(attempt_path.read_text())
            if latest.get("outcome") == "started":
                checkpoint_path = CHECKPOINTS / f"{name}.json"
                published = False
                if checkpoint_path.is_file():
                    try:
                        saved = json.loads(checkpoint_path.read_text())
                        published = saved.get("attempt_id") == attempt_id
                    except (OSError, json.JSONDecodeError):
                        published = False
                if published:
                    latest["checkpoint_published_before_interruption"] = True
                    atomic_json(attempt_path, latest)
                elif isinstance(error, OSError) and record is None:
                    latest.update(outcome="local_launch_failed", error_type=type(error).__name__,
                                  finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                                  local_elapsed_seconds_observed=True,
                                  error_message="The local SSH runner process could not be started.")
                    atomic_json(attempt_path, latest)
                else:
                    latest.update(outcome="transport_or_parse_failed", error_type=type(error).__name__,
                                  finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                                  local_elapsed_seconds_observed=True,
                                  error_message=(
                                      "Local SSH wrapper timed out; the remote outcome is unknown and may be recovered from its cache."
                                      if isinstance(error, subprocess.TimeoutExpired)
                                      else "SSH returned no verifiable checkpoint record; the remote outcome remains unresolved."))
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
            run_one(lock)
