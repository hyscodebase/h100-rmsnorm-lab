"""Run exactly one frozen width-dispatch H100 checkpoint."""
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

STATE = ROOT / ".lab/width-dispatch"
RAW = ROOT / "results/width-dispatch/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
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


def run_one(lock):
    if is_paused():
        raise SystemExit("Paused at checkpoint boundary; no H100 work submitted.")
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
    if failed_checkpoints:
        raise SystemExit("A saved checkpoint failed; inspect its evidence before continuing: "
                         + ", ".join(failed_checkpoints))
    else:
        unit = next(((seed, width) for seed, width in work_units()
                     if f"s{seed}-w{width}" not in saved_records), None)
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
    if len(receipts) >= standard_limit:
        raise SystemExit(f"{name} reached the three-attempt limit; inspect the saved receipts before manual recovery.")
    attempt_id = uuid.uuid4().hex
    attempt_path = ATTEMPTS / f"{name}-{attempt_id}.json"
    attempt = {"unit": name, "seed": seed, "width": width,
               "source_revision": revision, "source_sha256": source_digests,
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
