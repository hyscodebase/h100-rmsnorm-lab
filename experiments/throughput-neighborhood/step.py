"""Run or recover exactly one frozen H100 throughput-neighborhood checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lab

STATE = ROOT / ".lab/throughput-neighborhood"
RAW = ROOT / "results/throughput-neighborhood/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
TELEMETRY_DIR = RAW / "telemetry"
TELEMETRY_TOOL = ROOT.parent / "h100-rmsnorm-lab/.lab/telemetry_tool.py"
USAGE_DIR = ROOT.parent / "h100-rmsnorm-lab/.lab/usage"
RUNNER = ROOT / "experiments/throughput-neighborhood/ssh_runner.sh"
PAUSE_MARKERS = (STATE / "paused", ROOT / ".lab/vllm/paused")
COMPUTE_SOURCES = ("kernels.py", "experiments/throughput-neighborhood/run.py",
                   "experiments/throughput-neighborhood/PLAN.md", "plans/throughput-neighborhood.json")
EXECUTION_SOURCES = (*COMPUTE_SOURCES, "experiments/throughput-neighborhood/step.py",
                     "experiments/throughput-neighborhood/ssh_runner.sh",
                     "experiments/throughput-neighborhood/summarize.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with tmp.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in EXECUTION_SOURCES}


def freeze_source():
    pin_path = RAW / "source-pin.json"
    hashes = source_hashes()
    if not TELEMETRY_TOOL.is_file():
        raise SystemExit(f"Required GPU telemetry wrapper is missing: {TELEMETRY_TOOL}")
    telemetry_hash = hashlib.sha256(TELEMETRY_TOOL.read_bytes()).hexdigest()
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
        if pin["source_sha256"] != hashes or pin["telemetry_tool_sha256"] != telemetry_hash:
            raise ValueError("The working source or telemetry wrapper no longer matches its frozen pin.")
        revision = pin["source_freeze_revision"]
    else:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("A full Git commit is required to freeze the source.")
        for name in EXECUTION_SOURCES:
            subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", revision, "--", name], check=True)
            subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{revision}:{name}"], check=True)
        for name, digest in hashes.items():
            blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{name}"])
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError(f"Source hash differs from commit {revision}: {name}")
        pin = {"source_freeze_revision": revision, "source_sha256": hashes,
               "telemetry_tool_sha256": telemetry_hash,
               "research_assistant_model": "GPT-6 Luna (delegated runtime)",
               "frozen_utc": utc_now()}
        atomic_json(pin_path, pin)
    return revision, hashes, telemetry_hash


def publish_telemetry(start_epoch, end_epoch, attempt_tag):
    published = []
    if not USAGE_DIR.is_dir():
        return published
    paths = [p for p in USAGE_DIR.glob("*.jsonl")
             if start_epoch <= p.stat().st_mtime <= end_epoch]
    for index, source in enumerate(sorted(paths, key=lambda p: p.stat().st_mtime)):
        target = TELEMETRY_DIR / f"{attempt_tag}-{index:02d}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        published.append({"path": str(target.relative_to(ROOT)),
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                          "samples": sum(1 for _ in target.open())})
    return published


def work_units():
    plan = json.loads((ROOT / "plans/throughput-neighborhood.json").read_text())
    return [(seed, width) for seed in plan["seeds"] for width in plan["widths"]]


def paused():
    return any(path.exists() for path in PAUSE_MARKERS)


def step(lock):
    if paused():
        raise SystemExit("Paused at a checkpoint boundary; no H100 work submitted.")
    STATE.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    revision, hashes, pin_telemetry_hash = freeze_source()
    unit = next(((seed, width) for seed, width in work_units()
                 if not (CHECKPOINTS / f"s{seed}-w{width}.json").exists()), None)
    if unit is None:
        print("All 12 registered width/seed checkpoints are saved.")
        return
    seed, width = unit
    name = f"s{seed}-w{width}"
    earlier = list(ATTEMPTS.glob(f"{name}-*.json"))
    if len(earlier) >= 3:
        raise SystemExit(f"{name} has three attempt receipts; inspect them before retrying.")
    attempt_id = uuid.uuid4().hex
    attempt_path = ATTEMPTS / f"{name}-{attempt_id}.json"
    attempt = {"unit": name, "seed": seed, "width": width, "source_freeze_revision": revision,
               "attempt_id": attempt_id, "source_sha256": hashes,
               "started_utc": utc_now(), "outcome": "started"}
    atomic_json(attempt_path, attempt)
    started = time.monotonic()
    started_epoch = time.time()
    telemetry_published = False
    if paused():
        attempt.update(outcome="paused_before_submit", finished_utc=utc_now(),
                       elapsed_s=time.monotonic() - started)
        atomic_json(attempt_path, attempt)
        raise SystemExit("Pause marker appeared before remote submit; attempt was journaled.")
    print(f"Running {name} on the registered H100...", flush=True)
    try:
        command = [sys.executable, str(TELEMETRY_TOOL), "wrap", "--",
                   str(RUNNER), "bench", str(seed), str(width)]
        raw = subprocess.check_output(command, cwd=ROOT,
                                      env={**os.environ, "BENCH_REVISION": revision},
                                      pass_fds=(lock.fileno(),))
        if re.search(rb"/Users/|/root/|/home/|(?:GPU|MIG)-[0-9a-fA-F]{8}-|PRIVATE KEY", raw):
            raise ValueError("Unexpected private data in remote result; inspect before publishing.")
        record = json.loads(raw)
        expected = (seed, 32768, width, revision)
        actual = (record.get("seed"), record.get("rows"), record.get("width"), record.get("code_revision"))
        if actual != expected or record.get("source_sha256") != {k: hashes[k] for k in COMPUTE_SOURCES}:
            raise ValueError("Remote checkpoint identity, frozen revision, or compute-source hashes differ.")
        record["remote_source_sha256"] = record["source_sha256"]
        record["source_freeze_revision"] = revision
        record["source_sha256"] = hashes
        record["research_assistant_model"] = "GPT-6 Luna (delegated runtime)"
        record["telemetry_tool_sha256"] = pin_telemetry_hash
        record["telemetry"] = publish_telemetry(started_epoch, time.time(), f"{name}-{attempt_id}")
        telemetry_published = True
        attempt["telemetry"] = record["telemetry"]
        atomic_json(CHECKPOINTS / f"{name}.json", record)
        attempt.update(outcome="checkpoint_saved",
                       telemetry_samples=sum(item["samples"] for item in record["telemetry"]))
        print(f"Saved {name}: strict gate {record['correctness']['strict_gate']['passed']}, "
              f"{len(record['timing_intervals'])} intervals, "
              f"{sum(item['samples'] for item in record['telemetry'])} H100 samples.", flush=True)
    except BaseException as error:
        attempt.update(outcome="failed", error_type=type(error).__name__)
        if isinstance(error, subprocess.CalledProcessError):
            attempt["exit_code"] = error.returncode
        raise
    finally:
        if not telemetry_published:
            attempt["telemetry"] = publish_telemetry(started_epoch, time.time(), f"{name}-{attempt_id}")
        attempt.update(finished_utc=utc_now(), elapsed_s=time.monotonic() - started)
        atomic_json(attempt_path, attempt)


def show_status():
    completed = [p.stem for p in CHECKPOINTS.glob("s*-w*.json")] if CHECKPOINTS.exists() else []
    units = work_units()
    done = {name for name in completed}
    next_unit = next((f"s{seed}-w{width}" for seed, width in units
                      if f"s{seed}-w{width}" not in done), None)
    paused_markers = [str(path.relative_to(ROOT)) for path in PAUSE_MARKERS if path.exists()]
    print(f"Saved {len(done)}/{len(units)} checkpoints.")
    print(f"Next: {next_unit or 'none'}")
    print(f"Paused: {', '.join(paused_markers) if paused_markers else 'no'}")


if __name__ == "__main__":
    STATE.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true", help="Show saved work without contacting the GPU.")
    actions.add_argument("--pause", action="store_true", help="Pause before the next H100 checkpoint.")
    actions.add_argument("--resume", action="store_true", help="Clear this experiment's pause marker.")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.pause:
        PAUSE_MARKERS[0].touch()
        print("Paused before the next H100 checkpoint.")
    elif args.resume:
        PAUSE_MARKERS[0].unlink(missing_ok=True)
        print("Cleared this experiment's pause marker. A global vLLM pause marker still applies.")
    else:
        with lab.worker_lock(STATE) as lock:
            step(lock)
