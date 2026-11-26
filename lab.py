"""Run and resume a GPU experiment one correctness check or benchmark shape at a time."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import gemm_plan
import observer_plan

ROOT = Path(__file__).resolve().parent
SOURCES = ("kernels.py", "check.py", "benchmark.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def save_json(path, value):
    """Publish a complete file only after its contents reach disk."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w") as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def worker_lock(state):
    with (state / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another worker is active for this state directory.") from exc
        yield lock


def worker_active(state):
    with (state / ".worker.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def validate_plan(plan):
    require(isinstance(plan, dict), "The plan must be a JSON object.")
    for key in ("id", "hypothesis"):
        require(isinstance(plan.get(key), str) and plan[key].strip(), f"{key} must be nonempty text.")
    kind = plan.get("kind", "rmsnorm")
    require(kind in ("rmsnorm", "prefill_gemm", observer_plan.KIND), f"Unknown experiment kind: {kind}")
    if kind == observer_plan.KIND:
        observer_plan.validate_plan(plan)
        return
    for key in (("seeds",) if kind == "prefill_gemm" else ("seeds", "rows", "widths", "row_tiles")):
        values = plan.get(key)
        require(isinstance(values, list) and values and all(type(v) is int for v in values),
                f"{key} must contain integers.")
        require(len(set(values)) == len(values), f"{key} must contain distinct values.")
    require(min(plan["seeds"]) >= 0 and max(plan["seeds"]) < 2**64, "Seeds must fit unsigned 64 bits.")
    if kind == "prefill_gemm":
        gemm_plan.validate_plan(plan)
        return
    require(min(plan["rows"]) > 0 and min(plan["widths"]) > 0 and max(plan["widths"]) <= 16384,
            "rows must be positive and widths must be in [1, 16384].")
    require(1 in plan["row_tiles"] and set(plan["row_tiles"]) <= {1, 2, 4}, "row_tiles must include 1 and use 1, 2 or 4.")
    for key, minimum in (("ops", 1), ("rounds", 2)):
        require(type(plan.get(key)) is int and plan[key] >= minimum, f"{key} must be at least {minimum}.")
    require(type(plan.get("compiled")) is bool, "compiled must be true or false.")


def manifest(plan):
    sources = (gemm_plan.SOURCE,) if plan.get("kind") == "prefill_gemm" else SOURCES
    if plan.get("kind") == observer_plan.KIND:
        sources = observer_plan.SOURCES
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    payload = json.dumps([plan, hashes], sort_keys=True, allow_nan=False).encode()
    return {"version": 1, "experiment_hash": hashlib.sha256(payload).hexdigest(), "sources": hashes}


def tasks(plan):
    if plan.get("kind") == observer_plan.KIND:
        yield from observer_plan.tasks(plan)
        return
    if plan.get("kind") == "prefill_gemm":
        yield from gemm_plan.tasks(plan)
        return
    yield "check", None
    for seed, rows, width in product(plan["seeds"], plan["rows"], plan["widths"]):
        yield f"s{seed}-r{rows}-w{width}", (seed, rows, width)


def load_state(state):
    require((state / "manifest.json").is_file(), "This state directory has not been initialized.")
    plan = read_json(state / "plan.json")
    validate_plan(plan)
    actual = manifest(plan)
    require(actual == read_json(state / "manifest.json"),
            "The experiment source or saved plan changed. Restore it or initialize a new state directory.")
    return plan, actual["experiment_hash"]


def initialize(state, plan_file):
    plan = read_json(plan_file)
    validate_plan(plan)
    state.mkdir(parents=True, exist_ok=True)
    with worker_lock(state):
        if (state / "manifest.json").exists():
            saved, _ = load_state(state)
            require(saved == plan, "This directory already contains a different plan. Use a new directory.")
        else:
            saved_plan = state / "plan.json"
            require(not saved_plan.exists() or read_json(saved_plan) == plan, "Existing plan differs. Use a new directory.")
            folder = state / "tasks"
            folder.mkdir(exist_ok=True)
            require(not any(folder.glob("*.json")), "Existing task evidence has no manifest. Use a new directory.")
            save_json(saved_plan, plan)
            save_json(state / "manifest.json", manifest(plan))
    print(f"Initialized {plan['id']} in {state}.")


def validate_result(result, shape, plan):
    require(isinstance(result, dict), "Expected a JSON object from the runner.")
    if plan.get("kind") == observer_plan.KIND:
        return observer_plan.validate_result(result, shape)
    if shape is None:
        checks = result.get("checks")
        require(result.get("passed") is True and result.get("reference") == "fp32"
                and result.get("dtype") == "bfloat16" and isinstance(checks, list) and checks,
                "Missing or unsuccessful correctness evidence.")
        cases = {(c["rows"], c["width"], c["input"]) for c in checks}
        expected = {(*case, tile, warps) for case, tile, warps in product(cases, (1, 2, 4), (4, 8))}
        actual = {(c["rows"], c["width"], c["input"], c["rows_per_program"], c["num_warps"]) for c in checks}
        require(len(checks) == len(expected) and actual == expected, "The correctness suite is incomplete.")
        for check in checks:
            require(check["passed"] is True and math.isfinite(check["max_abs_error"]) and check["max_abs_error"] >= 0,
                    "A numerical check failed or returned a nonfinite error.")
        return None

    env = result["environment"]
    gemm = plan.get("kind") == "prefill_gemm"
    required_versions = ("driver", "python") if gemm else ("triton",)
    for key in ("gpu", "cuda_runtime", "torch", *required_versions):
        require(isinstance(env.get(key), str) and env[key].strip(), f"Missing GPU environment field {key}.")
    cap = env.get("compute_capability")
    require(isinstance(cap, list) and len(cap) == 2 and all(type(v) is int and v >= 0 for v in cap)
            and cap[0] >= 8, "Expected a CUDA compute capability supporting native BF16.")
    environment = {key: env.get(key) for key in ("gpu", "compute_capability", "cuda_runtime", "driver", "torch", "triton", "python")}
    if gemm:
        for key in ("total_memory_bytes", "sm_count"):
            require(type(env.get(key)) is int and env[key] > 0, f"Missing GPU hardware field {key}.")
            environment[key] = env[key]
        gemm_plan.validate_result(result, shape, plan)
        return environment
    seed, rows, width = shape
    expected = {"seed": seed, "rounds": plan["rounds"], "ops_per_graph": plan["ops"],
                "row_tiles": plan["row_tiles"], "dtype": "bfloat16"}
    require(all(result["method"].get(k) == v for k, v in expected.items()), "Benchmark method differs from plan.")
    backends = {"torch_eager"}
    if plan["compiled"]:
        backends.add("torch_compile")
    for tile, warps in product(plan["row_tiles"], (4, 8)):
        backends.add(f"triton_{warps}w" + (f"_{tile}r" if tile != 1 else ""))
    entries = result["results"]
    require(len(entries) == len(backends) and {r["backend"] for r in entries} == backends,
            "Benchmark backends differ from plan.")
    for entry in entries:
        require((entry["rows"], entry["width"]) == (rows, width), "Benchmark shape differs from plan.")
        samples = entry["samples_us"]
        require(len(samples) == plan["rounds"] and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in samples),
                "Incomplete, nonfinite or nonpositive benchmark samples.")
        q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
        for key, expected_value in (("q1_us", q1), ("median_us", median), ("q3_us", q3)):
            require(math.isclose(entry[key], expected_value, rel_tol=1e-10), f"{key} differs from raw samples.")
    return environment


def progress(state, plan, experiment_hash):
    pending = []
    environment = None
    for task_id, shape in tasks(plan):
        path = state / "tasks" / f"{task_id}.json"
        if not path.exists():
            pending.append((task_id, shape))
            continue
        artifact = read_json(path)
        require(artifact.get("experiment_hash") == experiment_hash and artifact.get("task_id") == task_id,
                f"Checkpoint identity mismatch in {path}.")
        backend = validate_result(artifact["result"], shape, plan)
        if backend:
            require(environment in (None, backend), "Checkpoints have inconsistent GPU environments.")
            environment = backend
    return pending, environment


def command(runner, shape, plan):
    if plan.get("kind") == observer_plan.KIND:
        cmd = [str(runner), "observer"] if runner else [sys.executable, str(ROOT / observer_plan.SOURCE)]
        return cmd + observer_plan.arguments(shape)
    if plan.get("kind") == "prefill_gemm":
        cmd = [str(runner), "gemm"] if runner else [sys.executable, str(ROOT / gemm_plan.SOURCE)]
        return cmd + gemm_plan.arguments(shape, plan)
    mode = "bench" if shape else "check"
    cmd = [str(runner), mode] if runner else [sys.executable, str(ROOT / ("benchmark.py" if shape else "check.py"))]
    if shape:
        seed, rows, width = shape
        cmd += ["--rows", str(rows), "--widths", str(width), "--seed", str(seed), "--rounds", str(plan["rounds"]),
                "--ops", str(plan["ops"]), "--row-tiles", *map(str, plan["row_tiles"])]
        if plan["compiled"]:
            cmd.append("--compiled")
    return cmd


def resolve_runner(explicit):
    if explicit is not None:
        return explicit.expanduser().resolve()
    if shutil.which("git"):
        configured = subprocess.run(
            ["git", "config", "--local", "--get", "rmsnorm.runner"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        value = configured.stdout.strip()
        if configured.returncode == 0 and value:
            return (ROOT / Path(value).expanduser()).resolve()
    return None


def local_gemm_revision(root, expected_hash, source=gemm_plan.SOURCE, ref="HEAD"):
    if not shutil.which("git"):
        return "unversioned"
    git = ["git", "--no-replace-objects", "-C", str(root)]
    head = subprocess.run([*git, "rev-parse", "--verify", f"{ref}^{{commit}}"], capture_output=True, text=True)
    if head.returncode:
        return "unversioned"
    revision = head.stdout.strip()
    blob = subprocess.run([*git, "cat-file", "blob", f"{revision}:{source}"], capture_output=True)
    if blob.returncode:
        return "unversioned"
    if hashlib.sha256(blob.stdout).hexdigest() != expected_hash:
        return f"{revision}-modified"
    return revision


@contextmanager
def record_attempt(state, task_id, experiment_hash, shape, plan):
    directory = state / "attempts"
    directory.mkdir(exist_ok=True)
    attempt_id = uuid.uuid4().hex
    path = directory / f"{task_id}-{attempt_id}.json"
    _, program, *arguments = command(None, shape, plan)
    record = {
        "attempt_id": attempt_id,
        "task_id": task_id,
        "experiment_hash": experiment_hash,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce_argv": ["python3", str(Path(program).relative_to(ROOT)), *arguments],
        "outcome": "started",
    }
    save_json(path, record)
    start = time.monotonic()
    try:
        yield
    except BaseException as exc:
        record["outcome"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error_type"] = type(exc).__name__
        if isinstance(exc, subprocess.CalledProcessError):
            record["exit_code"] = exc.returncode
        raise
    else:
        record["outcome"] = "checkpoint_saved"
    finally:
        record["controller_elapsed_s"] = time.monotonic() - start
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(path, record)


def execute(state, runner, max_steps):
    with worker_lock(state) as lock:
        plan, experiment_hash = load_state(state)
        pending, environment = progress(state, plan, experiment_hash)
        for task_id, shape in pending[:max_steps]:
            if (state / "paused").exists():
                print("Paused. Resume to start the next task.")
                return
            observer = plan.get("kind") == observer_plan.KIND
            if observer:
                budget = observer_plan.launch_limit(state, task_id, plan)
                sources = read_json(state / "manifest.json")["sources"]
                reference_file = state / "observer-revision.json"
                reference = read_json(reference_file) if reference_file.exists() else "HEAD"
                revisions = {local_gemm_revision(ROOT, digest, source, reference) for source, digest in sources.items()}
                require(len(revisions) == 1, "Observer sources must be frozen together before launch.")
                frozen = revisions.pop()
                require(frozen != "unversioned" and not frozen.endswith("-modified"), "Commit observer sources before launch.")
                if not reference_file.exists():
                    save_json(reference_file, frozen)
            print(f"Running {task_id} ...", flush=True)
            with record_attempt(state, task_id, experiment_hash, shape, plan):
                env = None
                if runner is None and plan.get("kind") == "prefill_gemm":
                    expected_hash = read_json(state / "manifest.json")["sources"][gemm_plan.SOURCE]
                    revision = local_gemm_revision(ROOT, expected_hash)
                    env = {**os.environ, "BENCH_REVISION": revision}
                    if revision == "unversioned" or revision.endswith("-modified"):
                        print("Local benchmark source has no matching HEAD revision. "
                              "This checkpoint cannot be exported as a verified Git run.", file=sys.stderr)
                # Keep a surviving local runner locked even if its coordinator is killed.
                cmd = command(runner, shape, plan)
                if observer:
                    env = {**os.environ, "BENCH_REVISION": frozen}
                    result = observer_plan.bounded_run(cmd, ROOT, env, lock.fileno(), budget)
                    require(result.get("code_revision") == frozen, "Remote observer ran a different source revision.")
                else:
                    output = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True, check=True,
                                            pass_fds=(lock.fileno(),), env=env)
                    result = json.loads(output.stdout)
                backend = validate_result(result, shape, plan)
                if backend:
                    require(environment in (None, backend), "GPU environment changed. Initialize a new state directory.")
                    environment = backend
                load_state(state)  # Detect source edits during a long subprocess before accepting its result.
                save_json(state / "tasks" / f"{task_id}.json", {
                    "task_id": task_id, "experiment_hash": experiment_hash, "result": result,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                })
            print(f"Saved {task_id}.", flush=True)
        if max_steps is None or len(pending) <= max_steps:
            print("Experiment complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("init", "step", "run", "pause", "resume", "status"):
        sub = subs.add_parser(name)
        sub.add_argument("--state-dir", type=Path, default=Path(".lab/current"),
                         help="Checkpoint directory (default: .lab/current).")
        if name == "init":
            sub.add_argument("--plan", type=Path, required=True)
        if name in ("step", "run"):
            sub.add_argument("--runner", type=Path,
                             help="Executable accepting check, bench, gemm or observer. Defaults to local Git setting rmsnorm.runner, then local Python.")
        if name == "run":
            sub.add_argument("--max-steps", type=int, help="Stop after this many newly completed tasks.")
    args = parser.parse_args()
    state = args.state_dir.resolve()
    if args.action == "init":
        initialize(state, args.plan)
        return
    require((state / "manifest.json").is_file(), "This state directory has not been initialized.")
    if args.action == "pause":
        save_json(state / "paused", True)
        print("Pause requested. Any current task will finish first.")
        return
    if args.action in ("step", "run"):
        limit = 1 if args.action == "step" else args.max_steps
        require(limit is None or limit > 0, "--max-steps must be positive.")
        execute(state, resolve_runner(args.runner), limit)
        return
    plan, experiment_hash = load_state(state)
    if args.action == "resume":
        (state / "paused").unlink(missing_ok=True)
        print("Resumed. Use step or run to execute pending tasks.")
        return
    pending, _ = progress(state, plan, experiment_hash)
    total = sum(1 for _ in tasks(plan))
    paused = (state / "paused").exists()
    if worker_active(state):
        phase = "pausing" if paused else "running"
    elif pending:
        phase = "paused" if paused else "ready"
    else:
        phase = "complete"
    print(f"{plan['id']} — {phase}. {total - len(pending)}/{total} tasks completed.")
    if phase == "pausing":
        print("A worker is finishing its current task. Wait for paused or complete before disconnecting.")
    if pending:
        print(f"Next task: {pending[0][0]}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        sys.exit(f"Experiment error: {error}")
