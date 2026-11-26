"""Coordinate one preflight or one prepared H100 observer arm per step."""
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "experiments/observer/benchmark.py"
SOURCES = (SOURCE, "experiments/observer/sensors.py", "experiments/prefill-gemm/benchmark.py")
KIND = "observer_coverage"


def validate_plan(plan):
    if (plan["seeds"] != list(range(51, 57)) or plan["arm_order"] != ["ABC", "BCA", "CAB", "CBA", "BAC", "ACB"]
            or [plan[k] for k in ("seconds", "attempt_seconds", "allocation_seconds", "extra_compute_attempts")]
            != [45, 90, 1800, 2]):
        raise ValueError("The observer plan differs from the preregistered schedule or budget.")


def tasks(plan):
    yield "preflight", ("preflight",)
    for seed, order in zip(plan["seeds"], plan["arm_order"]):
        for arm in order:
            yield f"s{seed}-{arm}", (seed, arm)


def arguments(shape):
    return ["--preflight"] if shape == ("preflight",) else ["--seed", str(shape[0]), "--arm", shape[1]]


def launch_limit(state, task, plan):
    attempts = [json.loads(path.read_bytes()) for path in (state / "attempts").glob("*.json")]
    compute = [a["task_id"] for a in attempts if a["task_id"] != "preflight"]
    if task == "preflight" and sum(a["task_id"] == task for a in attempts) >= 3:
        raise ValueError("Preflight retry limit reached.")
    if task in compute and len(compute) - len(set(compute)) >= plan["extra_compute_attempts"]:
        raise ValueError("The two extra compute attempts have already been used.")
    spent = 0.0
    now = datetime.now(timezone.utc)
    for a in attempts:
        if a["outcome"] == "checkpoint_saved":
            spent += a["controller_elapsed_s"]
        else:
            # Uncertain remote exits reserve the full watchdog interval, including cleanup.
            age = (now - datetime.fromisoformat(a["started_utc"])).total_seconds()
            if age < plan["attempt_seconds"] and (a["outcome"] in ("started", "interrupted")
                    or a.get("error_type") == "TimeoutExpired" or a.get("exit_code") == 255):
                raise ValueError("Previous remote watchdog has not expired. Retry after its 90-second window.")
            spent += plan["attempt_seconds"]
    remaining = min(plan["attempt_seconds"], math.floor(plan["allocation_seconds"] - spent))
    if remaining < (5 if task == "preflight" else plan["seconds"] + 3):
        raise ValueError("The remaining H100 allocation cannot cover this unit.")
    return remaining


def bounded_run(cmd, root, env, lock_fd, max_seconds):
    # The remote wrapper uses this same absolute deadline. The local timeout also covers transfer and SSH setup.
    env = {**env, "OBSERVER_DEADLINE_EPOCH": str(math.floor(time.time() + max_seconds))}
    deadline = time.monotonic() + max_seconds
    with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, text=True, env=env,
                          pass_fds=(lock_fd,), start_new_session=True) as child:
        try:
            raw, _ = child.communicate(timeout=max(0, deadline - time.monotonic() - 1))
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=max(0, min(0.5, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, cmd)
        return json.loads(raw)


def validate_result(result, shape):
    import re
    if not isinstance(result, dict) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", result.get("code_revision") or ""):
        raise ValueError("Missing frozen observer benchmark revision.")
    if shape == ("preflight",):
        if not (result.get("preflight") is True and result.get("passed") is True
                and result.get("physical_uuid_matched") is True and set(result.get("checks", {})) == {"cli", "nvml"}):
            raise ValueError("Both observers must verify the H100 physical device before compute.")
        for entry in result["checks"].values():
            values = entry.get("values") or {}
            activity = values.get("gpu_busy_pct")
            if entry.get("passed") is not True or type(activity) not in (int, float) or not 0 <= activity <= 100:
                raise ValueError("Preflight did not obtain a valid device activity sample.")
    else:
        from experiments.observer.analyze import validate
        validate(result, *shape)
    env = result["environment"]
    if "H100" not in env["gpu"]:
        raise ValueError("The observer benchmark requires an H100.")
    return {key: env[key] for key in ("gpu", "compute_capability", "total_memory_bytes", "sm_count",
                                      "driver", "cuda_runtime", "torch", "python")}
