"""Offline checkpoint contract for the preregistered two-output residual RMSNorm experiment."""
import hashlib
import math
import random
import re
import statistics
import subprocess
from datetime import datetime
from functools import lru_cache
from itertools import product
from pathlib import Path

BASE = Path("experiments/rmsnorm-residual")
SOURCE = str(BASE / "benchmark.py")
SOURCES = tuple(str(BASE / name) for name in ("PLAN.md", "check.py", "kernel.py", "benchmark.py"))
EAGER, CANDIDATE = "eager_explicit", "triton_1row_4warps"
COMPILED = tuple(f"compiled_{spelling}_{mode}" for spelling, mode in
                 product(("explicit", "native"), ("default", "max-autotune-no-cudagraphs")))
BACKENDS = (EAGER, CANDIDATE, *COMPILED)
FIXTURES = ("random", "zero_rows", "cancellation", "tiny", "bf16_ties")
FIXED = {"rows": [1024, 1, 16, 128], "width": 4096, "seeds": [101, 102, 103], "ops": 32, "rounds": 21}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_plan(plan):
    require(all(plan.get(k) == v for k, v in FIXED.items()) and
            all(type(v) is int for v in (*plan["rows"], *plan["seeds"], plan["width"], plan["ops"], plan["rounds"])),
            "Residual plan differs from its preregistration.")


def tasks(plan):
    for rows, seed in product(plan["rows"], plan["seeds"]):
        yield f"s{seed}-r{rows}-w{plan['width']}", (seed, rows)


def arguments(shape):
    seed, rows = shape
    return ["--rows", str(rows), "--seed", str(seed)]


@lru_cache(maxsize=32)
def verify_revision(root, revision, sources):
    """Verify each source blob once per immutable revision and source-hash tuple."""
    require(isinstance(revision, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision),
            "A clean full Git commit ID is required for the residual experiment.")
    git = ["git", "--no-replace-objects", "-C", str(root), "cat-file"]
    kind = subprocess.run([*git, "-t", revision], capture_output=True)
    require(kind.returncode == 0 and kind.stdout.strip() == b"commit",
            "Missing residual Git commit. Use a clone containing the recorded experiment history.")
    for name, sha in sources:
        blob = subprocess.run([*git, "blob", f"{revision}:{name}"], capture_output=True)
        require(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == sha,
                f"Residual source differs from {name} in the recorded Git commit.")


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.utcoffset() is not None, "A checkpoint timestamp must contain its UTC offset.")
    return stamp


def diagnostics(check):
    require(all(finite(check[k]) for k in ("updated_sum", "normalized_sum", "max_abs_error"))
            and check["max_abs_error"] >= 0, "Invalid evidence for one of the two residual outputs.")


def correctness(record, seed, rows):
    expected = {"rows": rows, "width": 4096, "seed": seed, "eps": 1e-6, "rtol": 0.005, "atol": 1e-6}
    require(all(record.get(k) == v for k, v in expected.items()), "Residual correctness settings differ.")
    fixtures = record["fixtures"]
    require([f["fixture"] for f in fixtures] == list(FIXTURES), "Missing or duplicate residual fixtures.")
    failed = []
    for fixture in fixtures:
        require(type(fixture["passed"]) is bool, "Invalid fixture outcome.")
        if fixture["passed"]:
            diagnostics(fixture)
        else:
            reason = fixture.get("reason")
            require(isinstance(reason, str) and reason.strip(), "Missing numerical failure reason.")
            failed.append({"fixture": fixture["fixture"], "reason": reason})
    require(record["passed"] is (not failed) and record["failure"] == (failed[0] if failed else None),
            "Declared correctness outcome differs from individual fixtures.")
    return not failed


def validate_backend(name, info, seed, rows, rounds, ops):
    correct = correctness(info["correctness"], seed, rows) if "correctness" in info else None
    replay = info.get("replay_checks")
    if replay is not None:
        require(len(replay) == ops, "Each captured invocation requires both output checks.")
        for pair in replay:
            diagnostics(pair)
    if info["status"] == "excluded":
        phase = info["excluded_at"]
        require(phase in ("compile", "numerical_fixtures", "warmup", "capture", "verification_replay")
                and "timing" not in info, "Invalid excluded backend stage or timing.")
        if phase == "compile":
            require(name in COMPILED and correct is None, "Invalid compile exclusion.")
        elif phase in ("warmup", "capture", "verification_replay"):
            require(correct is True, "Late exclusion requires passing fixture checks.")
        if correct is False:
            require(phase == "numerical_fixtures", "Numerical rejection must identify the failing fixture stage.")
        else:
            err = info.get("error", {})
            kinds = err.get("exception_types")
            require(isinstance(kinds, list) and kinds and all(isinstance(k, str) and k for k in kinds)
                    and isinstance(err.get("detail"), str), "Missing backend exception evidence.")
        return None
    require(info["status"] == "measured" and correct is True and replay is not None and "excluded_at" not in info,
            "A measured residual backend requires both fixture and replay evidence.")
    timing = info["timing"]
    events, samples = timing["graph_elapsed_ms"], timing["samples_us"]
    require(len(events) == len(samples) == rounds and all(
        finite(ms) and ms > 0 and finite(us) and us > 0 and math.isclose(us, 1000 * ms / ops, rel_tol=1e-10)
        for ms, us in zip(events, samples)), "Residual samples differ from raw CUDA events.")
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    require(all(math.isclose(timing[k], v, rel_tol=1e-10) for k, v in zip(("q1_us", "median_us", "q3_us"), quartiles)),
            "Residual summary differs from its 21 raw samples.")
    return quartiles[1]


def validate_result(result, shape, plan):
    seed, rows = shape
    require((result["seed"], result["rows"], result["width"], result["dtype"], result["eps"])
            == (seed, rows, plan["width"], "bfloat16", 1e-6), "Residual shape or numerical contract differs.")
    method = result["method"]
    require(method["rounds"] == plan["rounds"] and method["ops_per_graph"] == plan["ops"] and method["warmup_calls"] == 5
            and method["timing"] == "CUDA events around one graph replay; microseconds per two-output call"
            and method["cache"] == "reused input addresses without an explicit L2 flush"
            and method["outputs"] == "all 32 updated and normalized tensor pairs remain live during graph timing",
            "Residual measurement settings differ.")
    phase_wall, wall = method["timing_phase_wall_s"], result["wall_elapsed_s"]
    require(finite(phase_wall) and finite(wall) and 0 <= phase_wall <= wall and wall > 0
            and timestamp(result["started_utc"]) <= timestamp(method["timing_started_utc"]) <= timestamp(result["finished_utc"]),
            "Invalid residual timing window.")
    backends = result["backends"]
    require(set(backends) == set(BACKENDS), "Missing or unexpected residual backend attempt.")
    medians = {name: validate_backend(name, backends[name], seed, rows, plan["rounds"], plan["ops"]) for name in BACKENDS}
    measured = [name for name in BACKENDS if medians[name] is not None]
    round_order = method["round_order"]
    require(len(round_order) == plan["rounds"], "Incomplete randomized schedule.")
    rng = random.Random(seed)
    for group in round_order:
        rng.shuffle(measured)
        require(group == measured, "Residual backend order differs from the seeded schedule.")
    event_s = sum(sum(backends[name]["timing"]["graph_elapsed_ms"]) for name in measured) / 1000
    require(event_s <= phase_wall * 1.01 + 0.002, "CUDA intervals exceed the sequential graph timing window.")
    compiled = [name for name in COMPILED if medians[name] is not None]
    best = min(compiled, key=medians.get) if compiled else None
    eligible = best is not None and all(medians[k] is not None for k in (CANDIDATE, EAGER))
    comp = result["comparison"]
    expected = {"eligible": eligible, "fastest_correct_compiled": best, "compiled_variants_measured": compiled,
                "native_only_compiled_reference": bool(compiled) and all(k.startswith("compiled_native_") for k in compiled),
                "candidate_median_us": medians[CANDIDATE], "reference_median_us": medians.get(best)}
    require(all(comp[k] == v for k, v in expected.items()), "Residual comparison mismatches measured backends.")
    reduction = comp["latency_reduction_pct"]
    if eligible:
        require(finite(reduction) and math.isclose(reduction, 100 * (1 - medians[CANDIDATE] / medians[best]),
                                                   rel_tol=1e-10, abs_tol=1e-10), "Invalid candidate reduction.")
    else:
        require(reduction is None, "An ineligible comparison must omit the speedup.")


def verify_sources(result, root, sources):
    require(set(sources) == set(SOURCES) and all(re.fullmatch("[0-9a-f]{64}", s) for s in sources.values()),
            "Incomplete residual source manifest.")
    expected = {Path(name).name: digest for name, digest in sources.items()}
    require(result["source_sha256"] == expected, "Reported residual source hashes differ from the frozen plan.")
    verify_revision(root, result["code_revision"], tuple(sorted(sources.items())))
