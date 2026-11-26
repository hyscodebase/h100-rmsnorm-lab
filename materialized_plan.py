"""Checkpoint and comparison rules for the correctly rounded compiled baselines."""
from itertools import product
import math
import random
import statistics

import residual_plan

KIND = "materialized_rmsnorm"
SEEDS = (105, 106, 107, 108)
ROWS, WIDTH, OPS, ROUNDS = 1024, 4096, 32, 21
BASE = "experiments/rmsnorm-residual"
SOURCE = f"{BASE}/materialized/campaign.py"
SOURCES = (SOURCE, f"{BASE}/materialized/PLAN.md", f"{BASE}/materialized/baselines.py",
           f"{BASE}/benchmark.py", f"{BASE}/check.py", f"{BASE}/kernel.py",
           "materialized_plan.py", "residual_plan.py")
EAGER, TRITON = residual_plan.EAGER, residual_plan.CANDIDATE
MODES = ("default", "max-autotune-no-cudagraphs")
COMPILED = {f"compiled_{style}_{mode}": (style, mode) for style, mode in
            product(("explicit", "native", "materialized", "bitcast"), MODES)}
BACKENDS = (EAGER, TRITON, *COMPILED)
SIGNED_ZERO_BITS = [32768, 0]
require = residual_plan.require


def validate_plan(plan):
    require([plan.get(key) for key in ("rows", "width", "seeds", "operations", "rounds")]
            == [ROWS, WIDTH, list(SEEDS), OPS, ROUNDS], "The materialized plan differs from its preregistration.")


def tasks(plan):
    for seed in plan["seeds"]:
        yield f"s{seed}-r{ROWS}-w{WIDTH}", (seed,)


def arguments(shape):
    return ["--seed", str(shape[0])]


def gate(eligible):
    styles = {style for name, (style, _) in COMPILED.items() if name in eligible}
    return EAGER in eligible and TRITON in eligible and {"materialized", "bitcast"} <= styles


def comparison(medians):
    compiled = [name for name in COMPILED if name in medians]
    best = min(compiled, key=medians.get, default=None)
    eligible = gate(medians)
    candidate, reference = medians.get(TRITON), medians.get(best)
    return {
        "eligible": eligible,
        "fastest_correct_compiled": best,
        "candidate_median_us": candidate,
        "reference_median_us": reference,
        "latency_reduction_pct": 100 * (1 - candidate / reference) if eligible else None,
        "passes_5pct": candidate <= reference * 0.95 if eligible else None,
    }


def validate_timing(timing):
    events = timing["graph_elapsed_ms"]
    require(len(events) == ROUNDS and all(residual_plan.finite(v) and v > 0 for v in events),
            "Invalid raw CUDA graph intervals.")
    samples = [1_000 * ms / OPS for ms in events]
    require(timing["samples_us"] == samples, "Reported call latencies differ from the raw graph intervals.")
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    require([timing[k] for k in ("q1_us", "median_us", "q3_us")] == [q1, median, q3],
            "Latency quartiles differ from the recorded samples.")
    return median


def validate_backend(name, info, seed, preflight):
    status = info["status"]
    require(status in ("excluded", "verified", "measured"), "Unknown backend status.")
    fixtures = info.get("correctness")
    correct = fixtures is not None and residual_plan.correctness(fixtures, seed, ROWS)
    signed = info.get("signed_zero")
    if signed is not None:
        for key in ("updated_bits", "normalized_bits"):
            bits = signed[key]
            require(len(bits) == 2 and all(type(bit) is int and 0 <= bit <= 65535 for bit in bits),
                    "Invalid signed-zero output encodings.")
    replay = info.get("replay_checks")
    if replay is not None:
        require(len(replay) == OPS, "Each captured call needs both output checks.")
        for evidence in replay:
            residual_plan.diagnostics(evidence)
    if status == "excluded":
        require(info.get("excluded_at") in ("compile", "fixtures", "signed_zero", "midpoints", "warmup",
                                             "capture", "replay") and isinstance(info.get("reason"), str)
                and bool(info["reason"]) and "timing" not in info, "Invalid backend exclusion.")
        return None
    require(correct and signed is not None and
            signed["updated_bits"] == signed["normalized_bits"] == SIGNED_ZERO_BITS and
            "excluded_at" not in info, "Verified variants need passing output and signed-zero checks.")
    if name.startswith("compiled_bitcast_"):
        p = info["midpoints"]
        expected = midpoint_bits()[1]
        require(p["expected_bf16_bits"] == p["returned_bf16_bits"] == p["reconstructed_fp32_bf16_bits"] == expected
                and p["reconstructed_low_bits_are_zero"] is True,
                "Incorrect midpoint rounding evidence for the bitcast graph.")
    if status == "measured":
        require(not preflight and replay is not None, "Only a verified follow-up graph can be timed.")
        return validate_timing(info["timing"])
    require("timing" not in info and (preflight == (replay is None)),
            "Invalid numerical-only checkpoint evidence.")
    return None


def midpoint_bits():
    """Registered exact FP32 words and expected encodings, including a mantissa carry."""
    words, expected = [], []
    for sign in (0, 0x8000):
        for upper in (0x3F80, 0x3F81, 0x3FFF):
            high = sign | upper
            for offset, increment in ((-1, 0), (0, high & 1), (1, 1)):
                word = (high << 16) | (0x8000 + offset)
                words.append(word if word < 2**31 else word - 2**32)
                expected.append(high + increment)
    return words + [0, -2**31], expected + [0, 32768]


def validate_result(result, shape, sources=None):
    (seed,) = shape
    preflight = seed == SEEDS[0]
    require((result["seed"], result["rows"], result["width"], result["phase"])
            == (seed, ROWS, WIDTH, "preflight" if preflight else "timing"), "Unexpected checkpoint identity.")
    require(result["emulate_precision_casts"] is False and result["dtype"] == "bfloat16",
            "Different precision setting or dtype.")
    wall = result["wall_elapsed_s"]
    require(residual_plan.timestamp(result["started_utc"]) <= residual_plan.timestamp(result["finished_utc"])
            and residual_plan.finite(wall) and wall > 0, "Invalid checkpoint time.")
    env = result["environment"]
    require("H100" in env["gpu"] and env["compute_capability"] == [9, 0]
            and env["torch"].split("+", 1)[0] == "2.7.1"
            and all(type(env[k]) is int and env[k] > 0 for k in ("total_memory_bytes", "sm_count"))
            and all(isinstance(env[k], str) and env[k] for k in ("driver", "cuda_runtime", "torch", "triton", "python")),
            "Unexpected H100 environment.")
    if sources is not None:
        require(result["source_sha256"] == sources, "Executed files differ from the registered source hashes.")
    selected = result["selected_backends"]
    require(selected == [name for name in BACKENDS if name in selected] and (not preflight or selected == list(BACKENDS))
            and set(result["backends"]) == set(selected), "Incomplete or unexpected backend selection.")
    infos = result["backends"]
    medians = {}
    for name in selected:
        median = validate_backend(name, infos[name], seed, preflight)
        if median is not None:
            medians[name] = median
    verified = [name for name in selected if infos[name]["status"] in ("verified", "measured")]
    require(result["verified_backends"] == verified and result["gate_passed"] is gate(verified),
            "Recorded gate differs from verified variants.")
    method = result["method"]
    order = method["round_order"]
    elapsed = method["timing_phase_wall_s"]
    require(method["ops_per_graph"] == OPS and method["rounds"] == ROUNDS
            and residual_plan.finite(elapsed) and 0 <= elapsed <= wall, "Unexpected timing method.")
    if preflight or not gate(verified):
        require(not order and not medians and elapsed == 0 and result["comparison"] is None,
                "Timing is ineligible before the numerical gate passes.")
    else:
        require(set(medians) == set(verified) and len(order) == ROUNDS, "Missing eligible timing data.")
        seeded = list(medians)
        rng = random.Random(seed)
        for round_order in order:
            rng.shuffle(seeded)
            require(round_order == seeded, "CUDA graph measurement order differs from the registered schedule.")
        cuda_seconds = sum(sum(infos[n]["timing"]["graph_elapsed_ms"]) for n in medians) / 1_000
        require(cuda_seconds <= elapsed * 1.01 + 0.002, "CUDA events exceed the timed wall interval.")
        require(result["comparison"] == comparison(medians), "Comparison differs from the verified latencies.")
    return env
