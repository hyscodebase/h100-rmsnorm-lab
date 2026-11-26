"""Benchmark the complete two-output BF16 residual-add and RMSNorm operator."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTH, OPS, ROUNDS = 4096, 32, 21
EAGER = "eager_explicit"
CANDIDATE = "triton_1row_4warps"
MODES = ("default", "max-autotune-no-cudagraphs")
COMPILED = {f"compiled_{spelling}_{mode}": (spelling, mode)
            for spelling in ("explicit", "native") for mode in MODES}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def driver_version():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True, timeout=10)
    versions = set(proc.stdout.splitlines())
    if len(versions) != 1 or not re.fullmatch(r"[\d.]+", next(iter(versions))):
        raise RuntimeError("Could not establish a single NVIDIA driver version.")
    return versions.pop()


def schedule(backends, seed):
    order, rng = list(backends), random.Random(seed)
    rounds = []
    for _ in range(ROUNDS):
        rng.shuffle(order)
        rounds.append(order.copy())
    return rounds


def latency(graph_ms):
    if len(graph_ms) != ROUNDS or not all(math.isfinite(ms) and ms > 0 for ms in graph_ms):
        raise ValueError("Expected 21 positive CUDA event intervals.")
    samples = [ms * 1000 / OPS for ms in graph_ms]
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return {"graph_elapsed_ms": graph_ms, "samples_us": samples, "q1_us": q1, "median_us": median, "q3_us": q3}


def comparison(results):
    eligible = [name for name in COMPILED if results[name]["status"] == "measured"]
    best = min(eligible, key=lambda name: results[name]["timing"]["median_us"]) if eligible else None
    valid = best is not None and all(results[name]["status"] == "measured" for name in (EAGER, CANDIDATE))
    candidate = results[CANDIDATE].get("timing", {}).get("median_us")
    reference = results[best]["timing"]["median_us"] if best else None
    return {"eligible": valid, "fastest_correct_compiled": best, "compiled_variants_measured": eligible,
            "native_only_compiled_reference": bool(eligible) and all(COMPILED[name][0] == "native" for name in eligible),
            "candidate_median_us": candidate, "reference_median_us": reference,
            "latency_reduction_pct": 100 * (1 - candidate / reference) if valid else None}


def error_info(exc):
    # Compiler exceptions can contain source paths, device identifiers and multi-line build commands.
    # Preserve a safe first diagnostic line as well as exception classes from the causal chain.
    lines = str(exc).strip().splitlines()
    message = lines[0][:300] if lines else ""
    message = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s,;]+", "<path>", message)
    message = re.sub(r"(?:GPU-|MIG-)?[\da-fA-F]{8}(?:-[\da-fA-F]{4}){3}-[\da-fA-F]{12}", "<device>", message)
    message = re.sub(r"0x[\da-fA-F]+", "<address>", message)
    causes = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        causes.append(type(exc).__name__)
        exc = exc.__cause__ or exc.__context__
    return {"exception_types": causes, "detail": message}


def check_disjoint(outputs, inputs, byte_span):
    # Distinct views may share an allocation. Compare the actual byte spans of
    # the dense outputs rather than rejecting separate regions of pooled memory.
    spans = sorted(byte_span(t) for t in (*inputs, *(tensor for pair in outputs for tensor in pair)))
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise AssertionError("Captured input or output storage overlaps another live tensor.")


def capture(fn, inputs):
    import torch
    from check import byte_span
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    pairs = []
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(OPS):
            pairs.append(fn(*inputs))
    stream.synchronize()
    check_disjoint(pairs, inputs, byte_span)
    return graph, pairs


def verify_replay(graph, pairs, inputs, expected, verify_outputs):
    import torch
    # Poison every live output so that stale warmup or capture-time values
    # cannot pass the independent oracle check when the graph fails to write.
    for updated, normalized in pairs:
        updated.fill_(float("nan"))
        normalized.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    return [verify_outputs(pair, expected, inputs) for pair in pairs]


def time_graphs(torch, captured, order):
    elapsed = {name: [] for name in captured}
    first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for graph, _ in captured.values():
        graph.replay()
    first.record()
    last.record()
    last.synchronize()
    for group in order:
        for name in group:
            first.record()
            captured[name][0].replay()
            last.record()
            last.synchronize()
            elapsed[name].append(first.elapsed_time(last))
    return elapsed


def experiment(rows, seed):
    import torch
    import triton
    from check import eager_reference, native_reference, oracle, validate_backend, verify_outputs
    from kernel import fused_residual_rmsnorm

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a BF16-capable NVIDIA H100.")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if "H100" not in props.name or (props.major, props.minor) != (9, 0):
        raise RuntimeError("The preregistered GPU is an NVIDIA H100.")
    revision = os.getenv("BENCH_REVISION")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("BENCH_REVISION must be a full clean commit ID.")
    started, wall_start = utc_now(), time.perf_counter()
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ("benchmark.py", "check.py", "kernel.py", "PLAN.md")}
    env = {"gpu": props.name, "compute_capability": [props.major, props.minor],
           "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
           "driver": driver_version(), "cuda_runtime": torch.version.cuda, "torch": str(torch.__version__),
           "triton": triton.__version__, "python": platform.python_version()}
    spelling = {"explicit": eager_reference, "native": native_reference}

    with torch.inference_mode():
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
                       for shape in ((rows, WIDTH), (rows, WIDTH), (WIDTH,)))
        originals = tuple(t.clone() for t in inputs)
        expected = oracle(*originals)
        prepared, captured, results = {}, {}, {}

        def unchanged():
            if any(not torch.equal(t.view(torch.int16), original.view(torch.int16))
                   for t, original in zip(inputs, originals)):
                raise AssertionError("Backend mutated a reusable input.")

        def exclude(info, phase, exc):
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise exc
            # An asynchronous device failure invalidates subsequent comparisons.
            torch.cuda.synchronize()
            info.update(status="excluded", excluded_at=phase, error=error_info(exc))
            for t, original in zip(inputs, originals):
                t.copy_(original)

        for name in (EAGER, CANDIDATE, *COMPILED):
            info = results[name] = {"status": "pending"}
            phase = "compile"
            try:
                if name in COMPILED:
                    style, mode = COMPILED[name]
                    fn = torch.compile(spelling[style], mode=mode, fullgraph=True, dynamic=False)
                else:
                    fn = eager_reference if name == EAGER else fused_residual_rmsnorm
                phase = "numerical_fixtures"
                info["correctness"] = validate_backend(fn, rows, seed, device)
                if not info["correctness"]["passed"]:
                    info.update(status="excluded", excluded_at=phase)
                    continue
                phase = "warmup"
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        fn(*inputs)
                stream.synchronize()
                unchanged()
                prepared[name] = fn
            except Exception as exc:
                exclude(info, phase, exc)

        for name, fn in prepared.items():
            info = results[name]
            phase = "capture"
            try:
                graph, pairs = capture(fn, inputs)
                phase = "verification_replay"
                info["replay_checks"] = verify_replay(graph, pairs, inputs, expected, verify_outputs)
                unchanged()
                captured[name] = (graph, pairs)
            except Exception as exc:
                exclude(info, phase, exc)

        order = schedule(captured, seed)
        timing_started = utc_now()
        timed = time.perf_counter()
        measurements = time_graphs(torch, captured, order) if captured else {}
        timing_elapsed = time.perf_counter() - timed
        unchanged()
        for name, values in measurements.items():
            results[name].update(status="measured", timing=latency(values))

    return {"started_utc": started, "finished_utc": utc_now(), "wall_elapsed_s": time.perf_counter() - wall_start,
            "code_revision": revision, "source_sha256": sources, "environment": env,
            "rows": rows, "width": WIDTH, "seed": seed, "dtype": "bfloat16", "eps": 1e-6,
            "method": {"rounds": ROUNDS, "ops_per_graph": OPS, "warmup_calls": 5,
                       "timing": "CUDA events around one graph replay; microseconds per two-output call",
                       "cache": "reused input addresses without an explicit L2 flush",
                       "outputs": "all 32 updated and normalized tensor pairs remain live during graph timing",
                       "timing_started_utc": timing_started, "timing_phase_wall_s": timing_elapsed, "round_order": order},
            "backends": results, "comparison": comparison(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True, choices=(1, 16, 128, 1024))
    parser.add_argument("--seed", type=int, required=True, choices=(101, 102, 103))
    parser.add_argument("--output", type=Path, help="Write a complete JSON checkpoint atomically; default is stdout.")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = experiment(args.rows, args.seed)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
