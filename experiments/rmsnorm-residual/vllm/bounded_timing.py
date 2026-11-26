"""Measure one fresh seed after the bounded FP32 preflight passes."""
import argparse
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

import run
from bounded_preflight import SOURCES as PREFLIGHT_SOURCES, verify_case
from wheel import MEMBER, sha

SEEDS = (114, 115, 116)
SOURCES = (*PREFLIGHT_SOURCES, str(Path(__file__).resolve().relative_to(run.ROOT)))


def verify_methods(torch, methods, outputs, inputs, case):
    if len(outputs) == 2:
        return verify_case(torch, outputs, inputs, case)
    comparisons = {name: verify_case(torch, (outputs[0], output), inputs, case)
                   for name, output in zip(tuple(methods)[1:], outputs[1:])}
    return {"fixture": case, "comparisons": comparisons,
            "passed": all(result["passed"] for result in comparisons.values())}


def trials(torch, methods, seed, device, rows=run.ROWS, *, measure=True, require_bitwise=False):
    import check

    if require_bitwise and len(methods) != 2:
        raise ValueError("The exact timing gate requires one candidate.")

    generator = torch.Generator(device=device).manual_seed(seed)
    fixtures = []
    for case in (*check.FIXTURES, "weight_rounding", "signed_zero"):
        inputs = run.fixture(torch, check, case, generator, device, rows=rows)
        outputs = []
        for fn in methods.values():
            x, residual, weight = (tensor.clone() for tensor in inputs)
            fn(x, residual, weight)
            check.ensure(torch.equal(weight.view(torch.int16), inputs[2].view(torch.int16)), "mutated_weight")
            outputs.append((x, residual))
        try:
            fixtures.append(verify_methods(torch, methods, outputs, inputs, case))
        except AssertionError as exc:
            fixtures.append({"fixture": case, "passed": False, "reason": str(exc)})
    if not all(record["passed"] for record in fixtures):
        return fixtures, [], None

    generator = torch.Generator(device=device).manual_seed(seed)
    pair = tuple(torch.randn((run.OPS, rows, run.WIDTH), dtype=torch.bfloat16, device=device, generator=generator)
                 for _ in range(2))
    weight = torch.randn(run.WIDTH, dtype=torch.bfloat16, device=device, generator=generator)
    initial_weight = weight.clone()
    graphs = {name: run.capture_graph(torch, fn, pair, weight) for name, fn in methods.items()}
    for graph, x, residual in graphs.values():
        run.restore(pair, x, residual)
        check.ensure(torch.equal(x.view(torch.int16), pair[0].view(torch.int16)) and
                     torch.equal(residual.view(torch.int16), pair[1].view(torch.int16)), "restoration_bits")
        graph.replay()
    replay = []
    for i in range(run.OPS):
        try:
            outputs = [(x[i], residual[i]) for _, x, residual in graphs.values()]
            replay.append(verify_methods(torch, methods, outputs, (pair[0][i], pair[1][i], initial_weight), f"captured_{i}"))
        except AssertionError as exc:
            replay.append({"fixture": f"captured_{i}", "passed": False, "reason": str(exc)})
    check.ensure(torch.equal(weight.view(torch.int16), initial_weight.view(torch.int16)), "mutated_weight")
    if not measure or not all(record["passed"] for record in replay):
        return fixtures, replay, None
    if require_bitwise and any(record["pairs_differ_bitwise"] for record in fixtures + replay):
        return fixtures, replay, None

    first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    first.record()
    last.record()
    last.synchronize()
    order = run.original.schedule(graphs, seed)
    intervals = {name: [] for name in methods}
    started = run.original.utc_now()
    start = time.perf_counter()
    for round_order in order:
        for name in round_order:
            graph, x, residual = graphs[name]
            run.restore(pair, x, residual)
            first.record()
            graph.replay()
            last.record()
            last.synchronize()
            intervals[name].append(first.elapsed_time(last))
    elapsed = time.perf_counter() - start
    check.ensure(torch.equal(weight.view(torch.int16), initial_weight.view(torch.int16)), "mutated_weight")
    stats = {name: run.original.latency(raw) for name, raw in intervals.items()}
    medians = [stats[name]["median_us"] for name in methods]
    vllm, triton = medians[0], medians[-1]
    return fixtures, replay, {"ops_per_graph": run.OPS, "round_order": order, "inputs_restored_bitwise": True,
                              "started_utc": started, "wall_seconds": elapsed, "statistics": stats,
                              "triton_over_vllm": triton / vllm, "passes_5pct": triton <= vllm * .95}


def measure(seed, asset_dir, preflight, revision):
    import torch
    import triton
    from kernel import fused_inplace

    started, wall_start = run.original.utc_now(), time.perf_counter()
    if not preflight["gate_passed"] or preflight["seed"] != 113 or len(preflight["fixtures"]) != 8:
        raise ValueError("A passing fresh numerical preflight is required.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use the allocated H100.")
    props = torch.cuda.get_device_properties(0)
    environment = {"gpu": props.name, "compute_capability": [props.major, props.minor],
                   "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
                   "driver": run.original.driver_version(), "cuda": torch.version.cuda, "torch": str(torch.__version__),
                   "triton": triton.__version__, "python": platform.python_version()}
    if environment != preflight["environment"]:
        raise ValueError("H100 software or device differs from preflight.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    if preflight["source_sha256"] != {name: sources[name] for name in PREFLIGHT_SOURCES}:
        raise ValueError("The preflight source bytes differ.")
    lib = asset_dir / Path(MEMBER).name
    if sha(lib) != preflight["asset"]["extension_sha256"]:
        raise ValueError("CUDA operator binary differs from preflight.")
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {run.BACKENDS[0]: lambda x, r, w: op(x, r, w, 1e-6), run.BACKENDS[1]: fused_inplace}
    with torch.inference_mode():
        fixtures, replay, timing = trials(torch, methods, seed, torch.device("cuda:0"))
    return {"code_revision": revision, "source_sha256": sources, "asset": preflight["asset"],
            "preflight_revision": preflight["code_revision"], "environment": environment, "seed": seed,
            "rows": run.ROWS, "width": run.WIDTH, "gate_passed": timing is not None,
            "fixtures": fixtures, "replay_checks": replay, "timing": timing,
            "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - wall_start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch("[0-9a-f]{40}|[0-9a-f]{64}", revision):
        parser.error("Provide a frozen Git revision.")
    print(json.dumps(measure(args.seed, args.asset_dir, json.loads(args.preflight.read_text()), revision), indent=2, allow_nan=False))
