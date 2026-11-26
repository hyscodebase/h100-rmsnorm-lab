"""Compare the official vLLM CUDA operator and a Triton kernel one seed at a time."""
import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(1, str(HERE.parent))
import benchmark as original
from wheel import MEMBER, WHEEL, WHEEL_HASH, sha

ROWS, WIDTH, OPS, ROUNDS = 1024, 4096, 32, 21
SEEDS = (109, 110, 111, 112)
BACKENDS = ("vllm_0_10_0", "triton_1row_4warps")
SOURCES = tuple(str(p.relative_to(ROOT)) for p in (
    HERE / "PLAN.md", HERE / "run.py", HERE / "kernel.py", HERE / "wheel.py", HERE.parent / "benchmark.py",
    HERE.parent / "check.py",
))


def reference(x, residual, weight):
    updated = (x.double() + residual.double()).bfloat16()
    z = updated.double()
    inverse = z.square().sum(-1, keepdim=True).div(WIDTH).add(1e-6).rsqrt()
    scaled = (z * inverse).bfloat16()
    return updated, scaled.double() * weight.double()


def fixture(torch, check, name, generator, device, rows=ROWS):
    if name == "weight_rounding":
        x = torch.tensor([1, 1.25, 2, .75], dtype=torch.bfloat16, device=device).repeat(rows, WIDTH // 4)
        weight = x.new_tensor([.375, 1.5625, .7421875, 1.1328125]).repeat(WIDTH // 4)
        return x, torch.zeros_like(x), weight
    if name == "signed_zero":
        x = torch.zeros((rows, WIDTH), dtype=torch.bfloat16, device=device)
        x[0].fill_(-0.)
        return x, x.clone(), x.new_ones(WIDTH)
    return check.make_inputs(name, rows, generator, device)


def validate(torch, check, name, fn, inputs):
    x, residual, weight = inputs
    expected = reference(*inputs)
    output, updated, scale = x.clone(), residual.clone(), weight.clone()
    fn(output, updated, scale)
    check.ensure(torch.equal(scale.view(torch.int16), weight.view(torch.int16)), "mutated_weight")
    evidence = check.verify_outputs((updated, output), expected, inputs)
    if name == "weight_rounding":
        check.ensure(output[0, 1].item() == expected[1][0, 1].bfloat16().item() == 1.46875, "weighted_rounding_bits")
        evidence["weight_probe_value"] = output[0, 1].item()
    if name == "signed_zero":
        bits = (output[:2, 0].view(torch.int16).to(torch.int32) & 0xffff).tolist()
        check.ensure(bits == [32768, 0], "normalized_signed_zero")
        evidence["normalized_zero_bits"] = bits
    return evidence


def numeric(torch, check, methods, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    report = {name: [] for name in methods}
    for case in (*check.FIXTURES, "weight_rounding", "signed_zero"):
        inputs = fixture(torch, check, case, generator, device)
        for name, fn in methods.items():
            try:
                result = validate(torch, check, case, fn, inputs)
            except AssertionError as exc:
                report[name].append({"fixture": case, "passed": False, "reason": str(exc)})
            else:
                report[name].append({"fixture": case, "passed": True, **result})
    return report


def capture_graph(torch, fn, pair, weight):
    x, residual = (t.clone() for t in pair)
    fn(x[0], residual[0], weight)
    x[0].copy_(pair[0][0])
    residual[0].copy_(pair[1][0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for index in range(OPS):
            fn(x[index], residual[index], weight)
    stream.synchronize()
    return graph, x, residual


def restore(pair, x, residual):
    x.copy_(pair[0])
    residual.copy_(pair[1])


def time_trials(torch, check, methods, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    pair = tuple(torch.randn((OPS, ROWS, WIDTH), dtype=torch.bfloat16, device=device, generator=gen)
                 for _ in range(2))
    weight = torch.randn(WIDTH, dtype=torch.bfloat16, device=device, generator=gen)
    original_weight = weight.clone()
    graphs = {name: capture_graph(torch, fn, pair, weight) for name, fn in methods.items()}
    for graph, x, residual in graphs.values():
        restore(pair, x, residual)
        check.ensure(torch.equal(x.view(torch.int16), pair[0].view(torch.int16)) and
                     torch.equal(residual.view(torch.int16), pair[1].view(torch.int16)), "restoration_bits")
        graph.replay()
    replay = {name: [] for name in methods}
    for idx in range(OPS):
        expected = reference(pair[0][idx], pair[1][idx], original_weight)
        for name, (_, x, residual) in graphs.items():
            replay[name].append(check.verify_outputs((residual[idx], x[idx]), expected, (pair[0], pair[1], weight)))
    check.ensure(torch.equal(weight.view(torch.int16), original_weight.view(torch.int16)), "mutated_weight")

    first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    first.record()
    last.record()
    last.synchronize()
    order = original.schedule(graphs, seed)
    raw = {name: [] for name in methods}
    for round_order in order:
        for name in round_order:
            graph, x, residual = graphs[name]
            restore(pair, x, residual)
            first.record()
            graph.replay()
            last.record()
            last.synchronize()
            raw[name].append(first.elapsed_time(last))
    check.ensure(torch.equal(weight.view(torch.int16), original_weight.view(torch.int16)), "mutated_weight")
    medians = {name: original.latency(samples) for name, samples in raw.items()}
    ratio = medians[BACKENDS[1]]["median_us"] / medians[BACKENDS[0]]["median_us"]
    return {"ops_per_graph": OPS, "round_order": order, "inputs_restored_bitwise": True,
            "replay_checks": replay, "statistics": medians, "triton_over_vllm": ratio}


def run(seed, asset_dir, recorded_asset, revision, first):
    import torch
    import triton
    import check
    from kernel import fused_inplace

    begin, clock = original.utc_now(), time.perf_counter()
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1
            or torch.version.cuda != "12.8" or str(torch.__version__).split("+")[0] != "2.7.1"):
        raise ValueError("Use one visible H100 with CUDA 12.8 and PyTorch 2.7.1.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if "H100" not in props.name or (props.major, props.minor) != (9, 0):
        raise ValueError("The registered device is an H100.")
    env = {"gpu": props.name, "compute_capability": [props.major, props.minor],
           "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
           "driver": original.driver_version(), "cuda": torch.version.cuda, "torch": str(torch.__version__),
           "triton": triton.__version__, "python": platform.python_version()}
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    extension = Path(MEMBER).name
    lib = asset_dir / extension
    asset = {key: recorded_asset[key] for key in ("version", "wheel", "wheel_sha256", "extension",
                                                  "extension_sha256", "extension_size_bytes")}
    if (asset["version"], asset["wheel"], asset["wheel_sha256"], asset["extension"]) != ("0.10.0", WHEEL, WHEEL_HASH, extension):
        raise ValueError("Unexpected wheel manifest.")
    if not lib.is_file() or not (asset_dir / WHEEL).is_file() or sha(lib) != asset["extension_sha256"]:
        raise ValueError("The prepared operator extension does not match the recorded wheel.")
    if first and (not first["gate_passed"] or first["seed"] != SEEDS[0] or first["asset"] != asset
                  or first["source_sha256"] != sources or first["code_revision"] != revision or first["environment"] != env):
        raise ValueError("The numerical preflight must pass on the same binary, sources and hardware.")

    error = None
    methods = {}
    try:
        torch.ops.load_library(str(lib.resolve()))
        op = torch.ops._C.fused_add_rms_norm
        methods[BACKENDS[0]] = lambda x, r, w: op(x, r, w, 1e-6)
    except (OSError, RuntimeError, AttributeError) as exc:
        error = original.error_info(exc)
    methods[BACKENDS[1]] = fused_inplace
    with torch.inference_mode():
        results = numeric(torch, check, methods, seed, device)
        passed = set(results) == set(BACKENDS) and all(item["passed"] for items in results.values() for item in items)
        timings = time_trials(torch, check, methods, seed, device) if passed and seed != SEEDS[0] else None
    return {"code_revision": revision, "source_sha256": sources, "started_utc": begin, "finished_utc": original.utc_now(),
            "wall_seconds": time.perf_counter()-clock, "seed": seed, "rows": ROWS, "width": WIDTH, "asset": asset,
            "environment": env, "backend_load_error": error, "gate_passed": passed, "fixtures": results, "timing": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--wheel-record", required=True, type=Path)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        parser.error("Set BENCH_REVISION to the frozen commit ID.")
    if (args.seed != SEEDS[0]) != bool(args.preflight):
        parser.error("Timing requires the saved numerical preflight result.")
    first = json.loads(args.preflight.read_text()) if args.preflight else None
    with redirect_stdout(sys.stderr):
        result = run(args.seed, args.asset_dir, json.loads(args.wheel_record.read_text()), revision, first)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
