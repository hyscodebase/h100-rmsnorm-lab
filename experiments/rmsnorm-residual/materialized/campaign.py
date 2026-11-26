"""Run one seed for the BF16 materialization and explicit rounding comparison."""
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
sys.path[:0] = [str(HERE.parent), str(ROOT)]

import benchmark as original
import materialized_plan as protocol

FLAG = "TORCHINDUCTOR_EMULATE_PRECISION_CASTS"


def raw_bf16(tensor):
    import torch

    return (tensor.view(torch.int16).to(torch.int32) & 0xffff).tolist()


def signed_zero_check(fn, device):
    import torch
    import check

    x = torch.zeros((protocol.ROWS, protocol.WIDTH), dtype=torch.bfloat16, device=device)
    x[0].fill_(-0.0)
    residual = x.clone()
    weight = x.new_ones(protocol.WIDTH)
    originals = x.clone(), residual.clone(), weight.clone()
    inputs = x, residual, weight
    outputs = fn(*inputs)
    check.verify_outputs(outputs, check.oracle(*originals), inputs)
    check.ensure(all(torch.equal(t.view(torch.int16), copy.view(torch.int16))
                     for t, copy in zip(inputs, originals)), "mutated_signed_zero_inputs")
    return {name: raw_bf16(out[:2, 0].contiguous())
            for name, out in zip(("updated_bits", "normalized_bits"), outputs)}


def midpoint_check(mode, device):
    import torch
    from baselines import round_float32

    words, expected = protocol.midpoint_bits()
    values = torch.tensor(words, dtype=torch.int32, device=device).view(torch.float32)
    round_with_compiler = torch.compile(round_float32, mode=mode, fullgraph=True, dynamic=False)
    updated, reconstructed = round_with_compiler(values)
    return {"expected_bf16_bits": expected, "returned_bf16_bits": raw_bf16(updated),
            "reconstructed_fp32_bf16_bits": ((reconstructed.view(torch.int32) >> 16) & 0xffff).tolist(),
            "reconstructed_low_bits_are_zero": not bool((reconstructed.view(torch.int32) & 0xffff).any().item())}


def run(seed, selected, revision):
    import torch
    import triton
    import check
    from torch._inductor import config
    from baselines import build_compiled
    from kernel import fused_residual_rmsnorm

    started, wall_start = original.utc_now(), time.perf_counter()
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1
            or str(torch.__version__).split("+", 1)[0] != "2.7.1"):
        raise ValueError("This experiment requires one H100 with CUDA PyTorch 2.7.1.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if "H100" not in props.name or (props.major, props.minor) != (9, 0) or config.emulate_precision_casts is not False:
        raise ValueError("Expected an H100 with precision cast emulation disabled.")
    env = {"gpu": props.name, "compute_capability": [props.major, props.minor],
           "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
           "driver": original.driver_version(), "cuda_runtime": torch.version.cuda, "torch": str(torch.__version__),
           "torch_git_revision": torch.version.git_version, "triton": triton.__version__, "python": platform.python_version(),
           "inductor_config_sha256": hashlib.sha256(Path(config.__file__).read_bytes()).hexdigest()}
    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in protocol.SOURCES}
    preflight = seed == protocol.SEEDS[0]
    controls = {protocol.EAGER: check.eager_reference, protocol.TRITON: fused_residual_rmsnorm}
    backends, prepared, captured = {}, {}, {}

    with torch.inference_mode():
        for name in selected:
            info = backends[name] = {}
            phase = "compile"
            try:
                if name in controls:
                    fn = controls[name]
                else:
                    style, mode = protocol.COMPILED[name]
                    fn = build_compiled(style, mode, check.eager_reference, check.native_reference)
                phase = "fixtures"
                info["correctness"] = check.validate_backend(fn, protocol.ROWS, seed, device)
                phase = "signed_zero"
                info["signed_zero"] = signed_zero_check(fn, device)
                signed_good = all(bits == protocol.SIGNED_ZERO_BITS for bits in info["signed_zero"].values())
                if name.startswith("compiled_bitcast_"):
                    phase = "midpoints"
                    midpoints = info["midpoints"] = midpoint_check(mode, device)
                    expected = midpoints["expected_bf16_bits"]
                    check.ensure(midpoints["returned_bf16_bits"] == midpoints["reconstructed_fp32_bf16_bits"] == expected
                                 and midpoints["reconstructed_low_bits_are_zero"], "midpoint_encoding")
                if not info["correctness"]["passed"]:
                    phase = "fixtures"
                    check.ensure(False, info["correctness"]["failure"]["reason"])
                phase = "signed_zero"
                check.ensure(signed_good, "signed_zero_output")
                info["status"] = "verified"
                prepared[name] = fn
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                torch.cuda.synchronize()
                info.update(status="excluded", excluded_at=phase,
                            reason=str(exc) if isinstance(exc, AssertionError) and re.fullmatch("[a-z_]+", str(exc))
                            else type(exc).__name__, error=original.error_info(exc))

        order, timing_wall, medians = [], 0, {}
        if not preflight and prepared:
            generator = torch.Generator(device=device).manual_seed(seed)
            shapes = ((protocol.ROWS, protocol.WIDTH), (protocol.ROWS, protocol.WIDTH), (protocol.WIDTH,))
            inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator) for shape in shapes)
            originals = tuple(t.clone() for t in inputs)
            expected = check.oracle(*originals)

            def ensure_unchanged():
                check.ensure(all(torch.equal(t.view(torch.int16), copy.view(torch.int16))
                                 for t, copy in zip(inputs, originals)), "mutated_inputs")

            for name, fn in prepared.items():
                info = backends[name]
                phase = "warmup"
                try:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(5):
                            fn(*inputs)
                    stream.synchronize()
                    ensure_unchanged()
                    phase = "capture"
                    graph, pairs = original.capture(fn, inputs)
                    phase = "replay"
                    info["replay_checks"] = original.verify_replay(graph, pairs, inputs, expected, check.verify_outputs)
                    ensure_unchanged()
                    captured[name] = (graph, pairs)
                except Exception as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    torch.cuda.synchronize()
                    info.update(status="excluded", excluded_at=phase, reason=type(exc).__name__,
                                error=original.error_info(exc))
                    for tensor, copy in zip(inputs, originals):
                        tensor.copy_(copy)
            if protocol.gate(captured):
                order = original.schedule(captured, seed)
                timing_start = time.perf_counter()
                measurements = original.time_graphs(torch, captured, order)
                timing_wall = time.perf_counter() - timing_start
                ensure_unchanged()
                for name, intervals in measurements.items():
                    stats = original.latency(intervals)
                    backends[name].update(status="measured", timing=stats)
                    medians[name] = stats["median_us"]

    verified = [name for name in selected if backends[name]["status"] != "excluded"]
    return {"phase": "preflight" if preflight else "timing", "seed": seed, "rows": protocol.ROWS,
            "width": protocol.WIDTH, "dtype": "bfloat16", "emulate_precision_casts": config.emulate_precision_casts,
            "code_revision": revision, "source_sha256": sources, "environment": env,
            "started_utc": started, "finished_utc": original.utc_now(), "wall_elapsed_s": time.perf_counter() - wall_start,
            "selected_backends": selected, "backends": backends, "verified_backends": verified,
            "gate_passed": protocol.gate(verified), "comparison": protocol.comparison(medians) if medians else None,
            "method": {"rounds": protocol.ROUNDS, "ops_per_graph": protocol.OPS,
                       "timing_phase_wall_s": timing_wall, "round_order": order}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=protocol.SEEDS, required=True)
    args = parser.parse_args()
    eligible = os.environ.get("MATERIALIZED_PREFLIGHT_BACKENDS", "")
    if args.seed == protocol.SEEDS[0]:
        selected = list(protocol.BACKENDS)
    else:
        selected = eligible.split(",")
        if selected != [name for name in protocol.BACKENDS if name in selected] or not protocol.gate(selected):
            parser.error("Timed seeds require verified preflight backends from lab.py.")
    revision = os.getenv("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        parser.error("BENCH_REVISION must be a frozen Git commit ID.")
    os.environ[FLAG] = "0"
    with redirect_stdout(sys.stderr):
        result = run(args.seed, selected, revision)
    protocol.validate_result(result, (args.seed,))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
