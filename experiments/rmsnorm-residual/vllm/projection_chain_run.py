"""Qualify and time residual RMSNorm between two BF16 projections."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import time

import copy_free_run as copy_free
import out_of_place_run as direct
import run
import split_add_tuning_run as tiles
import sustained_run as base
from wheel import MEMBER

ROWS, WIDTH = tiles.ROWS, tiles.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 177, (178, 179, 180)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
TILE, FUSED, REFERENCE = 8192, tiles.FUSED, copy_free.REFERENCE
SPLIT = tiles.TILE_BACKENDS[TILE]
BACKENDS = (SPLIT, FUSED)
OUTPUT_FIELDS = (*base.PAIR_FIELDS, "upstream_bit_differences_by_pair", "downstream_bit_differences_by_pair")
WORKSPACE = ":4096:8"
WEIGHT_SEED_OFFSET = 100_000
WEIGHT_SCALE = WIDTH ** -0.5
THRESHOLD = 1.01
ANCHOR = run.HERE / "split-add-tiles-512-results/s173.json"
SOURCES = (*tiles.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "COPY_FREE_PROJECTION_CHAIN_512.md",
)))


def expected_kernels(anchor):
    if anchor["selected_tile"] != TILE or not anchor["gate_passed"]:
        raise ValueError("The archived 8,192-element selection did not qualify.")
    return {
        SPLIT: [anchor["addition_codegen"][SPLIT]["residual_add"]["binary_sha256"],
                anchor["codegen"][tiles.ORIGINAL]["normalization"]["binary_sha256"]],
        FUSED: [anchor["codegen"][FUSED]["binary_sha256"]],
    }


def gemm_contract():
    return {"operator": "torch.mm", "dtype": "bfloat16", "shape_mkn": [ROWS, WIDTH, WIDTH],
            "matmuls_per_pair": 2, "output": "preallocated", "workspace_config": WORKSPACE,
            "allow_bf16_reduced_precision_reduction": False,
            "weight_seed_offset": WEIGHT_SEED_OFFSET, "weight_scale": WEIGHT_SCALE}


def make_work(torch, seed, device):
    rms_works, norm_weight, layout = base.make_work(torch, seed, device, rows=ROWS)
    works = [base.Work(work.stream, work.source, (*work.buffer, *(torch.empty_like(t) for t in work.buffer)))
             for work in rms_works]
    generator = torch.Generator(device=device).manual_seed(seed + WEIGHT_SEED_OFFSET)
    weights = tuple(torch.randn((WIDTH, WIDTH), dtype=torch.bfloat16, device=device, generator=generator)
                    .mul_(WEIGHT_SCALE) for _ in range(2))
    tensors = [tensor for work in works for tensor in (*work.source, *work.buffer)] + [norm_weight, *weights]
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors)
    if (any(start % 16 for start, _ in spans) or any(not t.is_contiguous() for t in tensors)
            or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))):
        raise ValueError("Projection inputs, intermediates and outputs must be aligned and disjoint.")
    return works, norm_weight, weights, {
        **layout, "disjoint_allocations": len(spans),
        "input_and_output_bytes": sum(end - start for start, end in spans),
        "writable_arrays_per_stream": len(works[0].buffer),
    }


def fingerprints(works, norm_weight, weights):
    return {**base.input_fingerprints(works, norm_weight),
            "projection_weight_sha256": [base.tensor_digest(weight) for weight in weights]}


def capture_chain(torch, fn, work, norm_weight, weights, *, expected=None):
    x, residual = work.source
    norm, residual_out, upstream, downstream = work.buffer
    first_weight, last_weight = weights
    capture_kernels = []

    def apply(index):
        torch.mm(x[index], first_weight, out=upstream[index])
        if expected is None:
            norm[index].copy_(upstream[index])
            residual_out[index].copy_(residual[index])
            fn(norm[index], residual_out[index], norm_weight)
            compiled = None
        else:
            compiled = fn(upstream[index], residual[index], norm_weight, norm[index], residual_out[index])
        torch.mm(norm[index], last_weight, out=downstream[index])
        return compiled

    def record():
        for index in range(run.OPS):
            capture_kernels.append(apply(index))

    graph = base.capture_graph(torch, work.stream, lambda: apply(0), record)
    if expected is not None:
        for compiled in capture_kernels:
            sequence = compiled if isinstance(compiled, tuple) else (compiled,)
            if [hashlib.sha256(kernel.asm["cubin"]).hexdigest() for kernel in sequence] != expected:
                raise ValueError("A captured residual kernel differs from seed 173.")
    return graph, {"pairs": len(capture_kernels), "residual_kernel_binary_sha256": expected}


def compare_outputs(torch, works, reference):
    return [{"stream": index, **{
        field: base.differences(torch, observed, expected)
        for field, observed, expected in zip(OUTPUT_FIELDS, work.buffer, correct, strict=True)
    }} for index, (work, correct) in enumerate(zip(works, reference, strict=True))]


def numeric_gate(numerical):
    guards, checks = numerical["alias_guards"], numerical["graph_checks"]
    return (set(guards) == set(BACKENDS) and all(len(g) == 3 and all(g.values()) for g in guards.values())
            and set(checks) == {REFERENCE, *BACKENDS}
            and all(record["finite_outputs"] and base.outputs_match(record["streams"], fields=OUTPUT_FIELDS)
                    and record["source_fingerprints"] == numerical["source_fingerprints"]
                    for record in checks.values()))


def verify_outputs(torch, runner, graphs, works, norm_weight, weights):
    reference, checks = [], {}
    for name, graph in graphs.items():
        base.poison_buffers(torch, works)
        runner.burst(graph, 1)
        if name == REFERENCE:
            for work in works:
                norm, _, upstream, downstream = work.buffer
                expected_r = (upstream.float() + work.source[1].float()).bfloat16()
                reference.append((norm.clone(), expected_r, upstream.clone(), downstream.clone()))
        checks[name] = {
            "streams": compare_outputs(torch, works, reference),
            "finite_outputs": all(bool(torch.isfinite(t).all()) for work in works for t in work.buffer),
            "source_fingerprints": fingerprints(works, norm_weight, weights),
        }
    return reference, checks


def chain_effect(timing):
    if not tiles.exact_timing(timing):
        return None
    rates = timing["median_rows_per_second"]
    return {"fused_gain": rates[FUSED] / rates[SPLIT] - 1,
            "passes_1pct": rates[FUSED] >= THRESHOLD * rates[SPLIT]}


def checkpoint(seed, asset_dir, qualification, revision):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != WORKSPACE:
        raise ValueError(f"Set CUBLAS_WORKSPACE_CONFIG={WORKSPACE} before running the chain.")
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place, split_out_of_place

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    expected = expected_kernels(anchor)
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    asset = anchor["asset"]
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    gemm = gemm_contract()
    pinned = {"rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
              "environment": environment, "asset": asset, "gemm": gemm}
    if seed in TIMING_SEEDS and (
        not qualification or qualification["seed"] != NUMERIC_SEED or not qualification["gate_passed"]
        or not numeric_gate(qualification["numerical"]) or any(qualification[key] != value for key, value in pinned.items())
    ):
        raise ValueError("An exact seed-177 checkpoint from the same revision and GEMM configuration is required.")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise RuntimeError("PyTorch did not apply the registered BF16 reduction setting.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {SPLIT: partial(split_out_of_place, tile=TILE), FUSED: fused_out_of_place}
        works, norm_weight, weights, layout = make_work(torch, seed, device)
        original = fingerprints(works, norm_weight, weights)
        alias_work = base.Work(works[0].stream, works[0].source, works[0].buffer[:2])
        guards = {name: direct.verify_alias_guards(fn, alias_work, norm_weight) for name, fn in methods.items()}
        runner, graphs, audits = base.Replay(torch, works), {}, {}
        for name, fn in {REFERENCE: vllm, **methods}.items():
            captures = [capture_chain(torch, fn, work, norm_weight, weights, expected=expected.get(name))
                        for work in works]
            graphs[name], audits[name] = map(list, zip(*captures, strict=True))
        reference, checks = verify_outputs(torch, runner, graphs, works, norm_weight, weights)
        numerical = {"source_fingerprints": original, "buffers": layout, "alias_guards": guards,
                     "graph_checks": checks, "capture_audits": audits}
        passed = numeric_gate(numerical)
        timing = None
        if passed and seed in TIMING_SEEDS:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, runner, graphs, works, reference, uuid, backends=BACKENDS,
                                  rows=ROWS, compare_outputs=compare_outputs)
            timing["inputs_unchanged_after_timing"] = fingerprints(works, norm_weight, weights) == original
            timing["outputs_match_after_timing"] = all(
                base.outputs_match(block["post_graph_checks"], fields=OUTPUT_FIELDS) for block in timing["blocks"]
            )
    return {**pinned, "seed": seed, "numerical": numerical, "gate_passed": passed, "timing": timing,
            "effect": chain_effect(timing), "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "started_utc": started,
            "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--qualified", type=Path)
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    result = checkpoint(args.seed, args.asset_dir, qualification, revision)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
