"""Replay the pinned vLLM reduction comparison or measure one qualified seed."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import run
from bounded_preflight import verify_case
from bounded_timing import SOURCES as TIMING_SOURCES, trials
from wheel import MEMBER, sha

REPLAY_SEED, PREFLIGHT_SEED = 114, 117
TIMING_SEEDS = (118, 119, 120)
SEEDS = (REPLAY_SEED, PREFLIGHT_SEED, *TIMING_SEEDS)
REFERENCE = run.BACKENDS[0]
CANDIDATE = "triton_grouped_8warps"
COMPILED_SOURCES = (run.HERE / "grouped_kernel.py", Path(__file__).resolve(), run.HERE / "REDUCTION_PLAN.md")
SOURCES = (*TIMING_SOURCES, *(str(path.relative_to(run.ROOT)) for path in COMPILED_SOURCES))


def launch_metadata(kernel):
    ptx = kernel.asm["ptx"]
    instructions = re.findall(r"^\s*(?:@!?%[a-z0-9]+\s+)?([a-z][a-z0-9]*(?:\.[a-z0-9_]+)*)\s+[^\n]+;",
                              ptx, flags=re.MULTILINE)
    canonical = "\n".join(line.split("//", 1)[0].rstrip() for line in ptx.splitlines()
                          if not line.lstrip().startswith((".loc ", ".file ", "//")))
    return {
        "num_warps": kernel.metadata.num_warps,
        "shared_bytes": kernel.metadata.shared,
        "registers_per_thread": kernel.n_regs,
        "spill_count": kernel.n_spills,
        "ttgir_sha256": hashlib.sha256(kernel.asm["ttgir"].encode()).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "ptx_instruction_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "ptx_opcodes": dict(sorted(Counter(instructions).items())),
        "fp_fusion": False,
        "logical_group_width": 8,
    }


def environment_metadata(torch, triton):
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory, "sm_count": props.multi_processor_count,
        "driver": run.original.driver_version(), "cuda": torch.version.cuda, "torch": str(torch.__version__),
        "triton": triton.__version__, "python": platform.python_version(),
    }


def apply(torch, fn, inputs):
    x, r, w = (tensor.clone() for tensor in inputs)
    fn(x, r, w)
    if not torch.equal(w.view(torch.int16), inputs[2].view(torch.int16)):
        raise AssertionError("mutated_weight")
    return x, r


def numerical(torch, seed, methods, previous):
    import check
    from float32_gate import assess, bf16_bits

    old_triton = None
    if previous is not None:
        from kernel import fused_inplace as old_triton

    historical = {row["fixture"]: row for row in previous["fixtures"]} if previous is not None else {}
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    rows, replay = [], []
    for name in (*check.FIXTURES, "weight_rounding", "signed_zero"):
        inputs = run.fixture(torch, check, name, generator, "cuda:0")
        reference, candidate = [apply(torch, fn, inputs) for fn in methods.values()]
        check_result = verify_case(torch, (reference, candidate), inputs, name)
        rows.append(check_result)
        if previous is None:
            continue

        original = apply(torch, old_triton, inputs)
        original_result = verify_case(torch, (reference, original), inputs, name)
        fields = ("strict_outside_tolerance", "bounded_rejected", "pairs_differ_bitwise", "rejected_coords")
        checks_equal = all(original_result[key] == historical[name][key] for key in fields)
        if not checks_equal:
            raise ValueError(f"Seed 114 four-warp outcome differs from the registered evidence for {name}.")
        updated, expected = run.reference(*inputs)
        _, rejected = assess(updated, inputs[2], expected, reference[0], original[0])
        coords = rejected.nonzero().tolist()
        raw = [bf16_bits(outputs[0][rejected]).tolist() for outputs in (reference, original, candidate)]
        replay.append({
            "fixture": name,
            "original_rejected": len(coords),
            "still_different": sum(want != got for want, got in zip(raw[0], raw[2])),
            "coordinates": [
                {"row": i, "column": j, "vllm": a, "triton_four_warp": b, "triton_grouped": c}
                for (i, j), a, b, c in zip(coords, *raw)
            ],
        })
    return rows, replay


def run_checkpoint(seed, asset_dir, historical, prior, revision):
    import torch
    import triton
    from grouped_kernel import fused_grouped

    started, wall_start = run.original.utc_now(), time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or historical["seed"] != REPLAY_SEED:
        raise ValueError("Use the registered H100 and the saved seed-114 record.")
    environment = environment_metadata(torch, triton)
    if environment != historical["environment"]:
        raise ValueError("The runtime differs from the registered H100 environment.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    if any(p["source_sha256"] != sources or p["code_revision"] != revision or p["asset"] != historical["asset"]
           for p in prior):
        raise ValueError("The kernel, checkpoint revision or official library differs from the frozen run.")
    if [p["seed"] for p in prior] != list(SEEDS[:(0 if seed == REPLAY_SEED else 1 if seed == PREFLIGHT_SEED else 2)]):
        raise ValueError("Supply the required numerical checkpoints in seed order.")
    if seed in TIMING_SEEDS and not all(p["gate_passed"] for p in prior):
        raise ValueError("Numerical qualification is required before timing.")

    lib = asset_dir / Path(MEMBER).name
    if sha(lib) != historical["asset"]["extension_sha256"]:
        raise ValueError("CUDA operator binary differs from the original reference.")
    torch.cuda.set_device(0)
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {REFERENCE: lambda x, r, w: op(x, r, w, 1e-6), CANDIDATE: fused_grouped}
    with torch.inference_mode():
        x = torch.zeros((run.ROWS, run.WIDTH), dtype=torch.bfloat16, device="cuda:0")
        compiled = fused_grouped(x, x.clone(), torch.ones_like(x[0]))
        codegen = launch_metadata(compiled)
        if codegen["num_warps"] != 8 or not any(name.startswith("rsqrt.") for name in codegen["ptx_opcodes"]):
            raise ValueError("Compiler output does not contain the registered launch and reciprocal square root.")
        if prior and codegen["ptx_instruction_sha256"] != prior[0]["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("Candidate instructions changed across checkpoints.")
        timing, graph_checks, old_coordinates = None, [], []
        if seed in TIMING_SEEDS:
            fixtures, graph_checks, timing = trials(torch, methods, seed, torch.device("cuda:0"))
            passed = timing is not None
        else:
            fixtures, old_coordinates = numerical(torch, seed, methods, historical if seed == REPLAY_SEED else None)
            passed = all(f["passed"] for f in fixtures) and all(not row["still_different"] for row in old_coordinates)

    return {
        "code_revision": revision, "source_sha256": sources, "asset": historical["asset"],
        "environment": environment, "seed": seed, "rows": run.ROWS, "width": run.WIDTH,
        "gate_passed": passed, "started_utc": started, "finished_utc": run.original.utc_now(),
        "wall_seconds": time.perf_counter() - wall_start, "historical_revision": historical["code_revision"],
        "codegen": codegen, "fixtures": fixtures, "old_coordinates": old_coordinates,
        "replay_checks": graph_checks, "timing": timing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, action="append", default=[])
    args = parser.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch("[0-9a-f]{40}|[0-9a-f]{64}", revision):
        parser.error("Provide a frozen Git revision.")
    baseline = json.loads(args.historical.read_text())
    prior = [json.loads(p.read_text()) for p in args.qualified]
    print(json.dumps(run_checkpoint(args.seed, args.asset_dir, baseline, prior, revision), indent=2, allow_nan=False))
