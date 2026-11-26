"""Verify or time the ordered warp shuffle on one registered H100 seed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import grouped_run as grouped
import run
from bounded_timing import trials, verify_methods
from wheel import MEMBER, sha

REPLAY_SEED, PREFLIGHT_SEED = 114, 121
TIMING_SEEDS = (122, 123, 124)
SEEDS = (REPLAY_SEED, PREFLIGHT_SEED, *TIMING_SEEDS)
REFERENCE, GROUPED, CANDIDATE = grouped.REFERENCE, grouped.CANDIDATE, "triton_shuffle_8warps"
SOURCES = (*grouped.SOURCES, *(str(p.relative_to(run.ROOT)) for p in
            (Path(__file__).resolve(), run.HERE / "shuffle_kernel.py", run.HERE / "SHUFFLE_PLAN.md")))


def old_positions(torch, prior, outputs):
    from float32_gate import bf16_bits

    if not prior:
        return []
    result = []
    for row in prior:
        i, j = row["row"], row["column"]
        actual = [bf16_bits(output[0][i:i+1, j:j+1]).item() for output in outputs]
        if actual[:2] != [row["vllm"], row["triton_grouped"]]:
            raise ValueError("Persisted vLLM and grouped encodings must reproduce.")
        result.append({"row": i, "column": j, "vllm": actual[0], "triton_grouped": actual[1],
                       "triton_shuffle": actual[2]})
    return result


def numerical(torch, methods, seed, witness):
    import check

    prior = {item["fixture"]: item["coordinates"] for item in witness["old_coordinates"]} if witness else {}
    fixtures, coordinates = [], []
    gen = torch.Generator(device="cuda:0").manual_seed(seed)
    for name in (*check.FIXTURES, "weight_rounding", "signed_zero"):
        inputs = run.fixture(torch, check, name, gen, "cuda:0")
        outputs = [grouped.apply(torch, fn, inputs) for fn in methods.values()]
        try:
            fixtures.append(verify_methods(torch, methods, outputs, inputs, name))
        except AssertionError as exc:
            fixtures.append({"fixture": name, "passed": False, "reason": str(exc)})
        points = old_positions(torch, prior.get(name), outputs)
        coordinates.extend({"fixture": name, **point} for point in points)
    if witness and len(coordinates) != 47:
        raise ValueError("The numerical replay must cover all 47 registered coordinates.")
    return fixtures, coordinates


def compile_evidence(torch, implementation, rows=run.ROWS):
    a = torch.zeros((rows, run.WIDTH), dtype=torch.bfloat16, device="cuda:0")
    kernel = implementation(a, a.clone(), torch.ones_like(a[0]))
    data = grouped.launch_metadata(kernel)
    ptx = kernel.asm["ptx"]
    shared_loads = sum(n for name, n in data["ptx_opcodes"].items() if name.startswith("ld.shared."))
    offsets = [int(x) for x in re.findall(r"shfl\.sync\.bfly\.b32\s+\w+,\s*\w+,\s*(\d+),", ptx)]
    if data["num_warps"] != 8 or offsets != [1, 2, 4, 8, 16] or not 0 < shared_loads < 128:
        raise ValueError(f"Unexpected shuffle tree or shared loads: {offsets}, {shared_loads}.")
    data.update(warp_shuffle_offsets=offsets, emitted_shared_loads=shared_loads,
                inline_cta_shared_bytes=32, logical_thread_guard=True,
                binary_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest())
    data["reduction_instructions"] = [
        line.strip() for line in ptx.splitlines()
        if re.match(r"\s*(shfl\.sync\.bfly\.|ld\.shared\.|bar\.sync\s+|@leader st\.shared\.)", line)
    ]
    return data


def checkpoint(seed, asset_dir, historical, prior, revision):
    import torch
    import triton
    from grouped_kernel import fused_grouped
    from shuffle_kernel import fused_shuffle

    started, clock = run.original.utc_now(), time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or historical["seed"] != REPLAY_SEED:
        raise ValueError("Use the saved grouped replay and the registered H100.")
    environment = grouped.environment_metadata(torch, triton)
    if environment != historical["environment"] or not historical["gate_passed"]:
        raise ValueError("Device environment or grouped gate differs from the registered result.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    if any(r["code_revision"] != revision or r["source_sha256"] != sources or r["asset"] != historical["asset"]
           for r in prior):
        raise ValueError("Checkpoint provenance has changed.")
    required_previous = list(SEEDS[:(0 if seed == REPLAY_SEED else 1 if seed == PREFLIGHT_SEED else 2)])
    if [r["seed"] for r in prior] != required_previous:
        raise ValueError("Supply the required numerical checkpoints in order.")
    if seed in TIMING_SEEDS and not all(r["gate_passed"] for r in prior):
        raise ValueError("Both numerical checkpoints must pass before timing.")
    lib = asset_dir / Path(MEMBER).name
    if sha(lib) != historical["asset"]["extension_sha256"]:
        raise ValueError("CUDA extension binary differs.")
    torch.cuda.set_device(0)
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {REFERENCE: lambda x, r, w: op(x, r, w, 1e-6), GROUPED: fused_grouped, CANDIDATE: fused_shuffle}
    with torch.inference_mode():
        codegen = compile_evidence(torch, fused_shuffle)
        if prior and codegen["ptx_instruction_sha256"] != prior[0]["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The compiled reduction changed after replay.")
        coordinates, replay, timing = [], [], None
        if seed in TIMING_SEEDS:
            fixtures, replay, timing = trials(torch, methods, seed, torch.device("cuda:0"))
            passed = timing is not None
        else:
            fixtures, coordinates = numerical(torch, methods, seed, historical if seed == REPLAY_SEED else None)
            passed = all(f["passed"] for f in fixtures) and all(p["vllm"] == p["triton_shuffle"] for p in coordinates)
    return {
        "seed": seed, "rows": run.ROWS, "width": run.WIDTH, "code_revision": revision,
        "source_sha256": sources, "environment": environment, "asset": historical["asset"],
        "codegen": codegen, "gate_passed": passed, "previous_coordinates": coordinates,
        "fixtures": fixtures, "replay_checks": replay, "timing": timing,
        "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, action="append", default=[])
    args = parser.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        parser.error("Provide a frozen Git revision.")
    old = json.loads(args.historical.read_text())
    previous = [json.loads(p.read_text()) for p in args.qualified]
    print(json.dumps(checkpoint(args.seed, args.asset_dir, old, previous, revision), indent=2, allow_nan=False))
