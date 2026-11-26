"""Qualify and time one seed of the preregistered 16-warp comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import bitwise_case, trials
import grouped_run as grouped
import run
import small_rows_run as small
from wheel import MEMBER, sha

ROWS = small.ROWS
NUMERIC_SEEDS, TIMING_SEEDS = (133,), (134, 135, 136)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
REFERENCE, EIGHT_WARPS, CANDIDATE = small.shuffle.REFERENCE, small.CANDIDATE, "triton_16warps"
BACKENDS = (REFERENCE, EIGHT_WARPS, CANDIDATE)
ANCHOR = run.HERE / "small-rows-results/s130.json"
SOURCES = (*small.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "sixteen_warp_kernel.py",
    run.HERE / "sixteen_warp.ttgir.in", run.HERE / "SMALL_ROWS_16W.md",
)))


def compile_sixteen(torch, implementation):
    x = torch.zeros((ROWS, run.WIDTH), dtype=torch.bfloat16, device="cuda:0")
    return warp_metadata(implementation(x, x.clone(), torch.ones_like(x[0])))


def warp_metadata(compiled):
    code = grouped.launch_metadata(compiled, ttgir=compiled.src.src)
    ptx = compiled.asm["ptx"]
    offsets = [int(value) for value in re.findall(r"shfl\.sync\.bfly\.b32\s+\w+,\s*\w+,\s*(\d+),", ptx)]
    loads = code["ptx_opcodes"].get("ld.shared.v4.f32", 0)
    threads = re.search(r"\.reqntid\s+(\d+)", ptx)
    if (not threads or int(threads.group(1)) != 512 or code["num_warps"] != 16
            or offsets != [1, 2, 4, 8, 16] or loads != 4 or code["ptx_opcodes"].get("bar.sync") != 1):
        raise ValueError("The compiler emitted an unexpected 16-warp reduction.")
    code.update(required_threads=512, warp_shuffle_offsets=offsets, emitted_shared_loads=loads, inline_cta_shared_bytes=64,
                binary_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                reduction_instructions=[line.strip() for line in ptx.splitlines()
                                        if re.match(r"\s*(shfl\.sync\.bfly\.|ld\.shared\.|bar\.sync\s+|@leader st\.shared\.)", line)])
    return code


def numerically_eligible(fixtures, captures):
    return len(fixtures) == 8 and len(captures) == run.OPS and all(bitwise_case(case) for case in fixtures + captures)


def comparisons(timing):
    if timing is None:
        return None
    stats = timing["statistics"]
    candidate = stats[CANDIDATE]["median_us"]
    return {name: {"latency_reduction": 1 - candidate / stats[name]["median_us"],
                   "passes_5pct": candidate <= stats[name]["median_us"] * .95}
            for name in BACKENDS[:-1]}


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from sixteen_warp_kernel import fused_sixteen_warps
    from small_rows_kernel import fused_split_warps

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not anchor["gate_passed"]:
        raise ValueError("Use the recorded H100 and the previously qualified 8-warp checkpoint.")
    environment = grouped.environment_metadata(torch, triton)
    library = asset_dir / Path(MEMBER).name
    if environment != anchor["environment"] or sha(library) != anchor["asset"]["extension_sha256"]:
        raise ValueError("The device or released extension differs from the anchor.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    measure = seed in TIMING_SEEDS
    if measure and (
        qualification is None or not qualification["gate_passed"]
        or (qualification["rows"], qualification["width"], qualification["seed"]) != (ROWS, run.WIDTH, NUMERIC_SEEDS[0])
        or qualification["source_sha256"] != sources or qualification["code_revision"] != revision
        or qualification["environment"] != environment or qualification["asset"] != anchor["asset"]
    ):
        raise ValueError("The timing seed requires the exact preflight from this revision.")
    torch.cuda.set_device(0)
    torch.ops.load_library(str(library.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {REFERENCE: lambda x, r, w: op(x, r, w, 1e-6),
               EIGHT_WARPS: fused_split_warps, CANDIDATE: fused_sixteen_warps}
    with torch.inference_mode():
        eight = small.compile_split(torch, fused_split_warps)
        if eight["ptx_instruction_sha256"] != anchor["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The 8-warp baseline instructions changed.")
        sixteen = compile_sixteen(torch, fused_sixteen_warps)
        if measure and sixteen["ptx_instruction_sha256"] != qualification["codegen"][CANDIDATE]["ptx_instruction_sha256"]:
            raise ValueError("The 16-warp instructions changed after qualification.")
        fixtures, captures, timing = trials(torch, methods, seed, torch.device("cuda:0"), rows=ROWS,
                                            measure=measure, require_bitwise=True)
    exact = numerically_eligible(fixtures, captures)
    return {
        "seed": seed, "rows": ROWS, "width": run.WIDTH, "code_revision": revision, "source_sha256": sources,
        "environment": environment, "asset": anchor["asset"], "codegen": {EIGHT_WARPS: eight, CANDIDATE: sixteen},
        "gate_passed": exact, "fixtures": fixtures, "replay_checks": captures, "timing": timing,
        "comparisons": comparisons(timing), "started_utc": started, "finished_utc": run.original.utc_now(),
        "wall_seconds": time.perf_counter() - clock,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--asset-dir", required=True, type=Path)
    cli.add_argument("--qualified", type=Path)
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    previous = json.loads(args.qualified.read_bytes()) if args.qualified else None
    print(json.dumps(checkpoint(args.seed, args.asset_dir, previous, revision), indent=2, allow_nan=False))
