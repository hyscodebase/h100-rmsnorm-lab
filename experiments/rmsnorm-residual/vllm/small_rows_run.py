"""Verify or time one registered 128-row split-warp reduction checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import trials
import grouped_run as grouped
import row_transfer as transfer
import run
import shuffle_run as shuffle
from wheel import MEMBER, sha

ROWS = 128
REPLAY_SEEDS, FRESH_SEED = transfer.TIMING_SEEDS, 129
NUMERIC_SEEDS = (*REPLAY_SEEDS, FRESH_SEED)
TIMING_SEEDS = (130, 131, 132)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
ARCHIVE = run.HERE / "transfer-results"
CANDIDATE = "triton_split_warps"
SOURCES = (*transfer.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    Path(__file__).resolve(), run.HERE / "small_rows_kernel.py", run.HERE / "SMALL_ROWS.md",
    *(ARCHIVE / f"r{ROWS}-s{s}.json" for s in (transfer.QUALIFICATION, *REPLAY_SEEDS)),
)))


def load_archived(seed):
    return json.loads((ARCHIVE / f"r{ROWS}-s{seed}.json").read_bytes())


def compile_split(torch, implementation):
    x = torch.zeros((ROWS, run.WIDTH), dtype=torch.bfloat16, device="cuda:0")
    compiled = implementation(x, x.clone(), torch.ones_like(x[0]))
    metadata = grouped.launch_metadata(compiled)
    ptx = compiled.asm["ptx"]
    offsets = [int(raw) for raw in re.findall(r"shfl\.sync\.bfly\.b32\s+\w+,\s*\w+,\s*(\d+),", ptx)]
    shared_loads = metadata["ptx_opcodes"].get("ld.shared.v4.f32", 0)
    wanted = [offset for offset in (1, 2, 4, 8, 16) for _ in range(2)]
    if metadata["num_warps"] != 8 or offsets != wanted or shared_loads != 4:
        raise ValueError("The compiled split warp reduction differs from the registered design.")
    metadata.update(warp_shuffle_offsets=offsets, emitted_shared_loads=shared_loads,
                    inline_cta_shared_bytes=64, binary_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest())
    return metadata


def check_replay(seed, fixtures, captures):
    if len(fixtures) != 8 or len(captures) != run.OPS:
        return False, False
    cases = fixtures + captures
    reproduced = True
    if seed in REPLAY_SEEDS:
        saved = load_archived(seed)
        reproduced = [r.get("comparisons", {}).get(shuffle.CANDIDATE) for r in cases] == (
            saved["fixtures"] + saved["replay_checks"])
        cases = [r.get("comparisons", {}).get(CANDIDATE, {}) for r in cases]
    exact = all(r.get("passed") and r.get("pairs_differ_bitwise") == 0 for r in cases)
    return exact, reproduced


def checkpoint(seed, asset_dir, previous, revision):
    import torch
    import triton
    from shuffle_kernel import fused_shuffle
    from small_rows_kernel import fused_split_warps

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = load_archived(transfer.QUALIFICATION)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not anchor["gate_passed"]:
        raise ValueError("Use the registered H100 and the frozen 128-row preflight.")
    environment = grouped.environment_metadata(torch, triton)
    library = asset_dir / Path(MEMBER).name
    if environment != anchor["environment"] or sha(library) != anchor["asset"]["extension_sha256"]:
        raise ValueError("The device or released vLLM extension differs.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    timing_requested = seed in TIMING_SEEDS
    if timing_requested and (
        [r["seed"] for r in previous] != list(NUMERIC_SEEDS)
        or any(not r["gate_passed"] or r["rows"] != ROWS or r["width"] != run.WIDTH
               or r["code_revision"] != revision or r["source_sha256"] != sources
               or r["environment"] != environment or r["asset"] != anchor["asset"] for r in previous)
    ):
        raise ValueError("All four numerical checkpoints must pass before timing.")
    torch.cuda.set_device(0)
    torch.ops.load_library(str(library.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {shuffle.REFERENCE: lambda x, r, w: op(x, r, w, 1e-6)}
    with torch.inference_mode():
        codegen = compile_split(torch, fused_split_warps)
        if any(codegen["ptx_instruction_sha256"] != r["codegen"]["ptx_instruction_sha256"] for r in previous):
            raise ValueError("The compiled split reduction has changed.")
        if seed in REPLAY_SEEDS:
            old = shuffle.compile_evidence(torch, fused_shuffle, rows=ROWS)
            if old["ptx_instruction_sha256"] != anchor["codegen"]["ptx_instruction_sha256"]:
                raise ValueError("The original shuffle changed since the archived 128-row run.")
            methods[shuffle.CANDIDATE] = fused_shuffle
        methods[CANDIDATE] = fused_split_warps
        fixtures, captures, timing = trials(torch, methods, seed, torch.device("cuda:0"), rows=ROWS,
                                            measure=timing_requested, require_bitwise=timing_requested)
        exact, reproduced = check_replay(seed, fixtures, captures)
    return {
        "seed": seed, "rows": ROWS, "width": run.WIDTH, "code_revision": revision, "source_sha256": sources,
        "environment": environment, "asset": anchor["asset"], "codegen": codegen,
        "archived_comparison_reproduced": reproduced if seed in REPLAY_SEEDS else None,
        "gate_passed": exact and reproduced and (not timing_requested or timing is not None),
        "fixtures": fixtures, "replay_checks": captures, "timing": timing,
        "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--qualified", type=Path, action="append", default=[])
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    prior = [json.loads(path.read_bytes()) for path in args.qualified]
    print(json.dumps(checkpoint(args.seed, args.asset_dir, prior, revision), indent=2, allow_nan=False))
