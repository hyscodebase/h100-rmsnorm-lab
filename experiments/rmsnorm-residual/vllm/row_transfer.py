"""Run one row-count and seed from the predeclared H100 transfer study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_preflight import verify_case
from bounded_timing import trials
import grouped_run as grouped
import run
import shuffle_run as shuffle
from wheel import MEMBER, sha

ROWS = (128, 256, 512, 2048)
QUALIFICATION = 125
TIMING_SEEDS = (126, 127, 128)
SOURCES = (*shuffle.SOURCES, *(str(p.relative_to(run.ROOT)) for p in
            (Path(__file__).resolve(), run.HERE / "ROW_TRANSFER.md")))


def numerical(torch, methods, rows):
    import check

    generator = torch.Generator(device="cuda:0").manual_seed(QUALIFICATION)
    cases = []
    for case in (*check.FIXTURES, "weight_rounding", "signed_zero"):
        inputs = run.fixture(torch, check, case, generator, "cuda:0", rows=rows)
        outputs = [grouped.apply(torch, fn, inputs) for fn in methods.values()]
        try:
            cases.append(verify_case(torch, outputs, inputs, case))
        except AssertionError as error:
            cases.append({"fixture": case, "passed": False, "reason": str(error)})
    return cases


def checkpoint(rows, seed, asset_dir, baseline, previous, revision):
    import torch
    import triton
    from shuffle_kernel import fused_shuffle

    started, start = run.original.utc_now(), time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or baseline["seed"] != 114:
        raise ValueError("Use the registered H100 and the frozen shuffle replay.")
    environment = grouped.environment_metadata(torch, triton)
    if not baseline["gate_passed"] or environment != baseline["environment"]:
        raise ValueError("The official reference or device environment differs.")
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    if seed in TIMING_SEEDS and (
        previous is None or not previous["gate_passed"] or previous["seed"] != QUALIFICATION
        or previous["rows"] != rows or previous["source_sha256"] != sources or previous["code_revision"] != revision
    ):
        raise ValueError("A passing numerical preflight for this exact shape is required.")
    lib = asset_dir / Path(MEMBER).name
    if sha(lib) != baseline["asset"]["extension_sha256"]:
        raise ValueError("The reference binary differs.")
    torch.cuda.set_device(0)
    torch.ops.load_library(str(lib.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    methods = {shuffle.REFERENCE: lambda x, r, w: op(x, r, w, 1e-6), shuffle.CANDIDATE: fused_shuffle}
    with torch.inference_mode():
        compiler = shuffle.compile_evidence(torch, fused_shuffle, rows=rows)
        if compiler["ptx_instruction_sha256"] != baseline["codegen"]["ptx_instruction_sha256"]:
            raise ValueError("The candidate's canonical instructions differ from the frozen 1,024-row kernel.")
        if seed == QUALIFICATION:
            fixtures = numerical(torch, methods, rows)
            replay, timing = [], None
            passed = all(f["passed"] for f in fixtures)
        else:
            fixtures, replay, timing = trials(torch, methods, seed, torch.device("cuda:0"), rows=rows)
            passed = timing is not None
    return {
        "rows": rows, "width": run.WIDTH, "seed": seed, "code_revision": revision, "source_sha256": sources,
        "environment": environment, "asset": baseline["asset"], "codegen": compiler,
        "gate_passed": passed, "fixtures": fixtures, "replay_checks": replay, "timing": timing,
        "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--rows", type=int, choices=ROWS, required=True)
    cli.add_argument("--seed", type=int, choices=(QUALIFICATION, *TIMING_SEEDS), required=True)
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--qualified", type=Path)
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    baseline = json.loads(args.baseline.read_text())
    earlier = json.loads(args.qualified.read_text()) if args.qualified else None
    print(json.dumps(checkpoint(args.rows, args.seed, args.asset_dir, baseline, earlier, revision), indent=2, allow_nan=False))
