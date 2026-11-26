"""Qualify and benchmark the four-physical-warp residual RMSNorm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import copy_free_run as copy_free
import run
import sustained_run as base
from wheel import MEMBER

ROWS, WIDTH = copy_free.ROWS_512.rows, copy_free.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 189, (190, 191, 192)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
BASELINE, CANDIDATE = copy_free.ROWS_512.candidate, "triton_4physical_warps_out_of_place"
BACKENDS = (BASELINE, CANDIDATE)
VERIFICATION_BACKENDS = (copy_free.REFERENCE, *BACKENDS)
ANCHOR = run.HERE / "copy-free-512-results/s161.json"
SOURCES = (*copy_free.ROWS_512.sources, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "four_physical_warps.py",
    run.HERE / "COPY_FREE_FOUR_PHYSICAL_WARPS_512.md",
)))


def candidate_metadata(compiled):
    code = base.single.grouped.launch_metadata(compiled)
    ptx = compiled.asm["ptx"]
    threads = re.search(r"\.reqntid\s+(\d+)", ptx)
    lines = (line.strip() for line in ptx.splitlines())
    collective = [line for line in lines if line.startswith((
        "shfl.sync.bfly.b32", "@leader st.shared.f32", "bar.sync", "ld.shared.", "add.rn.f32 total,",
        "setp.ne.u32 mismatch,", "@mismatch mov.b32 total,",
    ))]
    return {**code, "required_threads": int(threads.group(1)) if threads else None,
            "binary_sha256": hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
            "collective_instructions": collective}


def identities(codegen):
    return {name: [codegen[name][key] for key in copy_free.IDENTITY_FIELDS] for name in BACKENDS}


def verify_codegen(codegen, anchor):
    if set(codegen) != set(BACKENDS):
        raise ValueError("A compiled comparator is missing.")
    baseline_id = [anchor["codegen"][BASELINE][key] for key in copy_free.IDENTITY_FIELDS]
    if identities(codegen)[BASELINE] != baseline_id:
        raise ValueError("The eight-warp comparator differs from its frozen 512-row binary.")
    candidate = codegen[CANDIDATE]
    if (candidate["num_warps"], candidate["required_threads"], candidate["fp_fusion"]) != (4, 128, False):
        raise ValueError("The candidate does not use the registered four-warp launch.")
    shuffles = [f"shfl.sync.bfly.b32 peer_{group}, {group}, {distance}, 31, -1;"
                for distance in (1, 2, 4, 8, 16) for group in ("lo", "hi")]
    collective = shuffles + [
        "@leader st.shared.f32 [slot], lo;", "@leader st.shared.f32 [slot+16], hi;", "bar.sync 0;",
        "ld.shared.v4.f32 {w0, w1, w2, w3}, [base];", "ld.shared.v4.f32 {w4, w5, w6, w7}, [base+16];",
        "add.rn.f32 total, w0, w1;",
        *(f"add.rn.f32 total, total, w{i};" for i in range(2, 8)),
    ]
    actual = candidate["collective_instructions"]
    if (actual[:-2] != collective or not actual[-2].startswith("setp.ne.u32 mismatch, thread, ")
            or actual[-1] != "@mismatch mov.b32 total, 0x7fc00000;"):
        raise ValueError("The compiler changed the registered shuffle, shared-memory or lane-guard sequence.")


def compile_backends(torch, methods):
    x = torch.zeros((ROWS, WIDTH), dtype=torch.bfloat16, device="cuda:0")
    residual, weight = torch.zeros_like(x), torch.ones_like(x[0])
    x_out, residual_out = torch.empty_like(x), torch.empty_like(x)
    return {
        BASELINE: copy_free.normalization_metadata(methods[BASELINE](x, residual, weight, x_out, residual_out), 8),
        CANDIDATE: candidate_metadata(methods[CANDIDATE](x, residual, weight, x_out, residual_out)),
    }


def pin_launcher(method, expected_binary):
    seen = {}

    def launch(*args):
        kernel = method(*args)
        if id(kernel) not in seen:
            if hashlib.sha256(kernel.asm["cubin"]).hexdigest() != expected_binary:
                raise ValueError("A captured kernel differs from the audited compilation.")
            seen[id(kernel)] = kernel
        return kernel

    return launch


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place as baseline
    from four_physical_warps import fused_out_of_place as candidate

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    provenance = {"rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
                  "environment": environment, "asset": anchor["asset"]}
    timed = seed in TIMING_SEEDS
    if timed and (not qualification or qualification["seed"] != NUMERIC_SEED or not qualification["gate_passed"]
                  or any(qualification[key] != value for key, value in provenance.items())
                  or not copy_free.numeric_gate(qualification["numerical"], VERIFICATION_BACKENDS)):
        raise ValueError("A passing seed-189 checkpoint from the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {BASELINE: baseline, CANDIDATE: candidate}
        codegen = compile_backends(torch, methods)
        verify_codegen(codegen, anchor)
        if timed and identities(codegen) != identities(qualification["codegen"]):
            raise ValueError("A kernel's compilation differs from the numerical checkpoint.")
        methods = {name: pin_launcher(method, codegen[name]["binary_sha256"]) for name, method in methods.items()}
        verified = copy_free.verify_input_graphs(torch, seed, device, copy_free.ROWS_512, methods, vllm)
        numerical = verified.numerical
        eligible = copy_free.numeric_gate(numerical, VERIFICATION_BACKENDS)
        timing = None
        if eligible and timed:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, verified.runner, verified.graphs, verified.works, verified.reference, uuid,
                                  backends=BACKENDS, period_s=copy_free.PERIOD_S, rows=ROWS)
            timing["inputs_unchanged_after_timing"] = (
                base.input_fingerprints(verified.works, verified.weight) == numerical["source_fingerprints"]
            )
            timing["outputs_match_after_timing"] = all(base.outputs_match(b["post_graph_checks"]) for b in timing["blocks"])
    return {**provenance, "seed": seed, "codegen": codegen, "numerical": numerical, "gate_passed": eligible,
            "graph_contract": copy_free.graph_contract(ROWS, backends=VERIFICATION_BACKENDS), "timing": timing,
            "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
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
        cli.error("Provide a frozen Git revision.")
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    result = checkpoint(args.seed, args.asset_dir, qualification, revision)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
