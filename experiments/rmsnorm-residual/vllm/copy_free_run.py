"""Compare two copy-free implementations of input-preserving residual RMSNorm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import verify_fixtures
import out_of_place_run as direct
import run
import sustained_run as base
from wheel import MEMBER

ROWS, WIDTH, STREAMS = base.ROWS, base.WIDTH, base.STREAMS
NUMERIC_SEEDS, TIMING_SEEDS = (149,), (150, 151, 152)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
REFERENCE, SPLIT, CANDIDATE = base.BACKENDS[0], "triton_two_kernel", direct.CANDIDATE
BACKENDS = (REFERENCE, SPLIT, CANDIDATE)
PERIOD_S = 6.0
ANCHOR = run.HERE / "out-of-place-results/s145.json"
SOURCES = (*direct.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "two_kernel.py", run.HERE / "COPY_FREE_FUSION.md",
)))
IDENTITY_FIELDS = ("ptx_instruction_sha256", "binary_sha256")


def graph_contract():
    return {
        name: {
            "rows": ROWS, "operator_calls_per_stream_graph": run.OPS,
            "kernels_per_operator": 2 if name == SPLIT else 1,
            "restoration_payload_bytes_per_stream_graph": 2 * run.OPS * ROWS * WIDTH * 2 if name == REFERENCE else 0,
        }
        for name in BACKENDS
    }


def binary_identity(codegen):
    return {
        "split_add": [codegen[SPLIT]["residual_add"][key] for key in IDENTITY_FIELDS],
        "split_norm": [codegen[SPLIT]["normalization"][key] for key in IDENTITY_FIELDS],
        "fused": [codegen[CANDIDATE][key] for key in IDENTITY_FIELDS],
    }


def compile_backends(torch, methods):
    from two_kernel import ADD_BLOCK, ADD_WARPS

    x = torch.zeros((ROWS, WIDTH), dtype=torch.bfloat16, device="cuda:0")
    residual, weight = torch.zeros_like(x), torch.ones_like(x[0])
    x_out, residual_out = torch.empty_like(x), torch.empty_like(residual)
    fused = methods[CANDIDATE](x, residual, weight, x_out, residual_out)
    add, norm = methods[SPLIT](x, residual, weight, x_out, residual_out)
    add_code = base.single.grouped.launch_metadata(add)
    opcodes = tuple(add_code["ptx_opcodes"])
    if (add_code["num_warps"] != ADD_WARPS
            or not any(op.startswith("ld.global.v") for op in opcodes)
            or not any(op.startswith("st.global.v") for op in opcodes)):
        raise ValueError("The residual-add baseline must have vectorized global loads and stores.")
    add_code.pop("logical_group_width")
    add_code.update(tile_elements=ADD_BLOCK, binary_sha256=hashlib.sha256(add.asm["cubin"]).hexdigest())
    return {
        CANDIDATE: base.single.warp_metadata(fused),
        SPLIT: {"residual_add": add_code, "normalization": base.single.warp_metadata(norm)},
    }


def numeric_gate(numeric):
    guards = numeric["alias_guards"]
    return (base.numeric_gate(numeric, BACKENDS) and set(guards) == {SPLIT, CANDIDATE}
            and all(len(states) == 3 and all(states.values()) for states in guards.values())
            and all(check["source_fingerprints"] == numeric["source_fingerprints"]
                    for check in numeric["graph_checks"].values()))


def inplace_fixture(fn):
    def call(x, residual, weight):
        fn(x.clone(), residual.clone(), weight, x, residual)

    return call


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from sixteen_warp_kernel import fused_out_of_place
    from two_kernel import split_out_of_place

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    timed = seed in TIMING_SEEDS
    if timed and (
        qualification is None or not qualification["gate_passed"]
        or (qualification["rows"], qualification["width"], qualification["seed"]) != (ROWS, WIDTH, NUMERIC_SEEDS[0])
        or qualification["source_sha256"] != sources or qualification["code_revision"] != revision
        or qualification["environment"] != environment or qualification["asset"] != anchor["asset"]
    ):
        raise ValueError("A passing copy-free preflight from the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {SPLIT: split_out_of_place, CANDIDATE: fused_out_of_place}
        codegen = compile_backends(torch, methods)
        if any(codegen[CANDIDATE][key] != anchor["codegen"][CANDIDATE][key] for key in IDENTITY_FIELDS):
            raise ValueError("The fused kernel differs from the preceding comparison.")
        if timed and binary_identity(codegen) != binary_identity(qualification["codegen"]):
            raise ValueError("A copy-free kernel changed after qualification.")

        fixtures = verify_fixtures(torch, {REFERENCE: vllm, **{name: inplace_fixture(fn) for name, fn in methods.items()}},
                                   seed, device, ROWS)
        works, weight, buffers = base.make_work(torch, seed, device)
        original_weight = weight.view(torch.int16).clone()
        fingerprints = base.input_fingerprints(works, weight)
        guards = {name: direct.verify_alias_guards(fn, works[0], weight) for name, fn in methods.items()}
        runner = base.Replay(torch, works)
        graphs = {REFERENCE: [base.capture(torch, vllm, work, weight) for work in works]}
        graphs.update({name: [direct.capture_candidate(torch, fn, work, weight) for work in works]
                       for name, fn in methods.items()})
        reference, checks = base.verify_graphs(torch, runner, graphs, works, weight, original_weight,
                                               backends=BACKENDS, check_inputs=True)
        numeric = {"fixtures": fixtures, "graph_checks": checks, "buffers": buffers,
                   "source_fingerprints": fingerprints, "alias_guards": guards,
                   "inputs_unchanged": base.input_fingerprints(works, weight) == fingerprints}
        eligible = numeric_gate(numeric)
        timing = None
        if eligible and timed:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, runner, graphs, works, reference, uuid,
                                  backends=BACKENDS, period_s=PERIOD_S)
            timing["inputs_unchanged_after_timing"] = base.input_fingerprints(works, weight) == fingerprints
            timing["outputs_match_after_timing"] = all(base.outputs_match(b["post_graph_checks"]) for b in timing["blocks"])
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": anchor["asset"], "codegen": codegen,
            "numerical": numeric, "graph_contract": graph_contract(), "gate_passed": eligible, "timing": timing,
            "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "started_utc": started,
            "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock}


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
    print(json.dumps(checkpoint(args.seed, args.asset_dir, previous, revision), separators=(",", ":"), allow_nan=False))
