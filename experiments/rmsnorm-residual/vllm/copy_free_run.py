"""Compare two copy-free implementations of input-preserving residual RMSNorm."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import verify_fixtures
import out_of_place_run as direct
import run
import shuffle_run
import sustained_run as base
from wheel import MEMBER

ROWS, WIDTH, STREAMS = base.ROWS, base.WIDTH, base.STREAMS
NUMERIC_SEEDS, TIMING_SEEDS = (149,), (150, 151, 152)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
REFERENCE, SPLIT, CANDIDATE = base.BACKENDS[0], "triton_two_kernel", direct.CANDIDATE
BOUNDARY_CANDIDATE = "triton_8warps_out_of_place"
BACKENDS = (REFERENCE, SPLIT, CANDIDATE)
PERIOD_S = 6.0
ANCHOR = run.HERE / "out-of-place-results/s145.json"
SOURCES = (*direct.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "two_kernel.py", run.HERE / "COPY_FREE_FUSION.md",
)))
IDENTITY_FIELDS = ("ptx_instruction_sha256", "binary_sha256")


@dataclass(frozen=True)
class Workload:
    rows: int
    numeric_seed: int
    timing_seeds: tuple[int, ...]
    anchor: Path
    sources: tuple[str, ...]
    candidate: str = CANDIDATE
    warps: int = 16

    @property
    def seeds(self):
        return (self.numeric_seed, *self.timing_seeds)

    @property
    def backends(self):
        return (REFERENCE, SPLIT, self.candidate)


DEFAULT = Workload(ROWS, NUMERIC_SEEDS[0], TIMING_SEEDS, ANCHOR, SOURCES)
TRANSFER_ANCHOR = run.HERE / "copy-free-results/s149.json"
TRANSFER = Workload(
    255, 153, (154, 155, 156), TRANSFER_ANCHOR,
    (*SOURCES, *(str(path.relative_to(run.ROOT)) for path in (TRANSFER_ANCHOR, run.HERE / "COPY_FREE_ROWS_255.md"))),
)
BOUNDARY_ANCHOR = run.HERE / "copy-free-255-results/s153.json"
BOUNDARY = Workload(
    256, 157, (158, 159, 160), BOUNDARY_ANCHOR,
    (*TRANSFER.sources, *(str(path.relative_to(run.ROOT)) for path in (
        BOUNDARY_ANCHOR, run.HERE / "eight_warp_outputs.py", run.HERE / "COPY_FREE_ROWS_256.md",
    ))),
    candidate=BOUNDARY_CANDIDATE, warps=8,
)
ROWS_512_ANCHOR = run.HERE / "copy-free-256-results/s157.json"
ROWS_512 = Workload(
    512, 161, (162, 163, 164), ROWS_512_ANCHOR,
    (*BOUNDARY.sources, *(str(path.relative_to(run.ROOT)) for path in (
        ROWS_512_ANCHOR, run.HERE / "COPY_FREE_ROWS_512.md",
    ))),
    candidate=BOUNDARY.candidate, warps=BOUNDARY.warps,
)
WORKLOADS = {workload.rows: workload for workload in (DEFAULT, TRANSFER, BOUNDARY, ROWS_512)}


def graph_contract(rows=ROWS, *, backends=BACKENDS):
    return {
        name: {
            "rows": rows, "operator_calls_per_stream_graph": run.OPS,
            "kernels_per_operator": 2 if name == SPLIT else 1,
            "restoration_payload_bytes_per_stream_graph": 2 * run.OPS * rows * WIDTH * 2 if name == REFERENCE else 0,
        }
        for name in backends
    }


def binary_identity(codegen, *, candidate=CANDIDATE):
    return {
        "split_add": [codegen[SPLIT]["residual_add"][key] for key in IDENTITY_FIELDS],
        "split_norm": [codegen[SPLIT]["normalization"][key] for key in IDENTITY_FIELDS],
        "fused": [codegen[candidate][key] for key in IDENTITY_FIELDS],
    }


def verify_anchor_identity(codegen, anchor, workload):
    if workload == BOUNDARY:
        actual, previous = (data[SPLIT]["residual_add"] for data in (codegen, anchor))
        if any(actual[key] != previous[key] for key in IDENTITY_FIELDS):
            raise ValueError("The residual-add binary changed at the dispatch boundary.")
        return
    if workload == DEFAULT:
        if any(codegen[CANDIDATE][key] != anchor[CANDIDATE][key] for key in IDENTITY_FIELDS):
            raise ValueError("The fused kernel differs from the preceding comparison.")
        return
    if binary_identity(codegen, candidate=workload.candidate) != binary_identity(anchor, candidate=workload.candidate):
        raise ValueError("The split or fused kernel changed at the new row count.")


def normalization_metadata(compiled, warps):
    if warps == 16:
        return base.single.warp_metadata(compiled)
    code = shuffle_run.warp_metadata(compiled)
    threads = re.search(r"\.reqntid\s+(\d+)", compiled.asm["ptx"])
    opcodes = code["ptx_opcodes"]
    if (not threads or int(threads.group(1)) != 256 or opcodes.get("bar.sync") != 1
            or opcodes.get("ld.shared.v4.f32") != 2
            or not opcodes.get("ld.global.v4.b32") or not opcodes.get("st.global.v4.b32")):
        raise ValueError("Unexpected memory layout or collective in the 8-warp normalization.")
    code["required_threads"] = 256
    return code


def compile_backends(torch, methods, workload):
    from two_kernel import ADD_BLOCK, ADD_WARPS

    x = torch.zeros((workload.rows, WIDTH), dtype=torch.bfloat16, device="cuda:0")
    residual, weight = torch.zeros_like(x), torch.ones_like(x[0])
    x_out, residual_out = torch.empty_like(x), torch.empty_like(residual)
    fused = methods[workload.candidate](x, residual, weight, x_out, residual_out)
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
        workload.candidate: normalization_metadata(fused, workload.warps),
        SPLIT: {"residual_add": add_code, "normalization": normalization_metadata(norm, workload.warps)},
    }


def numeric_gate(numeric, backends=BACKENDS):
    guards = numeric["alias_guards"]
    return (base.numeric_gate(numeric, backends) and set(guards) == set(backends[1:])
            and all(len(states) == 3 and all(states.values()) for states in guards.values())
            and all(check["source_fingerprints"] == numeric["source_fingerprints"]
                    for check in numeric["graph_checks"].values()))


def inplace_fixture(fn):
    def call(x, residual, weight):
        fn(x.clone(), residual.clone(), weight, x, residual)

    return call


def diagnose_boundary(torch, vllm, seed, device, works, weight, runner, reference, anchor):
    from sixteen_warp_kernel import fused_out_of_place

    first = works[0]
    x, residual = (tensor[0] for tensor in first.source)
    x_out, residual_out = (tensor[0] for tensor in first.buffer)
    code = base.single.warp_metadata(fused_out_of_place(x, residual, weight, x_out, residual_out))
    if any(code[key] != anchor[key] for key in IDENTITY_FIELDS):
        raise ValueError("The diagnostic kernel differs from its 255-row binary.")
    fixtures = verify_fixtures(torch, {REFERENCE: vllm, CANDIDATE: inplace_fixture(fused_out_of_place)},
                               seed, device, x.shape[0])
    graphs = [direct.capture_candidate(torch, fused_out_of_place, work, weight) for work in works]
    base.poison_buffers(torch, works)
    runner.burst(graphs, 1)
    return {"backend": CANDIDATE, "codegen": code, "fixtures": fixtures,
            "streams": base.compare(torch, works, reference),
            "source_fingerprints": base.input_fingerprints(works, weight)}


def checkpoint(seed, asset_dir, qualification, revision, *, workload=DEFAULT):
    import torch
    import triton

    if workload.warps == 8:
        from eight_warp_outputs import fused_out_of_place, split_out_of_place
    else:
        from sixteen_warp_kernel import fused_out_of_place
        from two_kernel import split_out_of_place

    started, clock = run.original.utc_now(), time.perf_counter()
    rows = workload.rows
    backends = workload.backends
    anchor = json.loads(workload.anchor.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in workload.sources}
    timed = seed in workload.timing_seeds
    if timed and (
        qualification is None or not qualification["gate_passed"]
        or (qualification["rows"], qualification["width"], qualification["seed"]) != (rows, WIDTH, workload.numeric_seed)
        or qualification["source_sha256"] != sources or qualification["code_revision"] != revision
        or qualification["environment"] != environment or qualification["asset"] != anchor["asset"]
    ):
        raise ValueError("A passing copy-free preflight from the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {SPLIT: split_out_of_place, workload.candidate: fused_out_of_place}
        codegen = compile_backends(torch, methods, workload)
        verify_anchor_identity(codegen, anchor["codegen"], workload)
        if timed and binary_identity(codegen, candidate=workload.candidate) != binary_identity(
            qualification["codegen"], candidate=workload.candidate,
        ):
            raise ValueError("A copy-free kernel changed after qualification.")

        fixtures = verify_fixtures(torch, {REFERENCE: vllm, **{name: inplace_fixture(fn) for name, fn in methods.items()}},
                                   seed, device, rows)
        works, weight, buffers = base.make_work(torch, seed, device, rows=rows)
        original_weight = weight.view(torch.int16).clone()
        fingerprints = base.input_fingerprints(works, weight)
        guards = {name: direct.verify_alias_guards(fn, works[0], weight) for name, fn in methods.items()}
        runner = base.Replay(torch, works)
        graphs = {REFERENCE: [base.capture(torch, vllm, work, weight) for work in works]}
        graphs.update({name: [direct.capture_candidate(torch, fn, work, weight) for work in works]
                       for name, fn in methods.items()})
        reference, checks = base.verify_graphs(torch, runner, graphs, works, weight, original_weight,
                                               backends=backends, check_inputs=True)
        numeric = {"fixtures": fixtures, "graph_checks": checks, "buffers": buffers,
                   "source_fingerprints": fingerprints, "alias_guards": guards,
                   "inputs_unchanged": base.input_fingerprints(works, weight) == fingerprints}
        if workload == BOUNDARY and seed == workload.numeric_seed:
            numeric["prior_16warps"] = diagnose_boundary(
                torch, vllm, seed, device, works, weight, runner, reference, anchor["codegen"][CANDIDATE],
            )
            if numeric["prior_16warps"]["source_fingerprints"] != fingerprints:
                raise ValueError("The diagnostic changed an original source buffer.")
        eligible = numeric_gate(numeric, backends)
        timing = None
        if eligible and timed:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, runner, graphs, works, reference, uuid,
                                  backends=backends, period_s=PERIOD_S, rows=rows)
            timing["inputs_unchanged_after_timing"] = base.input_fingerprints(works, weight) == fingerprints
            timing["outputs_match_after_timing"] = all(base.outputs_match(b["post_graph_checks"]) for b in timing["blocks"])
    return {"seed": seed, "rows": rows, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": anchor["asset"], "codegen": codegen,
            "numerical": numeric, "graph_contract": graph_contract(rows, backends=backends),
            "gate_passed": eligible, "timing": timing,
            "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "started_utc": started,
            "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--rows", type=int, choices=WORKLOADS, default=ROWS)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--asset-dir", required=True, type=Path)
    cli.add_argument("--qualified", type=Path)
    args = cli.parse_args()
    workload = WORKLOADS[args.rows]
    if args.seed not in workload.seeds:
        cli.error(f"Choose a seed from {workload.seeds} for {args.rows} rows.")
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    previous = json.loads(args.qualified.read_bytes()) if args.qualified else None
    result = checkpoint(args.seed, args.asset_dir, previous, revision, workload=workload)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
