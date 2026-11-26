"""Compare input-preserving residual RMSNorm across four eight-stream backends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import verify_fixtures
import run
import sustained_run as base
from wheel import MEMBER

ROWS, WIDTH, STREAMS = base.ROWS, base.WIDTH, base.STREAMS
NUMERIC_SEEDS, TIMING_SEEDS = (145,), (146, 147, 148)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
CANDIDATE = "triton_16warps_out_of_place"
BACKENDS = (*base.BACKENDS, CANDIDATE)
PERIOD_S = 4.0
ANCHOR = run.HERE / "sustained-results/s137.json"
SOURCES = (*base.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "SUSTAINED_OUT_OF_PLACE.md",
)))


def capture_candidate(torch, fn, work, weight):
    x, residual = work.source
    x_out, residual_out = work.buffer

    def apply(index):
        fn(x[index], residual[index], weight, x_out[index], residual_out[index])

    def record():
        for index in range(run.OPS):
            apply(index)

    return base.capture_graph(torch, work.stream, lambda: apply(0), record)


def verify_alias_guards(fn, work, weight):
    x, residual = (tensor[0] for tensor in work.source)
    x_out, residual_out = (tensor[0] for tensor in work.buffer)
    checks = {}
    for name, first, second in (
        ("activation_output_aliases_input", x, residual_out),
        ("residual_output_aliases_input", x_out, residual),
        ("outputs_alias_each_other", x_out, x_out),
    ):
        try:
            fn(x, residual, weight, first, second)
        except ValueError as error:
            checks[name] = str(error) == "Each output must be disjoint from the inputs and the other output."
        else:
            checks[name] = False
    return checks


def graph_contract():
    payload = 2 * run.OPS * ROWS * WIDTH * 2
    return {
        name: {"rows": ROWS, "operator_calls_per_stream_graph": run.OPS,
               "restoration_payload_bytes_per_stream_graph": 0 if name == CANDIDATE else payload}
        for name in BACKENDS
    }


def numeric_gate(numeric):
    proof = numeric["source_fingerprints"]
    guards = numeric["alias_guards"]
    return (base.numeric_gate(numeric, BACKENDS)
            and len(guards) == 3 and all(guards.values())
            and all(check["source_fingerprints"] == proof for check in numeric["graph_checks"].values()))


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from sixteen_warp_kernel import fused_out_of_place

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
        raise ValueError("A passing input-preserving preflight from the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        methods, codegen = base.prepare_backends(torch, library, anchor)

        def first_launch(x, residual, weight):
            return fused_out_of_place(x, residual, weight, torch.empty_like(x), torch.empty_like(residual))

        codegen[CANDIDATE] = base.single.compile_sixteen(torch, first_launch)
        if timed and any(codegen[CANDIDATE][key] != qualification["codegen"][CANDIDATE][key]
                         for key in ("ptx_instruction_sha256", "binary_sha256")):
            raise ValueError("The directly writing kernel changed after qualification.")

        def fixture_candidate(x, residual, weight):
            fused_out_of_place(x.clone(), residual.clone(), weight, x, residual)

        fixtures = verify_fixtures(torch, {**methods, CANDIDATE: fixture_candidate}, seed, device, ROWS)
        works, weight, buffers = base.make_work(torch, seed, device)
        original_weight = weight.view(torch.int16).clone()
        fingerprints = base.input_fingerprints(works, weight)
        guards = verify_alias_guards(fused_out_of_place, works[0], weight)
        runner = base.Replay(torch, works)
        graphs = {name: [base.capture(torch, fn, work, weight) for work in works] for name, fn in methods.items()}
        graphs[CANDIDATE] = [capture_candidate(torch, fused_out_of_place, work, weight) for work in works]
        reference, checks = base.verify_graphs(torch, runner, graphs, works, weight, original_weight,
                                               backends=BACKENDS, check_inputs=True)
        numeric = {"fixtures": fixtures, "graph_checks": checks, "buffers": buffers,
                   "source_fingerprints": fingerprints,
                   "inputs_unchanged": base.input_fingerprints(works, weight) == fingerprints,
                   "alias_guards": guards}
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
