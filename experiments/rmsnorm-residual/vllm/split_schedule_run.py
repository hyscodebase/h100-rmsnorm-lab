"""Measure adjacent and staged add-then-normalize graphs with identical kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import copy_free_run as copy_free
import out_of_place_run as direct
import run
import sustained_run as base
from wheel import MEMBER

WORKLOAD = copy_free.ROWS_512
ROWS, WIDTH = WORKLOAD.rows, copy_free.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 169, (170, 171, 172)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
FUSED, ADJACENT, STAGED = WORKLOAD.candidate, copy_free.SPLIT, "triton_two_kernel_staged"
BACKENDS = (FUSED, STAGED, ADJACENT)
OPERATORS = ("add", "normalize")
ANCHOR = run.HERE / "copy-free-512-results/s161.json"
SOURCES = (*WORKLOAD.sources, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "COPY_FREE_SPLIT_SCHEDULE_512.md",
)))


def launch_order(backend):
    if backend == ADJACENT:
        return [(kind, pair) for pair in range(run.OPS) for kind in OPERATORS]
    if backend == STAGED:
        return [(kind, pair) for kind in OPERATORS for pair in range(run.OPS)]
    raise ValueError("Only split graphs have a two-kernel launch order.")


def capture_split(torch, work, weight, backend):
    from eight_warp_outputs import launch_normalization
    from sixteen_warp_kernel import validate_output_buffers
    from two_kernel import launch_residual_add

    x, residual = work.source
    x_out, residual_out = work.buffer
    for pair in range(run.OPS):
        validate_output_buffers(x[pair], residual[pair], weight, x_out[pair], residual_out[pair])
    compiled, launches = {}, []

    def launch_add(pair):
        return launch_residual_add(x[pair], residual[pair], residual_out[pair])

    def normalize(pair):
        return launch_normalization(residual_out[pair], weight, x_out[pair])

    methods = {"add": launch_add, "normalize": normalize}

    def warmup():
        compiled.update((kind, methods[kind](0)) for kind in OPERATORS)

    def record():
        for kind, pair in launch_order(backend):
            if methods[kind](pair) is not compiled[kind]:
                raise ValueError("A captured split launch changed its compiled kernel.")
            launches.append([kind, pair])

    graph = base.capture_graph(torch, work.stream, warmup, record)
    return graph, launches, compiled


def captured_identity(compiled):
    add, norm = (compiled[kind] for kind in OPERATORS)
    add_metadata = base.single.grouped.launch_metadata(add)
    norm_metadata = copy_free.normalization_metadata(norm, WORKLOAD.warps)
    return {
        "split_add": [add_metadata["ptx_instruction_sha256"], hashlib.sha256(add.asm["cubin"]).hexdigest()],
        "split_norm": [norm_metadata[field] for field in copy_free.IDENTITY_FIELDS],
    }


def schedule_gate(checks, fingerprints):
    return set(checks) == {ADJACENT, STAGED} and all(
        len(record["launch_orders"]) == base.STREAMS
        and all(schedule == [list(node) for node in launch_order(name)] for schedule in record["launch_orders"])
        and base.outputs_match(record["streams"]) and record["source_fingerprints"] == fingerprints
        and len(record["alias_guards"]) == 3 and all(record["alias_guards"].values())
        for name, record in checks.items()
    )


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place, split_out_of_place
    from sixteen_warp_kernel import validate_output_buffers

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    timed = seed in TIMING_SEEDS
    if timed and (
        qualification is None or (qualification["seed"], qualification["gate_passed"]) != (NUMERIC_SEED, True)
        or qualification["code_revision"] != revision or qualification["source_sha256"] != sources
        or qualification["environment"] != environment or qualification["asset"] != anchor["asset"]
    ):
        raise ValueError("A passing schedule qualification at the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {ADJACENT: split_out_of_place, FUSED: fused_out_of_place}
        codegen = copy_free.compile_backends(torch, methods, WORKLOAD)
        copy_free.verify_anchor_identity(codegen, anchor["codegen"], WORKLOAD)
        identities = copy_free.binary_identity(anchor["codegen"], candidate=FUSED)
        expected = {name: identities[name] for name in ("split_add", "split_norm")}
        verified = copy_free.verify_input_graphs(torch, seed, device, WORKLOAD, methods, vllm)
        numeric = verified.numerical
        fingerprints = numeric["source_fingerprints"]
        checks = {}
        for backend in (ADJACENT, STAGED):
            captures = [capture_split(torch, work, verified.weight, backend) for work in verified.works]
            evidence = [captured_identity(compiled) for _, _, compiled in captures]
            if any(identity != expected for identity in evidence):
                raise ValueError("A captured kernel differs from the exact 512-row binary.")
            verified.graphs[backend] = [graph for graph, _, _ in captures]
            base.poison_buffers(torch, verified.works)
            verified.runner.burst(verified.graphs[backend], 1)
            checks[backend] = {
                "launch_orders": [launches for _, launches, _ in captures],
                "compiled_kernels_by_stream": evidence,
                "streams": base.compare(torch, verified.works, verified.reference),
                "source_fingerprints": base.input_fingerprints(verified.works, verified.weight),
                "alias_guards": direct.verify_alias_guards(validate_output_buffers, verified.works[0], verified.weight),
            }
        eligible = copy_free.numeric_gate(numeric, WORKLOAD.backends) and schedule_gate(checks, fingerprints)
        timing = None
        if timed and eligible:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, verified.runner, verified.graphs, verified.works, verified.reference, uuid,
                                  backends=BACKENDS, period_s=copy_free.PERIOD_S, rows=ROWS)
            timing["inputs_unchanged_after_timing"] = base.input_fingerprints(verified.works, verified.weight) == fingerprints
            timing["outputs_match_after_timing"] = all(base.outputs_match(block["post_graph_checks"])
                                                       for block in timing["blocks"])
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": anchor["asset"], "codegen": codegen, "numerical": numeric,
            "schedule_checks": checks, "gate_passed": eligible, "timing": timing,
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
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    row = checkpoint(args.seed, args.asset_dir, qualification, revision)
    print(json.dumps(row, separators=(",", ":"), allow_nan=False))
