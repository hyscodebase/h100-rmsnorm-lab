"""Select a 512-row exact split-add tile, then measure one held-out seed."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bounded_timing import bitwise_case
import copy_free_run as copy_free
import run
import sustained_run as base
from wheel import MEMBER

WORKLOAD = copy_free.ROWS_512
ROWS, WIDTH = WORKLOAD.rows, copy_free.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 173, (174, 175, 176)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
BASE_TILE, ADD_WARPS = 2048, 4
TILES = (1024, BASE_TILE, 4096, 8192)
FUSED, ORIGINAL = WORKLOAD.candidate, copy_free.SPLIT
TILE_BACKENDS = {tile: ORIGINAL if tile == BASE_TILE else f"triton_two_kernel_add_{tile}" for tile in TILES}
TUNING_PERIOD_S, TUNING_ROUNDS = 4., 4
ANCHOR = run.HERE / "copy-free-512-results/s161.json"
SOURCES = (*WORKLOAD.sources, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "COPY_FREE_SPLIT_ADD_TILES_512.md",
)))


def held_out_backends(tile):
    return tuple(dict.fromkeys((ORIGINAL, TILE_BACKENDS[tile], FUSED)))


def choose_tile(rates):
    candidates = [tile for tile, backend in TILE_BACKENDS.items() if backend in rates]
    return min(candidates, key=lambda tile: (-rates[TILE_BACKENDS[tile]], tile != BASE_TILE, tile))


def exact_timing(timing):
    return timing is not None and timing["inputs_unchanged_after_timing"] and timing["outputs_match_after_timing"]


def exact_graph(record, fingerprints):
    return (record["weight_unchanged"] and record["finite_activations"] and base.outputs_match(record["streams"])
            and record["source_fingerprints"] == fingerprints)


def eligible_backends(numerical, additions, backends):
    fingerprints = numerical["source_fingerprints"]
    checks, fixtures, guards = (numerical[key] for key in ("graph_checks", "fixtures", "alias_guards"))
    reference_ok = numerical["inputs_unchanged"] and exact_graph(checks[copy_free.REFERENCE], fingerprints)
    qualified = {copy_free.REFERENCE: reference_ok}
    for name in backends:
        code_ok = name == FUSED or copy_free.vectorized_add(additions[name]["residual_add"], ADD_WARPS)
        qualified[name] = (
            reference_ok and code_ok and exact_graph(checks[name], fingerprints)
            and len(guards[name]) == 3 and all(guards[name].values())
            and len(fixtures) == 8 and all(
                name in case.get("comparisons", {}) and bitwise_case(case["comparisons"][name]) for case in fixtures
            )
        )
    return qualified


def compile_additions(torch, methods, codegen):
    x = torch.zeros((ROWS, WIDTH), dtype=torch.bfloat16, device="cuda:0")
    residual, weight = torch.zeros_like(x), torch.ones_like(x[0])
    x_out, residual_out = torch.empty_like(x), torch.empty_like(residual)
    expected_norm = [codegen[ORIGINAL]["normalization"][key] for key in copy_free.IDENTITY_FIELDS]
    additions = {}
    for tile, backend in TILE_BACKENDS.items():
        if backend not in methods:
            continue
        add, norm = methods[backend](x, residual, weight, x_out, residual_out)
        code = copy_free.addition_metadata(add, tile)
        norm_code = copy_free.normalization_metadata(norm, WORKLOAD.warps)
        norm_identity = [norm_code[key] for key in copy_free.IDENTITY_FIELDS]
        if norm_identity != expected_norm:
            raise ValueError("A split tile changed the pinned eight-warp normalization.")
        if backend == ORIGINAL and any(code[key] != codegen[ORIGINAL]["residual_add"][key]
                                       for key in copy_free.IDENTITY_FIELDS):
            raise ValueError("The original 2,048-element addition changed.")
        additions[backend] = {"residual_add": code, "normalization_identity": norm_identity}
    return additions


def qualified_tile(qualification, sources, revision, environment, asset):
    if (qualification is None or (qualification["seed"], qualification["gate_passed"]) != (NUMERIC_SEED, True)
            or (qualification["rows"], qualification["width"]) != (ROWS, WIDTH)
            or qualification["code_revision"] != revision or qualification["source_sha256"] != sources
            or qualification["environment"] != environment or qualification["asset"] != asset
            or not exact_timing(qualification["timing"])):
        raise ValueError("A passing seed-173 selection from the exact revision is required.")
    timing = qualification["timing"]
    rates = timing["median_rows_per_second"]
    if (set(rates) != {name for name in TILE_BACKENDS.values() if qualification["eligibility"][name]}
            or choose_tile(rates) != qualification["selected_tile"]):
        raise ValueError("The selected tile does not match the qualified tuning rates.")
    return qualification["selected_tile"]


def held_out_comparison(timing, tile):
    rates = timing["median_rows_per_second"]
    tuned = TILE_BACKENDS[tile]
    upgrade = rates[tuned] / rates[ORIGINAL] - 1
    stronger = max(dict.fromkeys((ORIGINAL, tuned)), key=rates.__getitem__)
    return {
        "selected_tile_gain_over_original": upgrade,
        "selected_tile_passes_5pct": tile != BASE_TILE and rates[tuned] >= 1.05 * rates[ORIGINAL],
        "stronger_split_backend": stronger,
        "fused_gain_over_stronger_split": rates[FUSED] / rates[stronger] - 1,
    }


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place, split_out_of_place

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    asset = anchor["asset"]
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    tuning = seed == NUMERIC_SEED
    tile = None if tuning else qualified_tile(qualification, sources, revision, environment, asset)
    requested = tuple(TILE_BACKENDS.values()) if tuning else held_out_backends(tile)
    splits = {name: partial(split_out_of_place, tile=size) for size, name in TILE_BACKENDS.items() if name in requested}
    launchers = {**splits, FUSED: fused_out_of_place}
    methods = {name: launchers[name] for name in requested}
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        codegen = copy_free.compile_backends(torch, launchers, WORKLOAD)
        copy_free.verify_anchor_identity(codegen, anchor["codegen"], WORKLOAD)
        additions = compile_additions(torch, methods, codegen)
        if not tuning and any(
            any(additions[name]["residual_add"][key] != qualification["addition_codegen"][name]["residual_add"][key]
                for key in copy_free.IDENTITY_FIELDS)
            for name in additions
        ):
            raise ValueError("A held-out addition differs from its tuning-seed binary.")
        verified = copy_free.verify_input_graphs(torch, seed, device, WORKLOAD, methods, vllm)
        numerical = verified.numerical
        eligible = eligible_backends(numerical, additions, requested)
        selection_allowed = eligible[ORIGINAL] and (tuning or all(eligible.values()))
        timing = None
        measured = [name for name in requested if eligible[name]]
        if selection_allowed:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = base.measure(torch, seed, verified.runner, verified.graphs, verified.works, verified.reference, uuid,
                                  backends=measured, period_s=TUNING_PERIOD_S if tuning else copy_free.PERIOD_S,
                                  rows=ROWS, rounds=TUNING_ROUNDS if tuning else None)
            timing["inputs_unchanged_after_timing"] = (
                base.input_fingerprints(verified.works, verified.weight) == numerical["source_fingerprints"]
            )
            timing["outputs_match_after_timing"] = all(base.outputs_match(block["post_graph_checks"])
                                                       for block in timing["blocks"])
    exact = exact_timing(timing)
    if tuning and exact:
        tile = choose_tile(timing["median_rows_per_second"])
    comparison = held_out_comparison(timing, tile) if exact and not tuning else None
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": asset, "codegen": codegen, "addition_codegen": additions,
            "numerical": numerical, "eligibility": eligible, "selected_tile": tile, "gate_passed": exact,
            "timing": timing, "held_out_comparison": comparison,
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
        cli.error("Provide the frozen revision.")
    qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
    row = checkpoint(args.seed, args.asset_dir, qualification, revision)
    print(json.dumps(row, separators=(",", ":"), allow_nan=False))
