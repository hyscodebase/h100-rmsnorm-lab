"""Compare cumulative H100 energy for equal work by two exact RMSNorm kernels."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import copy_free_run as copy_free
import four_physical_warps_run as four
import run
import sustained_run as base
from sustained_sensors import Nvml, Observer, associate, summarize_load
from wheel import MEMBER

ROWS, WIDTH = four.ROWS, four.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 193, (194, 195, 196)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
BASELINE, CANDIDATE = four.BASELINE, four.CANDIDATE
BACKENDS = four.BACKENDS
WARMUP_S, PROBE_BURSTS, MEASURED_BURSTS = 3.0, 32, 512
ENERGY_RATIO, THROUGHPUT_RATIO = .985, .995
COUNTER_FUNCTION = "nvmlDeviceGetTotalEnergyConsumption"
ANCHOR = run.HERE / "four-physical-warps-512-results/s189.json"
SOURCES = (*four.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "COPY_FREE_FOUR_WARP_ENERGY_512.md",
)))


def energy_reading(sensor, origin):
    query_start = time.perf_counter() - origin
    value = C.c_ulonglong()
    try:
        query = sensor.function(COUNTER_FUNCTION, C.c_void_p, C.POINTER(C.c_ulonglong))
    except AttributeError:
        status = "missing_symbol"
    else:
        status = int(query(sensor.handle, C.byref(value)))
    return {"query_start_s": query_start, "query_end_s": time.perf_counter() - origin,
            "status": status, "millijoules": value.value if status == 0 else None}


def energy_delta(before, after):
    if (before["status"] == 0 and after["status"] == 0
            and after["millijoules"] > before["millijoules"]):
        return after["millijoules"] - before["millijoules"]
    return None


def fixed_window(runner, graphs, sensor, origin, bursts, *, before=None):
    if before is None:
        before = energy_reading(sensor, origin)
    work = base.collect_window(runner, graphs, 0.0, origin, rows=ROWS, fixed_bursts=bursts)
    after = energy_reading(sensor, origin)
    delta = energy_delta(before, after)
    work.update(energy_start=before, energy_end=after, energy_mj=delta,
                joules_per_billion_rows=None if delta is None else delta * 1_000_000 / work["normalized_rows"])
    return work


def probe_counter(sensor, runner, graphs):
    utc = run.original.utc_now()
    origin = time.perf_counter()
    first = energy_reading(sensor, origin)
    window = (fixed_window(runner, graphs, sensor, origin, PROBE_BURSTS, before=first)
              if first["status"] == 0 else None)
    return {"observer_origin_utc": utc, "first": first, "window": window}


def counter_passed(probe):
    return probe is not None and probe["window"] is not None and probe["window"]["energy_mj"] is not None


def median_energy(blocks):
    result = {}
    for backend in BACKENDS:
        values = [block["joules_per_billion_rows"] for block in blocks if block["backend"] == backend]
        result[backend] = statistics.median(values) if len(values) == 2 and None not in values else None
    return result


def monotonic_counter(blocks):
    endpoints = [block[field] for block in blocks for field in ("energy_start", "energy_end")]
    if not endpoints or any(endpoint["status"] != 0 for endpoint in endpoints):
        return False
    values = [endpoint["millijoules"] for endpoint in endpoints]
    return all(before <= after for before, after in zip(values, values[1:]))


def measure(torch, seed, verified, sensor, cuda_uuid):
    utc = run.original.utc_now()
    origin = time.perf_counter()
    observer = Observer(cuda_uuid, origin)
    order, blocks = base.balanced_order(seed, BACKENDS), []
    try:
        for round_index, group in enumerate(order):
            for position, backend in enumerate(group):
                graphs = verified.graphs[backend]
                warm = base.collect_window(verified.runner, graphs, WARMUP_S, origin, rows=ROWS)
                block = fixed_window(verified.runner, graphs, sensor, origin, MEASURED_BURSTS)
                block.update(id=len(blocks), round=round_index, position=position, backend=backend, warmup=warm,
                             post_graph_checks=base.compare(torch, verified.works, verified.reference))
                blocks.append(block)
    finally:
        telemetry = observer.finish()
    associate(telemetry["slots"], blocks)
    telemetry["by_backend"] = summarize_load(blocks, telemetry["slots"], BACKENDS)
    rates, gains = base.pace(blocks, BACKENDS)
    return {"observer_origin_utc": utc, "order": order, "blocks": blocks, "telemetry": telemetry,
            "median_rows_per_second": rates, "comparisons": gains,
            "median_joules_per_billion_rows": median_energy(blocks),
            "monotonic_counter": monotonic_counter(blocks),
            "inputs_unchanged_after_timing": base.input_fingerprints(verified.works, verified.weight)
            == verified.numerical["source_fingerprints"],
            "outputs_match_after_timing": all(base.outputs_match(b["post_graph_checks"]) for b in blocks)}


def comparison(timing):
    if (timing is None or not timing["inputs_unchanged_after_timing"] or not timing["outputs_match_after_timing"]
            or not timing["monotonic_counter"]):
        return None
    energy = timing["median_joules_per_billion_rows"]
    if None in energy.values():
        return None
    rates = timing["median_rows_per_second"]
    return {"energy_saving": 1 - energy[CANDIDATE] / energy[BASELINE],
            "throughput_change": rates[CANDIDATE] / rates[BASELINE] - 1,
            "passes_energy": energy[CANDIDATE] <= ENERGY_RATIO * energy[BASELINE],
            "passes_throughput": rates[CANDIDATE] >= THROUGHPUT_RATIO * rates[BASELINE]}


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
                  or not counter_passed(qualification["counter_probe"])
                  or not copy_free.numeric_gate(qualification["numerical"], four.VERIFICATION_BACKENDS)):
        raise ValueError("A passing seed-193 numerical and energy checkpoint from this revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        vllm = base.load_vllm(torch, library)
        methods = {BASELINE: baseline, CANDIDATE: candidate}
        codegen = four.compile_backends(torch, methods)
        four.verify_codegen(codegen, anchor)
        if four.identities(codegen) != four.identities(anchor["codegen"]):
            raise ValueError("The candidate or baseline differs from the pinned four-warp comparison.")
        methods = {name: four.pin_launcher(method, codegen[name]["binary_sha256"]) for name, method in methods.items()}
        verified = copy_free.verify_input_graphs(torch, seed, device, copy_free.ROWS_512, methods, vllm)
        numerical = verified.numerical
        exact = copy_free.numeric_gate(numerical, four.VERIFICATION_BACKENDS)
        cuda_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        counter_error, probe, timing = None, None, None
        try:
            sensor = Nvml(cuda_uuid)
        except (OSError, AttributeError, RuntimeError, ValueError) as error:
            counter_error = type(error).__name__
        else:
            try:
                probe = probe_counter(sensor, verified.runner, verified.graphs[BASELINE])
                if exact and counter_passed(probe) and timed:
                    timing = measure(torch, seed, verified, sensor, cuda_uuid)
            finally:
                sensor.close()
    return {**provenance, "seed": seed, "codegen": codegen, "numerical": numerical, "numerically_exact": exact,
            "gate_passed": exact and counter_passed(probe), "counter_function": COUNTER_FUNCTION,
            "counter_error": counter_error, "counter_probe": probe, "timing": timing, "effect": comparison(timing),
            "graph_contract": copy_free.graph_contract(ROWS, backends=four.VERIFICATION_BACKENDS),
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
