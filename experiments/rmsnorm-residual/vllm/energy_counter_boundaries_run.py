"""Repeat equal-work H100 energy measurements with short counter boundaries."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import four_warp_energy_run as energy
import run
from sustained_sensors import FIELDS

ROWS, WIDTH = energy.ROWS, energy.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 197, (198, 199, 200)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
BASELINE, CANDIDATE, BACKENDS = energy.BASELINE, energy.CANDIDATE, energy.BACKENDS
MAX_PADDING_S = 0.05
AUXILIARY_SCOPE = "Serialized NVML snapshots before warm-up and after the closing energy query"
ANCHOR = energy.ANCHOR
SOURCES = (*energy.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    Path(__file__).resolve(), run.HERE / "COPY_FREE_ENERGY_COUNTER_BOUNDARIES_512.md",
)))

counter_passed = energy.counter_passed


def auxiliary_snapshot(sensor, origin):
    utc = run.original.utc_now()
    start = time.perf_counter() - origin
    try:
        values, errors = sensor.read()
    except (OSError, AttributeError, RuntimeError, ValueError) as error:
        values, errors = dict.fromkeys(FIELDS), {"read": type(error).__name__}
    return {"utc": utc, "query_start_s": start, "query_end_s": time.perf_counter() - origin,
            "values": values, "errors": errors}


def pair_eligible(blocks):
    return energy.monotonic_counter(blocks) and all(
        block["energy_mj"] is not None and energy.boundary_padding_seconds(block) <= MAX_PADDING_S for block in blocks
    )


def collect_period(torch, verified, sensor, origin, backend, *, bursts, measure_window):
    base = energy.base
    graphs = verified.graphs[backend]
    before = auxiliary_snapshot(sensor, origin)
    warmup = base.collect_window(verified.runner, graphs, energy.WARMUP_S, origin, rows=ROWS)
    work = measure_window(verified.runner, graphs, sensor, origin, bursts)
    after = auxiliary_snapshot(sensor, origin)
    work.update(backend=backend, warmup=warmup, auxiliary_before=before, auxiliary_after=after,
                post_graph_checks=base.compare(torch, verified.works, verified.reference))
    return work


def collect_pair(torch, verified, sensor, origin, order, order_index, *, measure_window, qualifies_pair):
    blocks = [collect_period(torch, verified, sensor, origin, backend, bursts=energy.MEASURED_BURSTS,
                             measure_window=measure_window) for backend in order[order_index]]
    return {"order_index": order_index, "eligible": qualifies_pair(blocks), "blocks": blocks}


def accepted_summary(pairs):
    accepted = [index for index, pair in enumerate(pairs) if pair["eligible"]]
    blocks = [block for index in accepted for block in pairs[index]["blocks"]]
    rates = energy.base.pace(blocks, BACKENDS)[0] if len(accepted) == 2 else dict.fromkeys(BACKENDS)
    return {"accepted_pair_indices": accepted, "median_rows_per_second": rates,
            "median_joules_per_billion_rows": energy.median_energy(blocks),
            "monotonic_counter": energy.monotonic_counter(blocks)}


def measure(torch, seed, verified, sensor, _cuda_uuid, *, measure_window=energy.fixed_window,
            qualifies_pair=pair_eligible):
    base = energy.base
    utc = run.original.utc_now()
    origin = time.perf_counter()
    order = base.balanced_order(seed, BACKENDS)
    def collect(index):
        return collect_pair(torch, verified, sensor, origin, order, index,
                            measure_window=measure_window, qualifies_pair=qualifies_pair)

    pairs = [collect(index) for index in range(len(order))]
    rejected = next((pair["order_index"] for pair in pairs if not pair["eligible"]), None)
    if rejected is not None:
        pairs.append(collect(rejected))
    return {"origin_utc": utc, "order": order, "pairs": pairs, "auxiliary_scope": AUXILIARY_SCOPE,
            **accepted_summary(pairs),
            "inputs_unchanged_after_timing": base.input_fingerprints(verified.works, verified.weight)
            == verified.numerical["source_fingerprints"],
            "outputs_match_after_timing": all(base.outputs_match(block["post_graph_checks"])
                                              for pair in pairs for block in pair["blocks"])}


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
    result = energy.checkpoint(args.seed, args.asset_dir, qualification, revision, numeric_seed=NUMERIC_SEED,
                               timing_seeds=TIMING_SEEDS, source_paths=SOURCES, measure_work=measure)
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))
