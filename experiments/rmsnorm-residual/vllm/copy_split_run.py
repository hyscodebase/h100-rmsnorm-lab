"""Separate device copies and residual RMSNorm for one eight-stream checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from bounded_timing import verify_fixtures
import run
import sustained_run as base
from sustained_sensors import Observer, associate, summarize_load
from wheel import MEMBER

ROWS, WIDTH, STREAMS = base.ROWS, base.WIDTH, base.STREAMS
BACKENDS = base.BACKENDS
MODES = ("combined", "split")
CONDITIONS = tuple(f"{mode}:{name}" for mode in MODES for name in BACKENDS)
NUMERIC_SEEDS, TIMING_SEEDS = (141,), (142, 143, 144)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
WARMUP_S, PERIOD_S = 1.0, 4.0
PAIRS = STREAMS * run.OPS
SAMPLE_COLUMNS = ("start_s", "end_s", "cuda_ms", "restore_ms", "operator_ms")
ANCHOR = run.HERE / "sustained-results/s137.json"
SOURCES = (*base.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "SUSTAINED_COPY_SPLIT.md",
)))


class StagedReplay(base.Replay):
    def __init__(self, torch, works):
        super().__init__(torch, works)
        self.middle = torch.cuda.Event(enable_timing=True)
        self.copied = [torch.cuda.Event() for _ in works]

    def split(self, copies, operators):
        self.begin.record(self.control)
        self.enqueue(copies, self.begin, self.copied, 1)
        self.middle.record(self.control)
        self.enqueue(operators, self.middle, self.done, 1)
        self.end.record(self.control)
        self.end.synchronize()
        return (self.begin.elapsed_time(self.end), self.begin.elapsed_time(self.middle),
                self.middle.elapsed_time(self.end))


def verify_stages(torch, runner, copies, operators, combined, works, weight, original_weight):
    reference, joined = base.verify_graphs(torch, runner, combined, works, weight, original_weight)
    base.poison_buffers(torch, works)
    runner.burst(copies, 1)
    restored = base.compare(torch, works, [work.source for work in works])
    staged = {}
    for name in BACKENDS:
        base.poison_buffers(torch, works)
        runner.split(copies, operators[name])
        staged[name] = {"streams": base.compare(torch, works, reference),
                        "weight_unchanged": bool(torch.equal(weight.view(torch.int16), original_weight))}
    return reference, joined, restored, staged


def numeric_gate(data):
    return (base.numeric_gate(data)
            and base.outputs_match(data["restored_before_operator"])
            and set(data["split_checks"]) == set(BACKENDS)
            and all(item["weight_unchanged"] and base.outputs_match(item["streams"])
                    for item in data["split_checks"].values()))


def balanced_order(seed):
    return [
        [[mode, backend] for backend in backends
         for mode in (MODES if (seed + round_index + BACKENDS.index(backend)) % 2 == 0 else MODES[::-1])]
        for round_index, backends in enumerate(base.balanced_order(seed))
    ]


def window(replay, seconds, origin, mode):
    utc = run.original.utc_now()
    start = time.perf_counter() - origin
    samples = []
    while not samples or samples[-1][1] - start < seconds:
        begin = time.perf_counter() - origin
        device, restore, operator = replay()
        end = time.perf_counter() - origin
        samples.append((begin, end, device, restore, operator))
    end = samples[-1][1]
    graph_launches = len(samples) * STREAMS * (2 if mode == "split" else 1)
    return {"start_s": start, "end_s": end, "wall_seconds": end - start,
            "started_utc": utc, "finished_utc": run.original.utc_now(), "samples": samples,
            "attempted_graphs": graph_launches, "completed_graphs": graph_launches,
            "completed_pairs": len(samples) * PAIRS, "normalized_rows": len(samples) * PAIRS * ROWS,
            "cuda_ms": sum(s[2] for s in samples),
            "restore_ms": sum(s[3] for s in samples) if mode == "split" else None,
            "operator_ms": sum(s[4] for s in samples) if mode == "split" else None}


def summarize(blocks):
    rates = {}
    for group in CONDITIONS:
        current = [block for block in blocks if block["condition"] == group]
        summary = {
            "median_wall_rows_per_second": statistics.median(b["normalized_rows"] / b["wall_seconds"] for b in current),
            "median_device_rows_per_second": statistics.median(1000 * b["normalized_rows"] / b["cuda_ms"] for b in current),
            "median_operator_rows_per_second": None, "restoration_fraction": None,
        }
        if current[0]["mode"] == "split":
            summary["median_operator_rows_per_second"] = statistics.median(
                1000 * b["normalized_rows"] / b["operator_ms"] for b in current
            )
            summary["restoration_fraction"] = sum(b["restore_ms"] for b in current) / sum(b["cuda_ms"] for b in current)
        rates[group] = summary
    comparisons = {}
    for label, mode, key in (
        ("operator", "split", "median_operator_rows_per_second"),
        ("combined_wall", "combined", "median_wall_rows_per_second"),
        ("split_wall", "split", "median_wall_rows_per_second"),
    ):
        candidate = rates[f"{mode}:{BACKENDS[-1]}"][key]
        comparisons[label] = {}
        for baseline in BACKENDS[:-1]:
            ref = rates[f"{mode}:{baseline}"][key]
            comparisons[label][baseline] = {"throughput_gain": candidate / ref - 1, "passes_5pct": candidate >= ref * 1.05}
    return rates, comparisons


def measure(torch, seed, runner, copies, operators, combined, works, reference, cuda_uuid):
    origin_utc = run.original.utc_now()
    origin = time.perf_counter()
    observer = Observer(cuda_uuid, origin)
    order, blocks = balanced_order(seed), []
    try:
        for round_index, group in enumerate(order):
            for position, (mode, backend) in enumerate(group):
                if mode == "split":
                    def replay():
                        return runner.split(copies, operators[backend])
                else:
                    def replay():
                        return runner.burst(combined[backend], 1), None, None
                warmup = window(replay, WARMUP_S, origin, mode)
                block = window(replay, PERIOD_S, origin, mode)
                block.update(id=len(blocks), round=round_index, position=position, backend=backend, mode=mode,
                             condition=f"{mode}:{backend}", warmup=warmup,
                             post_graph_checks=base.compare(torch, works, reference))
                blocks.append(block)
    finally:
        telemetry = observer.finish()
    associate(telemetry["slots"], blocks)
    telemetry["by_condition"] = summarize_load(blocks, telemetry["slots"], CONDITIONS, group_by="condition")
    rates, comparisons = summarize(blocks)
    return {"observer_origin_utc": origin_utc, "sample_columns": SAMPLE_COLUMNS, "order": order,
            "blocks": blocks, "telemetry": telemetry, "rates": rates, "comparisons": comparisons}


def checkpoint(seed, asset_dir, previous, revision):
    import torch
    import triton

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    should_measure = seed in TIMING_SEEDS
    if should_measure and (
        previous is None or not previous["gate_passed"]
        or (previous["rows"], previous["width"], previous["seed"]) != (ROWS, WIDTH, NUMERIC_SEEDS[0])
        or previous["source_sha256"] != sources or previous["code_revision"] != revision
        or previous["environment"] != environment or previous["asset"] != anchor["asset"]
    ):
        raise ValueError("A passing split-schedule preflight from the frozen revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        methods, codegen = base.prepare_backends(torch, library, anchor)
        fixtures = verify_fixtures(torch, methods, seed, device, ROWS)
        works, weight, buffers = base.make_work(torch, seed, device)
        original_weight = weight.view(torch.int16).clone()
        fingerprints = base.input_fingerprints(works, weight)
        runner = StagedReplay(torch, works)
        copies = [base.capture(torch, None, work, weight) for work in works]
        combined = {name: [base.capture(torch, fn, work, weight) for work in works] for name, fn in methods.items()}
        operators = {name: [base.capture(torch, fn, work, weight, restore=False) for work in works]
                     for name, fn in methods.items()}
        reference, joined, restored, staged = verify_stages(torch, runner, copies, operators, combined, works, weight,
                                                            original_weight)
        numeric = {"fixtures": fixtures, "graph_checks": joined, "restored_before_operator": restored,
                   "split_checks": staged, "buffers": buffers, "source_fingerprints": fingerprints,
                   "inputs_unchanged": base.input_fingerprints(works, weight) == fingerprints}
        eligible = numeric_gate(numeric)
        timing = None
        if eligible and should_measure:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            timing = measure(torch, seed, runner, copies, operators, combined, works, reference, uuid)
            timing["inputs_unchanged_after_timing"] = base.input_fingerprints(works, weight) == fingerprints
            timing["outputs_match_after_timing"] = all(base.outputs_match(b["post_graph_checks"]) for b in timing["blocks"])
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": anchor["asset"], "codegen": codegen, "numerical": numeric,
            "gate_passed": eligible, "timing": timing,
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
