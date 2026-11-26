"""Run one 128-row residual RMSNorm seed under a sustained eight-stream schedule."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import time
from dataclasses import dataclass

from bounded_timing import bitwise_case, verify_fixtures
import run
import sixteen_warp_run as single
from sustained_sensors import Observer, associate, summarize_load
from wheel import MEMBER, sha

ROWS, WIDTH = single.ROWS, run.WIDTH
NUMERIC_SEEDS, TIMING_SEEDS = (137,), (138, 139, 140)
SEEDS = (*NUMERIC_SEEDS, *TIMING_SEEDS)
BACKENDS = single.BACKENDS
STREAMS, BURST_REPLAYS, WARMUP_S, PERIOD_S = 8, 32, 1.0, 6.0
ANCHOR = run.HERE / "sixteen-warp-results/s133.json"
SOURCES = (*single.SOURCES, *(str(path.relative_to(run.ROOT)) for path in (
    ANCHOR, Path(__file__).resolve(), run.HERE / "sustained_sensors.py", run.HERE / "SUSTAINED_128.md",
)))


@dataclass(frozen=True)
class Work:
    stream: object
    source: tuple
    buffer: tuple


class Replay:
    def __init__(self, torch, works):
        self.torch = torch
        self.streams = [work.stream for work in works]
        self.control = torch.cuda.Stream()
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.done = [torch.cuda.Event() for _ in works]

    def burst(self, graphs, repeats):
        self.begin.record(self.control)
        self.enqueue(graphs, self.begin, self.done, repeats)
        self.end.record(self.control)
        self.end.synchronize()
        return self.begin.elapsed_time(self.end)

    def enqueue(self, graphs, ready, completed, repeats):
        for stream in self.streams:
            stream.wait_event(ready)
        for _ in range(repeats):
            for stream, graph in zip(self.streams, graphs, strict=True):
                with self.torch.cuda.stream(stream):
                    graph.replay()
        for stream, event in zip(self.streams, completed, strict=True):
            event.record(stream)
            self.control.wait_event(event)


def make_work(torch, seed, device, *, rows=ROWS):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (run.OPS, rows, WIDTH)
    works = []
    for _ in range(STREAMS):
        source = tuple(torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator) for _ in range(2))
        works.append(Work(torch.cuda.Stream(), source, tuple(torch.empty_like(t) for t in source)))
    weight = torch.randn(WIDTH, dtype=torch.bfloat16, device=device, generator=generator)
    tensors = [t for work in works for t in (*work.source, *work.buffer)] + [weight]
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors)
    no_overlap = all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
    valid = all(start % 16 == 0 for start, _ in spans) and all(t.is_contiguous() for t in tensors)
    pair_stride = rows * WIDTH
    independent_pairs = all(t.shape == shape and t.stride(0) == pair_stride for t in tensors[:-1])
    stream_count = len({work.stream.cuda_stream for work in works})
    if not (no_overlap and valid and independent_pairs and stream_count == STREAMS):
        raise ValueError("The stream buffers are not independent contiguous BF16 pairs.")
    return works, weight, {"disjoint_aligned_buffers": True, "unique_streams": stream_count,
                           "disjoint_allocations": len(spans),
                           "input_and_output_bytes": sum(end - start for start, end in spans),
                           "pair_tensor_stride_bytes": pair_stride * 2, "pairs_per_stream": run.OPS}


def input_fingerprints(works, weight):
    def digest(tensor):
        cpu = tensor.cpu()
        return hashlib.sha256(ctypes.string_at(cpu.data_ptr(), cpu.numel() * cpu.element_size())).hexdigest()

    return {"weight_sha256": digest(weight),
            "stream_input_sha256": [[digest(t) for t in work.source] for work in works]}


def capture_graph(torch, stream, warmup, record):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        record()
    stream.synchronize()
    return graph


def capture(torch, fn, work, weight, *, restore=True):
    x, residual = work.buffer

    def warmup():
        run.restore(work.source, x, residual)
        if fn is not None:
            fn(x[0], residual[0], weight)

    def record():
        if restore:
            run.restore(work.source, x, residual)
        if fn is not None:
            for index in range(run.OPS):
                fn(x[index], residual[index], weight)

    return capture_graph(torch, work.stream, warmup, record)


def differences(torch, observed, reference):
    return torch.count_nonzero(observed.view(torch.int16) != reference.view(torch.int16), dim=(1, 2)).tolist()


def compare(torch, works, reference):
    return [
        {"stream": index, "activation_bit_differences_by_pair": differences(torch, work.buffer[0], expected_x),
         "residual_bit_differences_by_pair": differences(torch, work.buffer[1], expected_r)}
        for index, (work, (expected_x, expected_r)) in enumerate(zip(works, reference, strict=True))
    ]


def outputs_match(records):
    return tuple(record["stream"] for record in records) == tuple(range(STREAMS)) and all(
        len(record[field]) == run.OPS and not any(record[field])
        for record in records for field in ("activation_bit_differences_by_pair", "residual_bit_differences_by_pair")
    )


def poison_buffers(torch, works):
    for work in works:
        for tensor in work.buffer:
            tensor.fill_(float("nan"))
    torch.cuda.synchronize()


def verify_graphs(torch, runner, graphs, works, weight, initial_weight, *, backends=BACKENDS, check_inputs=False):
    checks = {}
    reference = []
    for name in backends:
        poison_buffers(torch, works)
        runner.burst(graphs[name], 1)
        if name == backends[0]:
            for work in works:
                x, residual = work.source
                expected_r = (x.float() + residual.float()).bfloat16()
                reference.append((work.buffer[0].clone(), expected_r))
        checks[name] = {"streams": compare(torch, works, reference),
                        "finite_activations": all(bool(torch.isfinite(work.buffer[0]).all()) for work in works),
                        "weight_unchanged": bool(torch.equal(weight.view(torch.int16), initial_weight))}
        if check_inputs:
            checks[name]["source_fingerprints"] = input_fingerprints(works, weight)
    return reference, checks


def numeric_gate(record, backends=BACKENDS):
    fixtures = record["fixtures"]
    checks = record["graph_checks"]
    return (len(fixtures) == 8 and all(bitwise_case(case) for case in fixtures)
            and set(checks) == set(backends)
            and all(c["weight_unchanged"] and c["finite_activations"] and outputs_match(c["streams"])
                    for c in checks.values())
            and record["inputs_unchanged"])


def balanced_order(seed, backends=BACKENDS):
    order = list(backends)
    random.Random(seed).shuffle(order)
    return [order[i:] + order[:i] for i in range(len(order))]


def collect_window(runner, graphs, minimum_seconds, origin, *, rows=ROWS):
    started_utc = run.original.utc_now()
    started = time.perf_counter() - origin
    bursts = []
    while not bursts or bursts[-1]["end_s"] - started < minimum_seconds:
        start = time.perf_counter() - origin
        milliseconds = runner.burst(graphs, BURST_REPLAYS)
        end = time.perf_counter() - origin
        bursts.append({"start_s": start, "end_s": end, "cuda_ms": milliseconds})
    finished = bursts[-1]["end_s"]
    completed = STREAMS * BURST_REPLAYS * len(bursts)
    return {"start_s": started, "end_s": finished, "wall_seconds": finished - started,
            "started_utc": started_utc, "finished_utc": run.original.utc_now(), "bursts": bursts,
            "attempted_graphs": completed, "completed_graphs": completed,
            "normalized_rows": completed * run.OPS * rows,
            "cuda_ms": sum(burst["cuda_ms"] for burst in bursts)}


def pace(blocks, backends=BACKENDS):
    rates = {backend: statistics.median(block["normalized_rows"] / block["wall_seconds"]
                                       for block in blocks if block["backend"] == backend) for backend in backends}
    new_rate = rates[backends[-1]]
    gains = {backend: {"throughput_gain": new_rate / rates[backend] - 1,
                       "passes_5pct": new_rate >= rates[backend] * 1.05} for backend in backends[:-1]}
    return rates, gains


def measure(torch, seed, runner, graphs, works, reference, cuda_uuid, *, backends=BACKENDS, period_s=PERIOD_S, rows=ROWS):
    origin_utc = run.original.utc_now()
    origin = time.perf_counter()
    observer = Observer(cuda_uuid, origin)
    order, blocks = balanced_order(seed, backends), []
    try:
        for round_index, group in enumerate(order):
            for position, backend in enumerate(group):
                warm = collect_window(runner, graphs[backend], WARMUP_S, origin, rows=rows)
                measured = collect_window(runner, graphs[backend], period_s, origin, rows=rows)
                measured.update(id=len(blocks), round=round_index, position=position, backend=backend, warmup=warm,
                                post_graph_checks=compare(torch, works, reference))
                blocks.append(measured)
    finally:
        telemetry = observer.finish()
    associate(telemetry["slots"], blocks)
    telemetry["by_backend"] = summarize_load(blocks, telemetry["slots"], backends)
    rates, gains = pace(blocks, backends)
    return {"observer_origin_utc": origin_utc, "order": order, "blocks": blocks, "telemetry": telemetry,
            "median_rows_per_second": rates, "comparisons": gains}


def validate_stack(torch, triton, library, anchor):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not anchor["gate_passed"]:
        raise ValueError("Use the recorded H100 and a passing kernel anchor.")
    environment = single.grouped.environment_metadata(torch, triton)
    if environment != anchor["environment"] or sha(library) != anchor["asset"]["extension_sha256"]:
        raise ValueError("The H100 or released extension differs from the anchor.")
    return environment


def load_vllm(torch, library):
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.ops.load_library(str(library.resolve()))
    op = torch.ops._C.fused_add_rms_norm
    return lambda x, r, w: op(x, r, w, 1e-6)


def prepare_backends(torch, library, anchor):
    from sixteen_warp_kernel import fused_sixteen_warps
    from small_rows_kernel import fused_split_warps

    methods = {BACKENDS[0]: load_vllm(torch, library), BACKENDS[1]: fused_split_warps,
               BACKENDS[2]: fused_sixteen_warps}
    codegen = {BACKENDS[1]: single.small.compile_split(torch, fused_split_warps),
               BACKENDS[2]: single.compile_sixteen(torch, fused_sixteen_warps)}
    if any(code["ptx_instruction_sha256"] != anchor["codegen"][backend]["ptx_instruction_sha256"]
           for backend, code in codegen.items()):
        raise ValueError("A Triton kernel's instructions differ from the registered comparison.")
    return methods, codegen


def checkpoint(seed, asset_dir, qualification, revision):
    import torch
    import triton

    started, clock = run.original.utc_now(), time.perf_counter()
    anchor = json.loads(ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = validate_stack(torch, triton, library, anchor)
    sources = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    should_measure = seed in TIMING_SEEDS
    if should_measure and (
        qualification is None or not qualification["gate_passed"]
        or (qualification["rows"], qualification["width"], qualification["seed"]) != (ROWS, WIDTH, NUMERIC_SEEDS[0])
        or qualification["source_sha256"] != sources or qualification["code_revision"] != revision
        or qualification["environment"] != environment or qualification["asset"] != anchor["asset"]
    ):
        raise ValueError("A passing preflight from the exact revision is required.")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        methods, codegen = prepare_backends(torch, library, anchor)
        fixtures = verify_fixtures(torch, methods, seed, device, ROWS)
        works, weight, address_check = make_work(torch, seed, device)
        initial_weight = weight.view(torch.int16).clone()
        fingerprints = input_fingerprints(works, weight)
        runner = Replay(torch, works)
        graphs = {name: [capture(torch, fn, work, weight) for work in works] for name, fn in methods.items()}
        reference, graph_checks = verify_graphs(torch, runner, graphs, works, weight, initial_weight)
        numerical = {"fixtures": fixtures, "graph_checks": graph_checks, "buffers": address_check,
                     "source_fingerprints": fingerprints,
                     "inputs_unchanged": input_fingerprints(works, weight) == fingerprints}
        eligible = numeric_gate(numerical)
        load = None
        if eligible and should_measure:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            load = measure(torch, seed, runner, graphs, works, reference, uuid)
            load["inputs_unchanged_after_timing"] = input_fingerprints(works, weight) == fingerprints
            load["outputs_match_after_timing"] = all(outputs_match(b["post_graph_checks"]) for b in load["blocks"])
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "environment": environment, "asset": anchor["asset"], "codegen": codegen, "numerical": numerical,
            "gate_passed": eligible, "timing": load,
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
    print(json.dumps(checkpoint(args.seed, args.asset_dir, previous, revision), indent=2, allow_nan=False))
