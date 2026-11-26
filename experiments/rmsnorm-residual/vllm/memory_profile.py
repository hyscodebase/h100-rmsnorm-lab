"""Collect one exact 512-row workload seed with Nsight Compute range replay."""
import argparse
import csv
import ctypes
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

import copy_free_run as copy_free
import run
import sustained_run as base
from wheel import MEMBER

sys.path.insert(1, str(run.ROOT))
from lab import save_json

WORKLOAD = copy_free.ROWS_512
ROWS, WIDTH = WORKLOAD.rows, copy_free.WIDTH
NUMERIC_SEED, TIMING_SEEDS = 165, (166, 167, 168)
SEEDS = (NUMERIC_SEED, *TIMING_SEEDS)
SPLIT, FUSED = WORKLOAD.backends[1:]
BACKENDS = (SPLIT, FUSED)
PROFILE_ANCHOR = run.HERE / "copy-free-512-results/s161.json"
SOURCES = (*WORKLOAD.sources, *(str(path.relative_to(run.ROOT)) for path in (
    PROFILE_ANCHOR, Path(__file__).resolve(), run.HERE / "COPY_FREE_MEMORY_512.md",
)), "lab.py", "gemm_plan.py", "materialized_plan.py", "residual_plan.py")
ARCHIVE_SHA256 = "6f55f6dfb33abe15b25a5f0c656ae8d9b6a6f33e55f0337b9b6a017daf8ad610"
VERSION, BUILD = "2025.1.1.0", "35528883"
METRICS = {
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "lts__t_sectors_op_read.sum": "sector",
    "gpu__time_duration.sum": "ns",
}
UNIT_FORMS = {"byte": {"byte", "bytes"}, "sector": {"sector", "sectors"}, "ns": {"ns", "nsecond"}}
PROFILE_OPTIONS = (
    "--replay-mode", "app-range", "--profile-from-start", "off", "--cache-control", "none",
    "--clock-control", "none", "--metrics", ",".join(METRICS),
)
WARMUP_BURSTS = 2
ROWS_PER_RANGE = base.STREAMS * base.BURST_REPLAYS * run.OPS * ROWS
ELEMENTS_PER_RANGE = ROWS_PER_RANGE * WIDTH


def schedule(seed):
    orders = base.balanced_order(seed, BACKENDS)
    if seed == NUMERIC_SEED:
        return orders[:1]
    pairs = orders * 2
    random.Random(seed).shuffle(pairs)
    return pairs


def range_sequence(seed):
    return [(pair, position, backend) for pair, order in enumerate(schedule(seed))
            for position, backend in enumerate(order)]


def source_hashes():
    return {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def profiler_identity(ncu):
    archive = (ncu.parent / ".verified-archive-sha256").read_text().strip()
    output = subprocess.check_output([str(ncu), "--version"], text=True)
    if archive != ARCHIVE_SHA256 or f"Version {VERSION} (build {BUILD})" not in output:
        raise ValueError("Use the verified Nsight Compute 2025.1.1 archive.")
    return {"archive_sha256": archive, "version": VERSION, "build": BUILD, "options": list(PROFILE_OPTIONS)}


def counter_value(raw, unit):
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        raise ValueError("A requested raw counter is unavailable.") from None
    if not value.is_finite() or value < 0 or (unit != "ns" and value != value.to_integral_value()):
        raise ValueError("A raw counter value is invalid.")
    return value


def parse_counter_csv(text):
    rows = csv.reader(io.StringIO(text))
    for header in rows:
        if set(METRICS) <= set(header) and "ID" in header and "Kernel Name" in header:
            break
    else:
        raise ValueError("Nsight did not export the four requested counters.")
    columns = {metric: header.index(metric) for metric in METRICS}
    units = next(rows, [])
    if len(units) != len(header) or any(units[column] not in UNIT_FORMS[METRICS[metric]]
                                      for metric, column in columns.items()):
        raise ValueError("The raw counter table uses unexpected units.")
    records = []
    for row in rows:
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError("The raw counter row is incomplete.")
        names = row[header.index("Kernel Name")]
        record_id = row[header.index("ID")]
        if not re.fullmatch(r"[0-9]+", record_id) or not re.fullmatch(r"Range[ :#]*[0-9]+", names, re.I):
            raise ValueError("The counter export contains a non-range row.")
        counters = {name: {"value": row[column], "unit": units[column]} for name, column in columns.items()}
        for name, item in counters.items():
            counter_value(item["value"], METRICS[name])
        records.append({"profiler_id": int(record_id), "range_name": names, "counters": counters})
    return records


def paired_summary(seed, counters):
    sequence = range_sequence(seed)
    ids = [row["profiler_id"] for row in counters]
    names = [int(re.search(r"([0-9]+)$", row["range_name"]).group(1)) for row in counters]
    if (len(counters) != len(sequence) or len(set(ids)) != len(ids) or ids != sorted(ids)
            or names != list(range(names[0], names[0] + len(names)))):
        raise ValueError("The profiler range order differs from the recorded execution.")
    pairs = []
    per_element = {backend: [] for backend in BACKENDS}
    values = {}
    for (pair, position, backend), row in zip(sequence, counters, strict=True):
        read, write, l2, ns = (counter_value(row["counters"][metric]["value"], unit)
                              for metric, unit in METRICS.items())
        if read <= 0 or write <= 0 or l2 <= 0 or ns <= 0:
            raise ValueError("The profiler returned an empty or unmeasured range.")
        values[backend] = {
            "dram_read_bytes_per_element": float(read / ELEMENTS_PER_RANGE),
            "dram_write_bytes_per_element": float(write / ELEMENTS_PER_RANGE),
            "l2_read_sectors_per_element": float(l2 / ELEMENTS_PER_RANGE),
            "duration_ns": float(ns),
        }
        per_element[backend].append(values[backend])
        if position == 1:
            delta = values[SPLIT]["dram_read_bytes_per_element"] - values[FUSED]["dram_read_bytes_per_element"]
            pairs.append({"pair": pair, "split_minus_fused_dram_read_bytes_per_element": delta})
    median = statistics.median(pair["split_minus_fused_dram_read_bytes_per_element"] for pair in pairs)
    return {
        "pairs": pairs, "median_split_minus_fused_dram_read_bytes_per_element": median,
        "passes_0_5_byte_threshold": median >= .5,
        "backend_medians": {backend: {key: statistics.median(row[key] for row in samples)
                                     for key in samples[0]} for backend, samples in per_element.items()},
    }


def profiler_call(driver, function):
    status = getattr(driver, function)()
    if status != 0:
        raise RuntimeError(f"{function} returned CUDA driver status {status}.")


def execute_target(seed, asset_dir, execution_dir):
    import torch
    import triton
    from eight_warp_outputs import fused_out_of_place, split_out_of_place

    started, clock = run.original.utc_now(), time.perf_counter()
    driver = ctypes.CDLL("libcuda.so.1")
    anchor = json.loads(PROFILE_ANCHOR.read_bytes())
    library = asset_dir / Path(MEMBER).name
    environment = base.validate_stack(torch, triton, library, anchor)
    device = torch.device("cuda:0")
    ranges = []
    with torch.inference_mode(), torch.cuda.device(device):
        vllm = base.load_vllm(torch, library)
        methods = {SPLIT: split_out_of_place, FUSED: fused_out_of_place}
        codegen = copy_free.compile_backends(torch, methods, WORKLOAD)
        copy_free.verify_anchor_identity(codegen, anchor["codegen"], WORKLOAD)
        verified = copy_free.verify_input_graphs(torch, seed, device, WORKLOAD, methods, vllm)
        if not copy_free.numeric_gate(verified.numerical, WORKLOAD.backends):
            raise ValueError("The input and output checks failed before profiling.")
        for pair, position, backend in range_sequence(seed):
            graphs = verified.graphs[backend]
            warmup = [verified.runner.burst(graphs, base.BURST_REPLAYS) for _ in range(WARMUP_BURSTS)]
            profiler_call(driver, "cuProfilerStart")
            try:
                elapsed = verified.runner.burst(graphs, base.BURST_REPLAYS)
            finally:
                profiler_call(driver, "cuProfilerStop")
            checks = base.compare(torch, verified.works, verified.reference)
            if not base.outputs_match(checks):
                raise ValueError("An output changed after a profiled range.")
            ranges.append({"pair": pair, "position": position, "backend": backend, "warmup_cuda_ms": warmup,
                           "cuda_ms": elapsed, "normalized_rows": ROWS_PER_RANGE, "post_graph_checks": checks})
        fingerprints = base.input_fingerprints(verified.works, verified.weight)
        if fingerprints != verified.numerical["source_fingerprints"]:
            raise ValueError("An immutable source buffer changed after profiling.")
    result = {
        "seed": seed, "environment": environment, "asset": anchor["asset"], "codegen": codegen,
        "numerical": verified.numerical, "ranges": ranges, "final_source_fingerprints": fingerprints,
        "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock,
        "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    }
    save_json(execution_dir / f"process-{os.getpid()}.json", result)


def adjudicate_collection(seed, attempt):
    profile_log = (attempt / "profile.log").read_text(errors="replace")
    counters = parse_counter_csv((attempt / "counters.csv").read_text())
    traces = [json.loads(path.read_bytes()) for path in (attempt / "executions").glob("process-*.json")]
    traces.sort(key=lambda record: record["started_utc"])
    if not traces:
        raise ValueError("Nsight did not complete an instrumented application execution.")
    first = traces[0]
    for record in traces:
        order = [(r["pair"], r["position"], r["backend"]) for r in record["ranges"]]
        if (record["seed"] != seed or order != range_sequence(seed)
                or record["final_source_fingerprints"] != first["final_source_fingerprints"]
                or copy_free.binary_identity(record["codegen"], candidate=FUSED)
                != copy_free.binary_identity(first["codegen"], candidate=FUSED)):
            raise ValueError("Application replays did not execute the same input and launch contract.")
        if any(row["normalized_rows"] != ROWS_PER_RANGE or not math.isfinite(row["cuda_ms"])
               or row["cuda_ms"] <= 0 for row in record["ranges"]):
            raise ValueError("A profiled launch did not complete the registered work.")
    pass_counts = [int(value) for value in re.findall(r"(\d+)\s+passes?\b", profile_log, re.I)]
    return counters, traces, pass_counts, paired_summary(seed, counters)


def collect(seed, asset_dir, ncu, artifact_dir, qualification, revision):
    started, clock = run.original.utc_now(), time.perf_counter()
    sources = source_hashes()
    profiler = profiler_identity(ncu)
    if seed in TIMING_SEEDS and (
        qualification is None or (qualification["seed"], qualification["gate_passed"]) != (NUMERIC_SEED, True)
        or qualification["code_revision"] != revision or qualification["source_sha256"] != sources
        or qualification["profiler"] != profiler
    ):
        raise ValueError("A qualified seed 165 with the exact source and tool is required.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    completed = [path.parent for path in artifact_dir.glob("attempt-*/capture.ncu-rep") if path.stat().st_size]
    if len(completed) > 1:
        raise ValueError("Multiple Nsight captures require reconciliation before publication.")
    if completed:
        attempt = completed[0]
    else:
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=artifact_dir))
        (attempt / "executions").mkdir()
        command = [str(ncu), *PROFILE_OPTIONS, "--export", str(attempt / "capture.ncu-rep"),
                   sys.executable, str(Path(__file__).resolve()), "--seed", str(seed), "--asset-dir", str(asset_dir),
                   "--execution-dir", str(attempt / "executions")]
        with (attempt / "profile.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            tail = (attempt / "profile.log").read_text(errors="replace").splitlines()[-16:]
            raise RuntimeError("Nsight Compute exited with status " + str(result.returncode) + "\n" + "\n".join(tail))
    report = attempt / "capture.ncu-rep"
    if not report.is_file():
        raise ValueError("Nsight did not write its capture report.")
    raw = subprocess.check_output([str(ncu), "--import", str(report), "--page", "raw", "--csv", "--print-units", "base"],
                                  text=True, stderr=subprocess.STDOUT, env={**os.environ, "LC_ALL": "C"})
    (attempt / "counters.csv").write_text(raw)
    counters, traces, pass_counts, summary = adjudicate_collection(seed, attempt)
    with report.open("rb") as stream:
        profile_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"seed": seed, "rows": ROWS, "width": WIDTH, "code_revision": revision, "source_sha256": sources,
            "profiler": profiler, "profile_sha256": profile_sha256,
            "ranges": counters, "executions": traces, "reported_pass_counts": pass_counts, "summary": summary,
            "gate_passed": True, "timing": None if seed == NUMERIC_SEED else summary,
            "started_utc": started, "finished_utc": run.original.utc_now(), "wall_seconds": time.perf_counter() - clock}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--asset-dir", type=Path, required=True)
    cli.add_argument("--ncu", type=Path)
    cli.add_argument("--artifact-dir", type=Path)
    cli.add_argument("--qualified", type=Path)
    cli.add_argument("--execution-dir", type=Path)
    args = cli.parse_args()
    revision = os.environ.get("BENCH_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        cli.error("Provide the frozen revision.")
    if args.execution_dir:
        execute_target(args.seed, args.asset_dir, args.execution_dir)
    else:
        if not args.ncu or not args.artifact_dir:
            cli.error("Provide --ncu and --artifact-dir for a collection.")
        qualification = json.loads(args.qualified.read_bytes()) if args.qualified else None
        row = collect(args.seed, args.asset_dir, args.ncu, args.artifact_dir, qualification, revision)
        print(json.dumps(row, allow_nan=False, separators=(",", ":")))
