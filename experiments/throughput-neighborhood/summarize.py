"""Audit and report the frozen 32,768-row RMSNorm throughput experiment."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "plans/throughput-neighborhood.json"
RAW = ROOT / "results/throughput-neighborhood/raw"
CHECKPOINTS = RAW / "checkpoints"
ATTEMPTS = RAW / "attempts"
SOURCE_PIN = RAW / "source-pin.json"
SUMMARY_PATH = RAW / "summary.json"
REPORT_PATH = ROOT / "results/throughput-neighborhood/REPORT.md"
SOURCES = ("kernels.py", "experiments/throughput-neighborhood/run.py",
           "experiments/throughput-neighborhood/PLAN.md", "plans/throughput-neighborhood.json",
           "experiments/throughput-neighborhood/step.py",
           "experiments/throughput-neighborhood/ssh_runner.sh",
           "experiments/throughput-neighborhood/summarize.py")
COMPUTE_SOURCES = SOURCES[:4]
BACKENDS = ("triton_4w_1r", "triton_8w_1r", "triton_4w_2r", "torch_compile")
GIB = 1024**3


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan():
    plan = json.loads(PLAN_PATH.read_text())
    need(plan["issue"] == 62 and plan["rows"] == 32768
         and plan["widths"] == [4103, 4104, 4105, 4112]
         and plan["seeds"] == [47, 48, 49]
         and plan["candidate"] == "triton_4w_2r"
         and plan["one_row_references"] == ["triton_4w_1r", "triton_8w_1r"]
         and plan["compiled_reference"] == "torch_compile"
         and plan["rounds"] == 401 and plan["ops_per_graph"] == 4,
         "Frozen shape/seed/backend/replay plan differs from issue #62.")
    output_bytes = plan["rows"] * plan["max_width"] * plan["output_element_bytes"]
    pool_bytes = output_bytes * plan["ops_per_graph"] * len(BACKENDS)
    need(plan["max_width"] == max(plan["widths"]), "The maximum width must bound the planned widths.")
    need(output_bytes == 269_484_032 == plan["estimated_output_bytes_per_graph_op"],
         "The one-output memory estimate differs from 32,768×4,112×BF16.")
    need(pool_bytes == 4_311_744_512 == plan["estimated_worst_case_graph_output_pool_bytes"],
         "The worst-case graph-output-pool estimate differs.")
    need(pool_bytes / GIB < 8 and pool_bytes < plan["measured_output_pool_ceiling_bytes"],
         "The conservative output-pool estimate exceeds its preregistered ceiling.")
    need(plan["free_memory_guard_bytes"] == 16 * GIB
         and plan["measured_output_pool_ceiling_bytes"] == 8 * GIB,
         "Memory preflight guards differ from the registered thresholds.")
    invocations = plan["rounds"] * plan["ops_per_graph"] * len(BACKENDS)
    traffic = invocations * 2 * output_bytes
    need(traffic == plan["estimated_nominal_input_plus_output_bytes_per_checkpoint"],
         "The nominal input-plus-output traffic estimate differs.")
    return plan, {"one_output_bytes": output_bytes, "worst_case_pool_bytes": pool_bytes,
                  "worst_case_pool_gib": pool_bytes / GIB,
                  "invocations_per_checkpoint": invocations,
                  "intervals_per_checkpoint": plan["rounds"] * len(BACKENDS),
                  "nominal_input_plus_output_bytes_per_checkpoint": traffic}


def verify_source_revision(revision, expected):
    for path, digest in expected.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{path}"])
        need(hashlib.sha256(blob).hexdigest() == digest, f"Frozen source hash mismatch at {revision}:{path}.")


def median(values):
    return statistics.median(values)


def telemetry_stats(samples, windows):
    if not samples:
        return {"sample_count": 0, "window_span_seconds_sum": 0,
                "max_within_checkpoint_gap_seconds": None, "gpu_busy_median_percent": None,
                "memory_busy_median_percent": None, "vram_median_mib": None, "power_median_w": None}
    def values(key):
        return [sample[key] for sample in samples if sample.get(key) is not None]
    return {"sample_count": len(samples),
            "window_span_seconds_sum": sum(window["span_seconds"] for window in windows),
            "max_within_checkpoint_gap_seconds": max(w["max_gap_seconds"] for w in windows),
            "gpu_busy_median_percent": median(values("util_pct")) if values("util_pct") else None,
            "memory_busy_median_percent": median(values("memory_busy_pct")) if values("memory_busy_pct") else None,
            "vram_median_mib": median(values("used_mib")) if values("used_mib") else None,
            "power_median_w": median(values("power_w")) if values("power_w") else None}


def audit():
    plan, estimate = load_plan()
    pin = json.loads(SOURCE_PIN.read_text())
    revision = pin["source_freeze_revision"]
    source_hashes = pin["source_sha256"]
    need(set(source_hashes) == set(SOURCES), "Source pin lacks execution or compute sources.")
    for name, digest in source_hashes.items():
        need(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
             f"Working source differs from pin: {name}.")
    verify_source_revision(revision, source_hashes)
    expected = {(seed, width) for seed in plan["seeds"] for width in plan["widths"]}
    files = {path.stem: path for path in CHECKPOINTS.glob("s*-w*.json")}
    wanted = {f"s{seed}-w{width}" for seed, width in expected}
    need(set(files) == wanted,
         f"Checkpoint coverage differs. Missing {sorted(wanted-set(files))}; extra {sorted(set(files)-wanted)}.")
    environment, cells = None, {}
    all_samples, windows = [], []
    graph_event_seconds = 0.0
    for seed, width in sorted(expected):
        key = f"s{seed}-w{width}"
        record = json.loads(files[key].read_text())
        need((record.get("seed"), record.get("rows"), record.get("width")) == (seed, plan["rows"], width),
             f"Checkpoint identity mismatch in {key}.")
        need(record.get("code_revision") == revision and record.get("source_freeze_revision") == revision,
             f"Checkpoint source revision differs in {key}.")
        need(record.get("source_sha256") == source_hashes
             and record.get("research_assistant_model") == pin["research_assistant_model"],
             f"Source/model metadata mismatch in {key}.")
        need(record.get("telemetry_tool_sha256") == pin["telemetry_tool_sha256"],
             f"Telemetry-tool source differs from its frozen pin in {key}.")
        need(record.get("remote_source_sha256") == {name: source_hashes[name] for name in COMPUTE_SOURCES},
             f"Remote compute source hashes differ in {key}.")
        env = record["environment"]
        need("H100" in env["gpu"] and env["compute_capability"] == [9, 0]
             and env["total_memory_bytes"] > 0, f"Non-H100 execution in {key}.")
        need(environment in (None, env), f"Environment changed within the sweep at {key}.")
        environment = env
        method = record["method"]
        need((method["rounds"], method["ops_per_graph"], method["timed_backends"])
             == (401, 4, list(BACKENDS)), f"Timing design differs in {key}.")
        correctness = record["correctness"]
        need(correctness["strict_gate"] == {"rtol": 0.005, "atol": 1e-6, "passed": True}
             and correctness["benchmark_crosscheck"] == {"rtol": 0.016, "atol": 1e-5, "passed": True},
             f"Numerical gate failed in {key}.")
        need(len(correctness["fixtures"]) == 32
             and all(x["strict_passed"] and x["crosscheck_passed"] for x in correctness["fixtures"]),
             f"Fixture coverage failed in {key}.")
        need(len(correctness["timed_input"]) == 4
             and all(x["strict_passed"] and x["crosscheck_passed"] for x in correctness["timed_input"]),
             f"Timed-input correctness failed in {key}.")
        mem = record["memory_preflight"]
        need(mem["minimum_free_bytes"] == 16*GIB and mem["output_pool_ceiling_bytes"] == 8*GIB
             and mem["pre_capture"]["free_bytes"] >= 16*GIB
             and mem["capture_reserved_delta_peak_bytes"] <= 8*GIB,
             f"Memory guard failed in {key}.")
        for snap in [mem["correctness_peak_before_capture"], mem["pre_capture"],
                     *mem["per_backend_capture"], mem["final"]]:
            need(all(type(snap[field]) is int and snap[field] >= 0
                     for field in ("allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes")),
                 f"Missing allocated/reserved/free/total memory snapshot in {key}.")
        need(len(mem["per_backend_capture"]) == 4
             and {s["backend"] for s in mem["per_backend_capture"]} == set(BACKENDS),
             f"Missing per-backend capture memory snapshot in {key}.")
        capture_delta_peak = max(s["capture_reserved_delta_bytes"] for s in mem["per_backend_capture"])
        need(mem["capture_reserved_delta_peak_bytes"] == capture_delta_peak
             and capture_delta_peak <= 8*GIB,
             f"Capture delta peak does not match the backend snapshots in {key}.")
        full_allocated = max(mem["correctness_peak_before_capture"]["peak_allocated_bytes"],
                             mem["capture_and_timing_peak_allocated_bytes"])
        full_reserved = max(mem["correctness_peak_before_capture"]["peak_reserved_bytes"],
                            mem["capture_and_timing_peak_reserved_bytes"])
        need(mem["full_checkpoint_peak_allocated_bytes"] == full_allocated
             and mem["full_checkpoint_peak_reserved_bytes"] == full_reserved,
             f"Full-checkpoint peak does not combine the two measured phases in {key}.")
        need(mem["full_checkpoint_peak_reserved_bytes"]
             == max(mem["correctness_peak_before_capture"]["peak_reserved_bytes"],
                    mem["capture_and_timing_peak_reserved_bytes"]),
             f"Full-checkpoint peak does not combine the two measured phases in {key}.")
        estimate_record = record["graph_output_estimate"]
        need(estimate_record["one_output_tensor_bytes_at_max_width"] == estimate["one_output_bytes"]
             and estimate_record["worst_case_graph_output_pool_bytes"] == estimate["worst_case_pool_bytes"],
             f"Runtime memory estimate differs in {key}.")
        intervals = record["timing_intervals"]
        need(len(intervals) == 401*4, f"Wrong interval count in {key}.")
        by_round, times = {}, {}
        for item in intervals:
            by_round.setdefault(item["round"], []).append(item)
            need(item["ops_per_graph"] == 4 and math.isfinite(item["graph_elapsed_us"])
                 and item["graph_elapsed_us"] > 0,
                 f"Invalid raw graph interval in {key}.")
            recomputed = item["graph_elapsed_us"] / item["ops_per_graph"]
            need(math.isfinite(item["us_per_op"]) and item["us_per_op"] > 0
                 and math.isfinite(recomputed) and math.isclose(item["us_per_op"], recomputed,
                                                              rel_tol=1e-12, abs_tol=1e-12),
                 f"Invalid interval in {key}.")
            times.setdefault(item["backend"], []).append(recomputed)
            graph_event_seconds += item["graph_elapsed_us"]/1e6
        need(set(by_round) == set(range(401)), f"Missing timing rounds in {key}.")
        for round_id, group in by_round.items():
            need(len(group) == 4 and {x["backend"] for x in group} == set(BACKENDS)
                 and sorted(x["order_index"] for x in group) == [0, 1, 2, 3],
                 f"Bad randomized order in {key}, round {round_id}.")
        medians = {name: median(values) for name, values in times.items()}
        one_row = min(medians["triton_4w_1r"], medians["triton_8w_1r"])
        candidate = medians["triton_4w_2r"]
        cells[seed, width] = {
            "seed": seed, "width": width, "candidate_us": candidate, "fastest_one_row_us": one_row,
            "torch_compile_us": medians["torch_compile"], "backend_median_us": medians,
            "gain_vs_one_row_percent": 100*(1-candidate/one_row),
            "gain_vs_torch_compile_percent": 100*(1-candidate/medians["torch_compile"]),
            "wall_seconds": record["wall_seconds"],
            "full_checkpoint_peak_allocated_bytes": mem["full_checkpoint_peak_allocated_bytes"],
            "full_checkpoint_peak_reserved_bytes": mem["full_checkpoint_peak_reserved_bytes"]}
        cell_samples = []
        for item in record.get("telemetry", []):
            path = ROOT / item["path"]
            need(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                 f"Telemetry hash mismatch in {key}.")
            for line in path.read_text().splitlines():
                sample = json.loads(line)
                cell_samples.append(sample)
                all_samples.append(sample)
        if cell_samples:
            stamps = sorted(datetime.fromisoformat(x["timestamp_utc"]) for x in cell_samples)
            gaps = [(b-a).total_seconds() for a,b in zip(stamps, stamps[1:])]
            windows.append({"checkpoint": key, "samples": len(stamps),
                            "span_seconds": (stamps[-1]-stamps[0]).total_seconds(),
                            "max_gap_seconds": max(gaps, default=0)})
    widths = {}
    for width in plan["widths"]:
        runs = [cells[seed, width] for seed in plan["seeds"]]
        qualifies = all(r["gain_vs_one_row_percent"] >= 5
                        and r["gain_vs_torch_compile_percent"] >= 5 for r in runs)
        widths[str(width)] = {"per_seed": runs, "qualifies": qualifies}
    attempts = [json.loads(path.read_text()) for path in ATTEMPTS.glob("s*-w*.json")]
    for attempt in attempts:
        for item in attempt.get("telemetry", []):
            path = ROOT / item["path"]
            need(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                 f"Attempt telemetry hash mismatch for {attempt.get('unit')}.")
            need(sum(1 for _ in path.open()) == item["samples"],
                 f"Attempt telemetry sample count mismatch for {attempt.get('unit')}.")
    return {"plan": plan, "memory_estimate": estimate, "source_freeze_revision": revision,
            "source_sha256": source_hashes, "research_assistant_model": pin["research_assistant_model"],
            "environment": environment, "widths": widths,
            "checkpoints": len(expected), "intervals_per_checkpoint": 401*4,
            "total_intervals": len(expected)*401*4,
            "total_kernel_invocations": len(expected)*estimate["invocations_per_checkpoint"],
            "graph_event_seconds": graph_event_seconds,
            "runner_wall_seconds": sum(cell["wall_seconds"] for cell in cells.values()),
            "maximum_full_checkpoint_peak_allocated_bytes": max(
                cell["full_checkpoint_peak_allocated_bytes"] for cell in cells.values()),
            "maximum_full_checkpoint_peak_reserved_bytes": max(
                cell["full_checkpoint_peak_reserved_bytes"] for cell in cells.values()),
            "attempts": len(attempts),
            "failed_attempts": sum(item.get("outcome") == "failed" for item in attempts),
            "telemetry": telemetry_stats(all_samples, windows),
            "checkpoint_telemetry_windows": windows}


def write_report(summary):
    plan, env = summary["plan"], summary["environment"]
    rows = ["# RMSNorm throughput-scale width neighborhood", "",
            "Issue [#62](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/62). This 32,768-row BF16 experiment checks whether the two-row/four-warp win near width 4,104 persists at a throughput-relevant workload.", "",
            f"Runtime: {env['gpu']} (CC {env['compute_capability']}); CUDA {env['cuda_runtime']}; PyTorch {env['torch']}; Triton {env['triton']}; driver {env['driver']}.", "",
            "The candidate is Triton four warps/two rows. Per seed, the one-row comparator is the faster of the four-warp and eight-warp kernels; the second reference is shape-specialized PyTorch. All checkpoints passed both recorded numerical gates before timing.", "",
            "| Width | Candidate µs (47 / 48 / 49) | Fastest one-row µs | Torch compile µs | Gain vs one-row | Gain vs torch compile | ≥5% in all seeds vs both? |",
            "|---:|---:|---:|---:|---:|---:|:---:|"]
    for width in plan["widths"]:
        runs = summary["widths"][str(width)]["per_seed"]
        def group(key):
            return " / ".join(f"{run[key]:.3f}" for run in runs)
        qualifies = summary["widths"][str(width)]["qualifies"]
        rows.append(f"| {width} | {group('candidate_us')} | {group('fastest_one_row_us')} | {group('torch_compile_us')} | {group('gain_vs_one_row_percent')}% | {group('gain_vs_torch_compile_percent')}% | {'Yes' if qualifies else 'No'} |")
    mem, telem = summary["memory_estimate"], summary["telemetry"]
    rows += ["",
             f"**Memory preflight:** one maximum-width BF16 output is {mem['one_output_bytes']:,} bytes; conservative graph-output pool is {mem['worst_case_pool_bytes']:,} bytes ({mem['worst_case_pool_gib']:.2f} GiB), below the 8 GiB ceiling. Every checkpoint records allocated/reserved/free/total memory before and after each capture. Largest full-checkpoint allocated/reserved peaks were {summary['maximum_full_checkpoint_peak_allocated_bytes']:,}/{summary['maximum_full_checkpoint_peak_reserved_bytes']:,} bytes.",
             "",
             f"**H100 use:** {summary['total_kernel_invocations']:,} kernel invocations across {summary['checkpoints']} checkpoints; CUDA event intervals total {summary['graph_event_seconds']:.2f}s and host runner wall totals {summary['runner_wall_seconds']:.1f}s. The observer recorded {telem['sample_count']} samples over {telem['window_span_seconds_sum']:.1f}s summed checkpoint spans; median GPU busy {telem['gpu_busy_median_percent']}%, memory busy {telem['memory_busy_median_percent']}%, VRAM {telem['vram_median_mib']} MiB, power {telem['power_median_w']}W. Telemetry gaps and per-checkpoint samples are archived; no saturation claim is inferred from sparse samples.",
             "",
             f"Attempt receipts: {summary['attempts']} total, {summary['failed_attempts']} failed. Model: {summary['research_assistant_model']}.",
             "",
             f"Sources were frozen at {summary['source_freeze_revision']}. Raw data and telemetry are in [raw/](raw); audited derivations are in [summary.json](raw/summary.json).",
             "",
             "The existing 1,024-row width 4,104 recommendation remains outside this throughput experiment's decision.", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true",
                        help="Validate frozen scope and memory estimates without GPU results.")
    args = parser.parse_args()
    if args.check_plan:
        plan, estimate = load_plan()
        print(json.dumps({"plan_id": plan["id"], **estimate}, indent=2))
    else:
        result = audit()
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_report(result)
        print(f"Verified {result['checkpoints']} checkpoints / {result['total_intervals']} intervals.")
