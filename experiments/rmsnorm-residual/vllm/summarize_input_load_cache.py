"""Audit archived cache-policy checkpoints and regenerate their summary."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "input-load-cache-512-results"
PLAN = json.loads((HERE / "input-load-cache-512-plan.json").read_text())
BASELINE = "triton_8warps_out_of_place"
CANDIDATE = "triton_8warps_cg_inputs_out_of_place"
REFERENCE = "vllm_0_10_0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def verify_exact(record):
    fixtures = record["numerical"]["fixtures"]
    require(len(fixtures) == 8 and all(row["passed"] for row in fixtures), "Numerical fixtures failed.")
    for fixture in fixtures:
        comparisons = fixture["comparisons"]
        require(set(comparisons) == {BASELINE, CANDIDATE}, "Fixture backend set differs from the plan.")
        require(all(row["pairs_differ_bitwise"] == 0 for row in comparisons.values()), "Fixture outputs differ.")
    checks = record["numerical"]["graph_checks"]
    require(set(checks) == {REFERENCE, BASELINE, CANDIDATE}, "Captured graph backend set differs from the plan.")
    for check in checks.values():
        require(check["weight_unchanged"] and check["finite_activations"], "Captured graph mutated the weight or produced nonfinite output.")
        require(check["source_fingerprints"] == record["source_fingerprints"], "Captured graph fingerprints differ.")
        require(len(check["streams"]) == 8, "Missing captured graph stream checks.")
        for stream in check["streams"]:
            require(not any(stream["activation_bit_differences_by_pair"])
                    and not any(stream["residual_bit_differences_by_pair"]), "Captured graph output differs.")
    require(record["numerical"]["inputs_unchanged"], "Qualification changed its inputs.")


def main():
    plan = PLAN
    timing_plan = plan["timing"]
    qualification = read("s209")
    require(qualification["gate_passed"] and qualification["period"] is None, "Seed 209 qualification did not pass.")
    verify_exact(qualification)
    revision = qualification["code_revision"]
    source_hashes = qualification["source_sha256"]
    environment, asset = qualification["environment"], qualification["asset"]
    subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=ROOT, check=True, capture_output=True)
    for name, expected in source_hashes.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        require(digest == expected, f"Source hash mismatch: {name}")

    anchor = json.loads((HERE / "long-window-energy-512-results/s205.json").read_text())
    base_code, candidate_code = qualification["codegen"][BASELINE], qualification["codegen"][CANDIDATE]
    require(base_code["binary_sha256"] == anchor["codegen"][BASELINE]["binary_sha256"], "Baseline binary differs from seed 205.")
    require(base_code["num_warps"] == candidate_code["num_warps"] == 8, "Launch geometry differs.")
    require(base_code["registers_per_thread"] == candidate_code["registers_per_thread"] == 32, "Register counts differ.")
    require(base_code["spills"] == candidate_code["spills"] == 0, "A kernel spilled registers.")
    loads = qualification["ptx_load_cache_policy"]
    require(loads[BASELINE] == {"global_load_instructions": 6, "cg_load_instructions": 0,
                                "ordinary_global_load_instructions": 6}, "Baseline PTX load evidence differs.")
    require(loads[CANDIDATE] == {"global_load_instructions": 6, "cg_load_instructions": 4,
                                 "ordinary_global_load_instructions": 2}, "Candidate PTX load evidence differs.")

    rows_per_burst = timing_plan["streams"] * timing_plan["pairs_per_stream"] * timing_plan["graph_replays_per_burst"] * plan["shape"]["rows"]
    require(rows_per_burst * timing_plan["bursts_per_period"] == timing_plan["period_normalized_rows"],
            "The fixed-work count differs from the plan.")
    periods, seeds, warmup_total = [], [], 0.0
    for seed in timing_plan["seeds"]:
        pair = []
        for index, backend in enumerate(timing_plan["orders"][str(seed)]):
            record = read(f"s{seed}-p{index}")
            require(record["gate_passed"] and record["seed"] == seed, f"Seed {seed} period {index} failed its gate.")
            require((record["code_revision"], record["source_sha256"], record["environment"], record["asset"])
                    == (revision, source_hashes, environment, asset), "Checkpoint provenance changed.")
            verify_exact(record)
            period = record["period"]
            require((period["index"], period["order_index"], period["backend"], period["order"])
                    == (index, index, backend, timing_plan["orders"][str(seed)]), "Backend order differs from the plan.")
            if index == 0:
                require(record["seed_fingerprints_match"] is None, "The first period should establish fingerprints.")
            else:
                require(record["seed_fingerprints_match"] is True
                        and record["source_fingerprints"] == pair[0]["source_fingerprints"], "Seed fingerprints differ.")
            timing = record["timing"]
            window = timing["window"]
            bursts = window["bursts"]
            require(len(bursts) == timing_plan["bursts_per_period"]
                    and window["normalized_rows"] == timing_plan["period_normalized_rows"], "Work count differs from the plan.")
            durations = [burst["end_s"] - burst["start_s"] for burst in bursts]
            require(all(seconds > 0 for seconds in durations), "A burst duration is not positive.")
            require(timing["warmup"]["wall_seconds"] >= timing_plan["minimum_warmup_s"], "Warmup is too short.")
            require(timing["outputs_match_after_timing"] and timing["inputs_unchanged_after_timing"], "Post-period checks failed.")
            warmup_total += timing["warmup"]["wall_seconds"]
            resource = timing["telemetry"]["by_backend"][backend]
            metrics = resource["metrics"]
            require(metrics["gpu_busy_pct"]["count"] > 0 and metrics["memory_busy_pct"]["count"] > 0, "Missing utilization samples.")
            periods.append({
                "seed": seed,
                "index": index,
                "backend": backend,
                "median_rows_per_second": statistics.median(rows_per_burst / seconds for seconds in durations),
                "aggregate_rows_per_second": window["normalized_rows"] / window["wall_seconds"],
                "wall_seconds": window["wall_seconds"],
                "cuda_seconds": window["cuda_ms"] / 1000,
                "warmup_seconds": timing["warmup"]["wall_seconds"],
                "normalized_rows": window["normalized_rows"],
                "nvml_samples": metrics["gpu_busy_pct"]["count"],
                "gpu_busy_median_pct": metrics["gpu_busy_pct"]["median"],
                "memory_busy_median_pct": metrics["memory_busy_pct"]["median"],
                "vram_median_mib": metrics["vram_used_mib"]["median"],
                "power_median_w": metrics["power_w"]["median"],
                "sm_clock_median_mhz": metrics["sm_clock_mhz"]["median"],
                "maximum_sample_gap_seconds": resource["maximum_gap_s"],
            })
            pair.append(record)
        rates = {row["backend"]: row["median_rows_per_second"] for row in periods if row["seed"] == seed}
        baseline, candidate = rates[BASELINE], rates[CANDIDATE]
        gain = candidate / baseline - 1
        seeds.append({
            "seed": seed,
            "order": timing_plan["orders"][str(seed)],
            "baseline_median_rows_per_second": baseline,
            "candidate_median_rows_per_second": candidate,
            "candidate_gain_fraction": gain,
            "candidate_gain_percent": 100 * gain,
            "candidate_passes_5_percent": candidate >= baseline * plan["adoption"]["minimum_throughput_ratio"],
        })

    attempts = json.loads((RESULTS / "attempts.json").read_text())
    require(len({row["attempt_id"] for row in attempts}) == len(attempts), "Duplicate attempt receipts were archived.")
    totals = {
        "timing_periods": len(periods),
        "normalized_rows": sum(row["normalized_rows"] for row in periods),
        "fixed_work_wall_seconds": sum(row["wall_seconds"] for row in periods),
        "warmup_seconds": warmup_total,
        "nvml_samples": sum(row["nvml_samples"] for row in periods),
        "median_gpu_busy_pct": statistics.median(row["gpu_busy_median_pct"] for row in periods),
        "median_memory_busy_pct": statistics.median(row["memory_busy_median_pct"] for row in periods),
        "vram_median_mib": statistics.median(row["vram_median_mib"] for row in periods),
        "median_power_range_w": [min(row["power_median_w"] for row in periods), max(row["power_median_w"] for row in periods)],
        "maximum_sample_gap_seconds": max(row["maximum_sample_gap_seconds"] for row in periods),
        "attempt_receipts": len(attempts),
    }
    adopted = all(row["candidate_passes_5_percent"] for row in seeds)
    require(not adopted, "The recorded decision unexpectedly meets the 5% rule.")
    summary = {
        "experiment": plan["id"],
        "candidate_adopted": adopted,
        "source_revision": revision,
        "source_sha256": source_hashes,
        "environment": environment,
        "qualification_seed": qualification["seed"],
        "qualification_gate_passed": True,
        "codegen": {
            "baseline": {key: base_code[key] for key in ("num_warps", "registers_per_thread", "spills", "binary_sha256", "ptx_instruction_sha256")},
            "candidate": {key: candidate_code[key] for key in ("num_warps", "registers_per_thread", "spills", "binary_sha256", "ptx_instruction_sha256")},
            "ptx_load_cache_policy": loads,
        },
        "per_seed": seeds,
        "periods": periods,
        "totals": totals,
        "pytorch_peak_allocated_bytes": qualification["pytorch_peak_allocated_bytes"],
        "pytorch_peak_reserved_bytes": qualification["pytorch_peak_reserved_bytes"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")

    report = [
        "# H100 input-load cache policy at 512 × 4,096 BF16",
        "",
        "**Decision: retain the eight-warp baseline.** The .cg candidate kept both outputs bitwise identical but did not reach the preregistered 5% median-throughput gain in any of three independent seeds.",
        "",
        "## Method",
        "",
        "The candidate changes only the activation and residual input-load cache hint. Each period repeats eight numerical fixtures and captured-graph checks against the pinned vLLM 0.10.0 operator, checks unchanged inputs and weights, warms up for at least three seconds, then records 512 bursts. Each period processes 2,147,483,648 normalized rows across eight streams and 32 pairs per stream.",
        "",
        "Median burst throughput is the median of 4,194,304 rows divided by each burst's wall time. The registered backend order is baseline then candidate in seeds 210 and 212, and candidate then baseline in seed 211.",
        "",
        "## Results",
        "",
        "| Seed | Order | Baseline median rows/s | Candidate median rows/s | Candidate gain | Exact inputs | Passes 5% |",
        "| ---: | --- | ---: | ---: | ---: | :---: | :---: |",
    ]
    for row in seeds:
        order = "baseline → candidate" if row["order"][0] == BASELINE else "candidate → baseline"
        report.append(
            f"| {row['seed']} | {order} | {row['baseline_median_rows_per_second']:,.0f} | "
            f"{row['candidate_median_rows_per_second']:,.0f} | {row['candidate_gain_percent']:+.6f}% | Yes | No |"
        )
    report.extend([
        "",
        "Every fixture and timing period passed both-output bitwise checks. Input and weight fingerprints matched within each seed. The candidate gained less than 0.01% in all three seeds, so this H100 workload provides no evidence to adopt .cg.",
        "",
        "## Compilation and numerical gates",
        "",
        "Seed 209 passed all eight fixtures and captured-graph checks. The eight-warp baseline reproduced the seed-205 anchor binary. Both kernels used eight warps and 32 registers per thread with zero spills. PTX contained six ordinary global-load instructions in the baseline, and four ld.global.cg input loads plus two ordinary weight loads in the candidate. The candidate binary differed from baseline.",
        "",
        "## H100 resource use",
        "",
        f"Device and software: {environment['gpu']}, compute capability {environment['compute_capability'][0]}.{environment['compute_capability'][1]}, driver {environment['driver']}, CUDA {environment['cuda']}, PyTorch {environment['torch']}, Triton {environment['triton']}, and Python {environment['python']}.",
        "",
        f"The six measured periods processed {totals['normalized_rows']:,} normalized rows in {totals['fixed_work_wall_seconds']:.3f} seconds, plus {totals['warmup_seconds']:.3f} seconds of warm-up. The observer recorded {totals['nvml_samples']} NVML samples. Median GPU busy and memory busy were both {totals['median_gpu_busy_pct']:.0f}%; median VRAM use was {totals['vram_median_mib']:,.2f} MiB. Per-period median power ranged from {totals['median_power_range_w'][0]:.3f} to {totals['median_power_range_w'][1]:.3f} W. The largest sample gap was {totals['maximum_sample_gap_seconds']:.3f} seconds.",
        "",
        f"Seed 209 qualification took {qualification['wall_seconds']:.3f} seconds of runner wall time. PyTorch peak allocated and reserved memory were {qualification['pytorch_peak_allocated_bytes'] / 1048576:,.2f} MiB and {qualification['pytorch_peak_reserved_bytes'] / 1048576:,.2f} MiB. These are qualification-run memory statistics; timed-period samples and auxiliary snapshots are stored in each raw checkpoint.",
        "",
        "## Audit trail",
        "",
        f"Frozen source revision: {revision}. Full compiler, fixture, fingerprint, burst, timing, resource, and snapshot records are in s209.json and s210-p0.json through s212-p1.json. Sanitized submission receipts are in attempts.json; derived rates and per-period resource summaries are in summary.json.",
        "",
        "Run python3 experiments/rmsnorm-residual/vllm/summarize_input_load_cache.py to verify source hashes, checkpoint identity, PTX gates, exactness, fingerprints, fixed-work counts, and derived statistics.",
        "",
        "The .cg load hint is described in the [Triton load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [NVIDIA PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#cache-operators).",
        "",
        "## Next hypothesis",
        "",
        "Keep the eight-warp baseline. Revalidate the existing 4,104-wide two-row RMSNorm recommendation at widths 4,103, 4,104, 4,105, and 4,112 with fresh seeds to see how local the measured gain is. See [issue #60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60) and [GOAL.md](../../../../GOAL.md).",
        "",
        "Tracking: [issue #58](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/58) and [draft PR #59](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/59).",
    ])
    (RESULTS / "REPORT.md").write_text("\n".join(report) + "\n")
    print("Verified six timing periods; gains: " + ", ".join(
        f"{row['seed']} {row['candidate_gain_percent']:+.6f}%" for row in seeds
    ))
    print("Candidate adopted:", adopted)


if __name__ == "__main__":
    main()
