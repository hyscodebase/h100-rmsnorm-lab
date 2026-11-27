"""Audit raw H100 checkpoints and regenerate the width-neighborhood report."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "plans/width-neighborhood.json"
RAW = ROOT / "results/width-neighborhood/raw"
RESULTS = RAW / "checkpoints"
SUMMARY = RAW / "summary.json"
REPORT = ROOT / "results/width-neighborhood/REPORT.md"
SOURCES = ("kernels.py", "experiments/width-neighborhood/run.py", "experiments/width-neighborhood/PLAN.md", "plans/width-neighborhood.json")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def median(values):
    return statistics.median(values)


def audit():
    plan = json.loads(PLAN_PATH.read_text())
    need(plan["issue"] == 60 and plan["rows"] == 1024 and plan["widths"] == [4103, 4104, 4105, 4112]
         and plan["seeds"] == [44, 45, 46] and plan["candidate"] == "triton_4w_2r"
         and plan["rounds"] == 21 and plan["ops_per_graph"] == 32 and plan["threshold_percent"] == 5.0,
         "Frozen plan differs from issue #60.")
    source_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}
    expected = {(seed, width) for seed in plan["seeds"] for width in plan["widths"]}
    files = {path.stem: path for path in RESULTS.glob("s*-w*.json")}
    expected_names = {f"s{seed}-w{width}" for seed, width in expected}
    need(set(files) == expected_names, f"Checkpoint coverage differs. Missing {sorted(expected_names-set(files))}; extra {sorted(set(files)-expected_names)}")
    cells = {}
    environment = None
    revision = None
    telemetry = []
    for seed, width in sorted(expected):
        key = f"s{seed}-w{width}"
        record = json.loads(files[key].read_text())
        need((record.get("seed"), record.get("rows"), record.get("width")) == (seed, 1024, width), f"Wrong identity in {key}.")
        need(record.get("research_assistant_model") == "GPT-6 Luna (delegated runtime)",
             f"Missing GPT-6 Luna execution metadata in {key}.")
        need(record.get("source_sha256") == source_hashes, f"Source hash mismatch in {key}.")
        need(isinstance(record.get("code_revision"), str) and len(record["code_revision"]) == 40,
             f"Missing frozen revision in {key}.")
        if revision is None:
            revision = record["code_revision"]
        need(record["code_revision"] == revision, f"Mixed source revisions in {key}.")
        env = record["environment"]
        need("H100" in env["gpu"] and env["compute_capability"] == [9, 0] and env["total_memory_bytes"] > 0,
             f"Non-H100 execution in {key}.")
        need(environment in (None, env), f"GPU environment changed in {key}.")
        environment = env
        method = record["method"]
        need((method["rounds"], method["ops_per_graph"], method["eps"], method["timed_backends"])
             == (21, 32, 1e-6, ["triton_4w_1r", "triton_8w_1r", "triton_4w_2r", "torch_compile"]),
             f"Method differs in {key}.")
        correctness = record["correctness"]
        need(correctness["strict_gate"] == {"rtol": 0.005, "atol": 1e-6, "passed": True},
             f"Strict correctness gate failed in {key}.")
        need(correctness["benchmark_crosscheck"] == {"rtol": 0.016, "atol": 1e-5, "passed": True},
             f"Benchmark correctness cross-check failed in {key}.")
        fixtures = correctness["fixtures"]
        need(len(fixtures) == 32 and all(f["strict_passed"] and f["benchmark_passed"] for f in fixtures),
             f"Incomplete or failed numerical fixtures in {key}.")
        timing_correctness = correctness["timing_input"]
        need(len(timing_correctness) == 4 and all(c["strict_passed"] for c in timing_correctness),
             f"Timed input failed strict correctness in {key}.")
        samples = record["timing_intervals"]
        need(len(samples) == 21 * 4, f"Wrong interval count in {key}.")
        by_round = {}
        backend_times = {}
        for sample in samples:
            by_round.setdefault(sample["round"], []).append(sample)
            need(sample["ops_per_graph"] == 32 and sample["graph_elapsed_us"] > 0 and sample["us_per_op"] > 0,
                 f"Invalid raw event interval in {key}.")
            backend_times.setdefault(sample["backend"], []).append(sample["us_per_op"])
        need(set(by_round) == set(range(21)), f"Missing timing rounds in {key}.")
        for round_id, group in by_round.items():
            need(len(group) == 4 and {g["backend"] for g in group} == set(method["timed_backends"])
                 and sorted(g["order_index"] for g in group) == [0, 1, 2, 3],
                 f"Bad order record in {key}, round {round_id}.")
        medians = {name: median(values) for name, values in backend_times.items()}
        one_row = min(medians["triton_4w_1r"], medians["triton_8w_1r"])
        candidate = medians["triton_4w_2r"]
        cells[seed, width] = {
            "seed": seed, "width": width, "backend_median_us": medians,
            "fastest_one_row_us": one_row, "candidate_us": candidate,
            "torch_compile_us": medians["torch_compile"],
            "gain_vs_fastest_one_row_percent": 100 * (1 - candidate / one_row),
            "gain_vs_torch_compile_percent": 100 * (1 - candidate / medians["torch_compile"]),
            "wall_seconds": record["wall_seconds"],
            "telemetry": record.get("telemetry", []),
        }
        for item in record.get("telemetry", []):
            path = ROOT / item["path"]
            need(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
                 f"Telemetry hash mismatch in {key}.")
            for line in path.read_text().splitlines():
                sample = json.loads(line)
                telemetry.append(sample)
    by_width = {}
    for width in plan["widths"]:
        rows = [cells[seed, width] for seed in plan["seeds"]]
        qualifies = all(r["gain_vs_fastest_one_row_percent"] >= 5 and r["gain_vs_torch_compile_percent"] >= 5 for r in rows)
        by_width[str(width)] = {"per_seed": rows, "qualifies": qualifies}
    qualifies_4104 = by_width["4104"]["qualifies"]
    if telemetry:
        stamps = [s["timestamp_utc"] for s in telemetry]
        parsed = sorted(__import__("datetime").datetime.fromisoformat(s) for s in stamps)
        gaps = [(b-a).total_seconds() for a, b in zip(parsed, parsed[1:])]
        def finite(key): return [s[key] for s in telemetry if s.get(key) is not None]
        resources = {"sample_count": len(telemetry), "span_seconds": (parsed[-1]-parsed[0]).total_seconds(),
                     "max_gap_seconds": max(gaps, default=0),
                     "gpu_util_median_percent": median(finite("util_pct")) if finite("util_pct") else None,
                     "memory_busy_median_percent": median(finite("memory_busy_pct")) if finite("memory_busy_pct") else None,
                     "vram_median_mib": median(finite("used_mib")) if finite("used_mib") else None,
                     "power_median_w": median(finite("power_w")) if finite("power_w") else None}
    else:
        resources = {"sample_count": 0, "span_seconds": 0, "max_gap_seconds": None,
                     "gpu_util_median_percent": None, "memory_busy_median_percent": None,
                     "vram_median_mib": None, "power_median_w": None}
    for path, digest in source_hashes.items():
        blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{path}"])
        need(hashlib.sha256(blob).hexdigest() == digest, f"Frozen source hash differs in {revision}:{path}.")
    summary = {"plan": plan, "code_revision": revision, "source_sha256": source_hashes,
               "environment": environment, "widths": by_width, "recommend_width_4104": qualifies_4104,
               "checkpoints": len(expected), "timing_intervals": len(expected)*21*4,
               "runner_wall_seconds": sum(r["wall_seconds"] for r in cells.values()), "resources": resources}
    return summary


def write_report(summary):
    plan = summary["plan"]
    env = summary["environment"]
    lines = ["# RMSNorm width neighborhood on H100", "",
             "Issue [#60](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/60). Rows: 1,024 BF16. All checkpoints passed the strict FP32 gate (`rtol=0.005`, `atol=1e-6`) and width-sweep cross-check (`rtol=0.016`, `atol=1e-5`). The candidate is fixed at two rows/program and four warps.", "",
             f"Runtime: {env['gpu']} (compute capability {env['compute_capability']}); CUDA {env['cuda_runtime']}; PyTorch {env['torch']}; Triton {env['triton']}; driver {env['driver']}; Python {env['python']}.", "",
             "Each cell below shows the 44/45/46 seed gains. The one-row comparator is chosen as the faster of the four-warp and eight-warp variants within each same-seed checkpoint.", "",
             "| Width | Candidate µs (44 / 45 / 46) | Fastest one-row µs | Torch compile µs | Gain vs one-row | Gain vs torch compile | ≥5% in all seeds vs both? |",
             "|---:|---:|---:|---:|---:|---:|:---:|"]
    for width in plan["widths"]:
        rows = summary["widths"][str(width)]["per_seed"]
        def group(field): return " / ".join(f"{r[field]:.3f}" for r in rows)
        lines.append(f"| {width} | {group('candidate_us')} | {group('fastest_one_row_us')} | {group('torch_compile_us')} | {group('gain_vs_fastest_one_row_percent')}% | {group('gain_vs_torch_compile_percent')}% | {'Yes' if summary['widths'][str(width)]['qualifies'] else 'No'} |")
    res = summary["resources"]
    lines += ["", f"**Decision:** {'Retain' if summary['recommend_width_4104'] else 'Reject'} the width 4,104 recommendation under the preregistered rule; all three seeds must beat both references by at least 5%. Neighbor results do not broaden the recommendation.", "",
              f"**H100 observations:** {res['sample_count']} one-second device samples across {res['span_seconds']:.1f}s of observed span; median GPU busy {res['gpu_util_median_percent']}%; median memory busy {res['memory_busy_median_percent']}%; median VRAM {res['vram_median_mib']} MiB; median power {res['power_median_w']} W; largest telemetry gap {res['max_gap_seconds']}s. Host runner wall across 12 checkpoints: {summary['runner_wall_seconds']:.1f}s. Device sample span is not billed GPU time.", "",
              f"Raw archive: [`raw/`](raw). Audited source revision: `{summary['code_revision']}`. The machine-readable derivation is [`summary.json`](raw/summary.json).", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    data = audit()
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    write_report(data)
    print(f"Verified {data['checkpoints']} checkpoints / {data['timing_intervals']} intervals; 4104 recommendation={data['recommend_width_4104']}.")
