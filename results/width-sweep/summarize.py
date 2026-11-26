"""Rebuild the width-sweep decision table from the archived H100 checkpoints."""
import hashlib
import json
import re
import statistics
import subprocess
import sys
from decimal import Decimal
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONE_ROW = ("triton_4w", "triton_8w")
CANDIDATES = tuple(f"triton_{warps}w_{tile}r" for warps, tile in product((4, 8), (2, 4)))
BACKENDS = {*ONE_ROW, *CANDIDATES, "torch_eager", "torch_compile"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_revision(revision, sources):
    require(isinstance(revision, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision),
            f"Expected a full, clean Git commit ID, got {revision!r}.")
    git = ["git", "--no-replace-objects", "-C", str(HERE.parents[1])]
    kind = subprocess.run([*git, "cat-file", "-t", revision], capture_output=True)
    require(kind.returncode == 0,
            f"Missing Git commit {revision}. Run in a Git checkout with the experiment history. "
            "For a shallow clone, run 'git fetch --unshallow origin' and retry.")
    require(kind.stdout.strip() == b"commit", f"Recorded revision {revision} is not a commit.")
    for path, digest in sources.items():
        blob = subprocess.run([*git, "show", f"{revision}:{path}"], capture_output=True)
        require(blob.returncode == 0, f"Cannot read {path} from Git commit {revision}.")
        require(hashlib.sha256(blob.stdout).hexdigest() == digest, f"Source hash mismatch for {path} at {revision}.")


def load_archive(plan, raw):
    manifest = json.loads((raw / "manifest.json").read_text())
    sources = manifest["sources"]
    require(manifest["version"] == 1 and set(sources) == {"kernels.py", "check.py", "benchmark.py"}
            and all(re.fullmatch("[0-9a-f]{64}", value) for value in sources.values()), "Invalid source manifest.")
    digest = hashlib.sha256(json.dumps([plan, sources], sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(manifest["experiment_hash"] == digest, "Frozen source hashes and plan do not match the manifest.")
    require(plan["rows"] == [1024] and plan["seeds"] == [42, 43] and plan["rounds"] == 21
            and plan["ops"] == 32 and plan["compiled"] is True and plan["row_tiles"] == [1, 2, 4],
            "Unexpected width-sweep design.")
    require(len(set(plan["widths"])) == len(plan["widths"]) > 0, "Invalid width list.")
    cells = {f"s{seed}-r1024-w{width}": (seed, width) for seed, width in product(plan["seeds"], plan["widths"])}
    expected = {"check", *cells}
    observed = {p.stem for p in raw.glob("*.json") if p.name != "manifest.json"}
    require(observed == expected, f"Checkpoint mismatch: missing {sorted(expected-observed)}, extra {sorted(observed-expected)}.")

    def read_task(name):
        data = json.loads((raw / f"{name}.json").read_text(), parse_float=Decimal)
        require(data["task_id"] == name and data["experiment_hash"] == digest, f"Provenance mismatch in {name}.")
        return data["result"]

    check = read_task("check")
    checks = check["checks"]
    check_settings = {"seed": 42, "eps": Decimal("1e-6"), "rtol": Decimal("0.005"), "atol": Decimal("1e-6")}
    require(all(check.get(k) == v for k, v in check_settings.items()), "Correctness seed or tolerances differ.")
    require(check["passed"] is True and check["dtype"] == "bfloat16" and check["reference"] == "fp32"
            and len(checks) == 60 and all(c["passed"] is True and Decimal(c["max_abs_error"]).is_finite()
                                        and c["max_abs_error"] >= 0 for c in checks), "Correctness check failed.")
    cases = {(c["rows"], c["width"], c["input"]) for c in checks}
    policies = {(c["rows"], c["width"], c["input"], c["rows_per_program"], c["num_warps"]) for c in checks}
    require(len(cases) == 10 and policies == {(*case, tile, warps) for case, tile, warps in product(cases, (1, 2, 4), (4, 8))},
            "Missing or duplicate correctness case.")

    medians = {}
    environment = None
    verified_revisions = set()
    for name, (seed, width) in cells.items():
        result = read_task(name)
        revision = result.get("code_revision")
        if revision not in verified_revisions:
            verify_revision(revision, sources)
            verified_revisions.add(revision)
        env = result["environment"]
        require("H100" in env["gpu"] and env["compute_capability"] == [9, 0], f"Expected H100 evidence in {name}.")
        require(environment in (None, env), f"Inconsistent GPU environment in {name}.")
        environment = env
        method = result["method"]
        expected_method = {"seed": seed, "row_tiles": plan["row_tiles"], "rounds": plan["rounds"],
                           "ops_per_graph": plan["ops"], "dtype": "bfloat16", "backends_shuffled_each_round": True,
                           "eps": Decimal("1e-6"), "cache": "same input addresses; no explicit L2 flush",
                           "timing": "CUDA events around one graph replay, microseconds per operation",
                           "reference": "FP32 RMSNorm, rtol=0.016, atol=1e-5"}
        require(all(method.get(k) == v for k, v in expected_method.items()), f"Method mismatch in {name}.")
        entries = result["results"]
        require(len(entries) == len(BACKENDS) and {r["backend"] for r in entries} == BACKENDS,
                f"Backend mismatch in {name}.")
        times = {}
        for entry in entries:
            samples = entry["samples_us"]
            require(entry["width"] == width and entry["rows"] == 1024 and len(samples) == plan["rounds"]
                    and all(type(v) in (int, Decimal) and Decimal(v).is_finite() and v > 0 for v in samples),
                    f"Invalid shape or samples in {name}.")
            median = statistics.median(samples)
            require(median == entry["median_us"], f"Stored median differs from raw samples in {name}.")
            times[entry["backend"]] = median
        medians[width, seed] = times
    return medians


def table(plan, medians):
    print("Each pair lists seed 42 then seed 43. Positive reductions mean lower latency. "
          "The displayed candidate maximizes its smallest reduction across both references and both seeds.")
    print()
    print("| Width | Fixed candidate | Candidate µs | Best one-row µs | Compiled PyTorch µs | "
          "Reduction vs one-row | Reduction vs compiled | All candidates passing 5% |")
    print("|---:|---|---:|---:|---:|---:|---:|---|")
    for width in plan["widths"]:
        runs = [medians[width, seed] for seed in plan["seeds"]]
        one_row = [min(run[k] for k in ONE_ROW) for run in runs]
        compiled = [run["torch_compile"] for run in runs]
        reductions = {}
        passing = []
        for candidate in CANDIDATES:
            candidate_us = [run[candidate] for run in runs]
            reductions[candidate] = [
                [100 * (1 - latency / reference) for latency, reference in zip(candidate_us, reference_pair)]
                for reference_pair in (one_row, compiled)
            ]
            if all(100 * t <= 95 * ref for refs in (one_row, compiled) for t, ref in zip(candidate_us, refs)):
                passing.append(candidate)
        best = max(CANDIDATES, key=lambda k: min(min(pair) for pair in reductions[k]))
        fields = [str(width), best]
        fields.extend(" / ".join(f"{v:.3f}" for v in pair) for pair in ([run[best] for run in runs], one_row, compiled))
        fields.extend(" / ".join(f"{v:+.2f}%" for v in pair) for pair in reductions[best])
        fields.append(", ".join(passing) or "None")
        print("| " + " | ".join(fields) + " |")


if __name__ == "__main__":
    try:
        plan = json.loads((HERE.parents[1] / "plans/width-sweep.json").read_text())
        table(plan, load_archive(plan, HERE / "raw"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        sys.exit(f"Width-sweep summary failed: {exc}")
