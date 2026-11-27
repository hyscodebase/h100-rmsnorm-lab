"""Run a bounded number of registered units and optionally publish each audited result."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/width-dispatch"))
import step

BRANCH = "codex/width-dispatch-issue66"
REPORT = ROOT / "results/width-dispatch/PROGRESS.md"
AUDIT = step.RAW / "partial-audit.json"


def write_report(audit):
    records = audit["checkpoint_results"]
    raw = [json.loads((ROOT / r["raw_path"]).read_text()) for r in records]
    phases = [p for record in raw for p in record.get("sustained_phases", [])]
    samples = [s for p in phases for s in p["telemetry_samples"]]
    complete = audit["successful_checkpoints"]
    expected = audit["expected_checkpoints"]
    eligible = sum(p["activity_telemetry_eligible"] for p in phases)
    seconds = sum(p["event_seconds"] for p in phases)
    last = max(r["finished_utc"] for r in raw)
    submitted = sum(audit["submitted_by_unit"].values())
    rows = [
        "# H100 width dispatch measurements", "",
        f"{complete}/{expected} registered units completed at 32,768 BF16 rows. Latest saved measurement (UTC) `{last}`.",
        f"The {submitted} submitted attempts include the two early SSH failures and the initial driver preflight failure at width 4107.",
        "All completed units passed the numerical, replay and memory checks.", "",
        f"**{seconds / 60:.2f} minutes** of sustained CUDA-event measurements, {len(samples):,} device samples, "
        f"and **{eligible}/{len(phases)}** phases eligible for activity estimates.",
    ]
    busy = [s["gpu_busy_percent"] for s in samples if s.get("gpu_busy_percent") is not None]
    power = [s["power_w"] for s in samples if s.get("power_w") is not None]
    vram = [s["memory_used_mib"] for s in samples if s.get("memory_used_mib") is not None]
    temp = [s["temperature_c"] for s in samples if s.get("temperature_c") is not None]
    if all((busy, power, vram, temp)):
        rows.append(f"Recorded samples have median/peak GPU busy {statistics.median(busy):.1f}/{max(busy):.1f}%, "
                    f"median/peak power {statistics.median(power):.1f}/{max(power):.1f} W, "
                    f"peak VRAM {max(vram):,.0f} MiB and peak temperature {max(temp):.0f} °C. "
                    "Sparse traces are retained as diagnostics. Each formal activity claim requires the phase coverage audit.")
    rows += ["", "## Paired measurements", "",
             "Rates are logical input plus output GB/s. Gains compare the four-row candidate with each reference from the same run.", "",
             "| Seed / width | 4-row GB/s | Best 1-row GB/s | compile GB/s | vs 1-row | vs compile | Both ≥5% |",
             "| --- | ---: | ---: | ---: | ---: | ---: | :---: |"]
    by_unit = {r["checkpoint"]: r for r in records}
    for c in audit["checkpoint_comparisons"]:
        unit = f"s{c['seed']}-w{c['width']}"
        link = (ROOT / by_unit[unit]["raw_path"]).relative_to(REPORT.parent)
        rates, gains = c["rates_GB_per_s"], c["gain_percent"]
        rows.append(f"| [{c['seed']} / {c['width']}]({link}) | {rates['triton_4w_4r']:,.2f} | "
                    f"{rates[c['fastest_one_row_backend']]:,.2f} | {rates['torch_compile']:,.2f} | "
                    f"{gains['fastest_one_row']:+.2f}% | {gains['torch_compile']:+.2f}% | {'yes' if c['qualifies'] else 'no'} |")
    rows += ["", "## Decision", ""]
    if audit["final_complete"]:
        selected = [w for w, v in audit["per_width_across_seeds"].items() if v["all_registered_seeds_qualify"]]
        rows.append("Widths clearing both references in all three seeds: " + (", ".join(selected) or "none") + ".")
        decision = audit["bounded_dispatch_screen_decision"]
        rows.append(f"Registered odd/even split reproduced on these eight widths: {decision['supports_registered_odd_even_split']}.")
    elif any(r["status"] != "complete" for r in records):
        for r in records:
            if r["status"] != "complete":
                rows.append(f"Stopped at `{r['checkpoint']}` during `{r['failure']['stage']}`. {r['failure']['message']}")
    else:
        next_unit = next(f"s{s}-w{w}" for s, w in step.work_units() if f"s{s}-w{w}" not in by_unit)
        rows.append(f"Next registered unit: `{next_unit}`. A width qualifies after the candidate beats both references by at least 5% in all three seeds.")
    revisions = Counter(r["source_revision"][:12] for r in records if r["status"] == "complete")
    rows += ["", "## Evidence", "",
             "Accepted units by benchmark revision: " + ", ".join(f"`{rev}` ({n})" for rev, n in revisions.items()) + ".",
             "The first width-4107 attempt stopped before any measured work. Its original archive and the resumed measurement have separate source and run identities.",
             "[Plan](../../experiments/width-dispatch/PLAN.md) · [Continuation and attempt rules](../../experiments/width-dispatch/RESUME.md) · "
             "[Original preflight failure](raw/archives/driver-preflight-d42481a/checkpoints/s53-w4107.json) · "
             "[Current audit](raw/partial-audit.json) · [PR #67](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/67)", ""]
    with tempfile.NamedTemporaryFile(dir=REPORT.parent, mode="w", encoding="utf-8", delete=False) as temp:
        temp.write("\n".join(rows))
    os.replace(temp.name, REPORT)


def git(*args, **kwargs):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, **kwargs)


def verify_destination():
    branch = git("branch", "--show-current", capture_output=True).stdout.strip()
    url = git("remote", "get-url", "--push", "origin", capture_output=True).stdout.strip().removesuffix(".git")
    if branch != BRANCH or url not in {"https://github.com/hyscodebase/h100-rmsnorm-lab",
                                       "git@github.com:hyscodebase/h100-rmsnorm-lab"}:
        raise SystemExit("Automatic publication requires the registered portfolio remote and PR branch.")


def publish_saved(audit, title):
    """Include every closed receipt and measured unit that passed this audit."""
    verify_destination()
    paths = sorted(step.ATTEMPTS.glob("*.json"))
    paths.extend(ROOT / item["raw_path"] for item in audit["checkpoint_results"]
                 if item["source_revision"] == audit["source_revision"])
    paths.append(REPORT)
    paths = [str(path.relative_to(ROOT)) for path in paths]
    if git("status", "--porcelain", "--", *paths, capture_output=True).stdout:
        paths.append(str(AUDIT.relative_to(ROOT)))
        git("add", "--", *paths)
        git("commit", "--only", "-m", title, "--", *paths)
    git("push", "origin", f"HEAD:refs/heads/{BRANCH}", timeout=45,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def run_unit(publish):
    with step.lab.worker_lock(step.STATE) as lock:
        if step.is_paused():
            print("Paused at a checkpoint boundary.", flush=True)
            return False
        if publish:
            verify_destination()
        before = set(step.ATTEMPTS.glob("*.json"))
        try:
            step.run_one(lock)
        finally:
            new = sorted(set(step.ATTEMPTS.glob("*.json")) - before)
            if new:
                audit = step.refresh_partial_audit()
                write_report(audit)
                if publish:
                    if len(new) != 1:
                        raise SystemExit("Unexpected concurrent receipts. Inspect the evidence before publication.")
                    receipt = json.loads(new[0].read_text())
                    publish_saved(audit, f"Record H100 {receipt['unit']} ({receipt['outcome']})")
                print(f"Audited {audit['successful_checkpoints']}/{audit['expected_checkpoints']} units.", flush=True)
        if not new:
            return False
        return not audit["final_complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--publish", action="store_true", help="Commit each audited unit and push it to the fixed PR branch.")
    parser.add_argument("--report-only", action="store_true", help="Update the local report from the last saved audit.")
    args = parser.parse_args()
    if args.report_only:
        write_report(json.loads(AUDIT.read_text()))
        return
    if not 1 <= args.max_steps <= len(step.work_units()):
        parser.error("Choose between one step and the registered matrix size.")
    step.STATE.mkdir(parents=True, exist_ok=True)
    if args.publish:
        with step.lab.worker_lock(step.STATE):
            verify_destination()
            step.reconcile_dangling_attempts(step.read_saved_checkpoints())
            audit = step.refresh_partial_audit()
            write_report(audit)
            publish_saved(audit, "Publish outstanding audited H100 evidence")
    for _ in range(args.max_steps):
        if not run_unit(args.publish):
            break


if __name__ == "__main__":
    main()
