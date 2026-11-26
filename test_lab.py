"""CPU-only CLI tests for checkpointing GPU runs. Run with python -m unittest test_lab."""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = """\
import json
import sys
import time
from pathlib import Path

directory = Path(__file__).parent
args = sys.argv[1:]
if args[0] == "check":
    unit = "check"
elif args[0] == "bench":
    unit = "s" + args[args.index("--seed") + 1]
elif args[0] == "gemm":
    fields = [args[args.index("--" + key) + 1] for key in ("seed", "m", "k", "n")]
    assert float(args[args.index("--seconds") + 1]) == 5
    unit = "s%s-m%s-k%s-n%s" % tuple(fields)
else:
    raise SystemExit("unknown action " + args[0])
with (directory / "calls").open("a") as stream:
    print(unit, file=stream)
while (directory / "hold").exists():
    time.sleep(0.02)
if (directory / ("fail-" + unit)).exists():
    raise SystemExit(17)
print((directory / (unit + ".json")).read_text())
"""


class LabCLI(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        for name in ("lab.py", "gemm_plan.py", "kernels.py", "check.py", "benchmark.py"):
            shutil.copy2(ROOT / name, self.directory / name)
        self.runner = self.directory / "fake-runner"
        self.runner.write_text(f"#!{sys.executable}\n" + RUNNER)
        self.runner.chmod(0o755)
        self.state = self.directory / "state"
        plan = {
            "id": "checkpoint-test", "hypothesis": "Test checkpoint recovery",
            "rows": [1], "widths": [4096], "seeds": [42, 43],
            "row_tiles": [1], "rounds": 2, "ops": 1, "compiled": False,
        }
        self.write_json("plan.json", plan)
        self.write_json("check.json", {
            "passed": True, "dtype": "bfloat16", "reference": "fp32", "seed": 42,
            "eps": 1e-6, "atol": 1e-6, "rtol": 5e-3,
            "checks": [{"passed": True, "max_abs_error": 0, "num_warps": warps,
                        "rows_per_program": tile, "rows": 1, "width": 4096, "input": "random"}
                       for tile in (1, 2, 4) for warps in (4, 8)],
        })
        for seed in plan["seeds"]:
            self.write_json(f"s{seed}.json", {
                "timestamp_utc": "2026-11-26T00:00:00+00:00", "code_revision": None,
                "environment": {
                    "gpu": "Test H100", "compute_capability": [9, 0], "driver": "550",
                    "cuda_runtime": "12.8", "torch": "2.7.1", "triton": "3.3.1", "python": "3.12",
                },
                "method": {
                    "seed": seed, "rounds": 2, "ops_per_graph": 1, "row_tiles": [1],
                    "dtype": "bfloat16", "eps": 1e-6, "backends_shuffled_each_round": True,
                },
                "results": [{"rows": 1, "width": 4096, "backend": name, "median_us": 1.0,
                             "q1_us": 1.0, "q3_us": 1.0, "samples_us": [1.0, 1.0]}
                            for name in ("torch_eager", "triton_4w", "triton_8w")],
            })
        self.cli("init", "--plan", str(self.directory / "plan.json"))

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def command(self, action, *args):
        command = [sys.executable, str(self.directory / "lab.py"), action, "--state-dir", str(self.state)]
        if action in ("run", "step"):
            command.extend(["--runner", str(self.runner)])
        return [*command, *args]

    def cli(self, action, *args, expect_success=True):
        result = subprocess.run(
            self.command(action, *args), cwd=self.directory, capture_output=True, text=True, timeout=10
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stderr + result.stdout)
        return result

    def calls(self):
        log = self.directory / "calls"
        return log.read_text().splitlines() if log.exists() else []

    def completed(self):
        return {p.name for p in (self.state / "tasks").glob("*.json")}

    def wait_for_calls(self, expected):
        deadline = time.monotonic() + 5
        while self.calls() != expected and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.calls(), expected)

    def test_failed_command_resumes_after_last_complete_unit(self):
        fault = self.directory / "fail-s43"
        fault.touch()
        self.cli("run", expect_success=False)
        self.assertEqual(self.calls(), ["check", "s42", "s43"])
        self.assertEqual(self.completed(), {"check.json", "s42-r1-w4096.json"})
        fault.unlink()
        self.cli("run")
        self.cli("run")
        self.assertEqual(self.calls(), ["check", "s42", "s43", "s43"])
        self.assertEqual(self.completed(), {"check.json", "s42-r1-w4096.json", "s43-r1-w4096.json"})

    def test_attempt_history_records_failures_and_retries(self):
        self.prepare_gemm()
        task = "s42-m2-k3-n4"
        fault = self.directory / ("fail-" + task)
        fault.touch()
        self.cli("step", expect_success=False)
        fault.unlink()
        self.cli("step")
        directory = self.state / "attempts"
        attempts = sorted((json.loads(p.read_text()) for p in directory.glob("*.json")),
                          key=lambda entry: entry["started_utc"])
        self.assertEqual([a["outcome"] for a in attempts], ["failed", "checkpoint_saved"])
        self.assertEqual(attempts[0]["exit_code"], 17)
        self.assertNotEqual(attempts[0]["attempt_id"], attempts[1]["attempt_id"])
        for attempt in attempts:
            self.assertEqual(attempt["task_id"], task)
            self.assertEqual(attempt["reproduce_argv"][:2],
                             ["python3", "experiments/prefill-gemm/benchmark.py"])
            self.assertLessEqual(attempt["started_utc"], attempt["finished_utc"])
            self.assertGreaterEqual(attempt["controller_elapsed_s"], 0)

    def test_bad_output_cannot_complete_a_unit(self):
        self.cli("step")
        fixture = self.directory / "s42.json"
        good = fixture.read_text()
        bad = json.loads(good)
        bad["results"][0]["samples_us"][0] = -1
        for payload in ("{partial", json.dumps(bad)):
            with self.subTest(payload=payload[:20]):
                fixture.write_text(payload)
                self.cli("step", expect_success=False)
                self.assertEqual(self.completed(), {"check.json"})
        fixture.write_text(good)
        self.cli("step")
        self.assertEqual(self.completed(), {"check.json", "s42-r1-w4096.json"})

    def test_duplicate_policy_cannot_replace_missing_correctness_evidence(self):
        path = self.directory / "check.json"
        result = json.loads(path.read_text())
        result["checks"][-1] = result["checks"][0]
        self.write_json("check.json", result)
        self.cli("step", expect_success=False)
        self.assertEqual(self.completed(), set())

    def prepare_gemm(self, local_source=None):
        self.state = self.directory / "gemm-state"
        script = Path("experiments/prefill-gemm/benchmark.py")
        (self.directory / script.parent).mkdir(parents=True)
        if local_source is None:
            shutil.copy2(ROOT / script, self.directory / script)
        else:
            (self.directory / script).write_text(local_source)
        shapes = [{"m": m, "k": 3, "n": 4} for m in (2, 4)]
        self.write_json("gemm.json", {
            "kind": "prefill_gemm", "id": "test-gemm", "hypothesis": "Compare two projections",
            "seeds": [42], "seconds": 5, "shapes": shapes,
        })
        for shape in shapes:
            m, k, n = shape.values()
            flops = 2 * m * k * n
            metrics = {"gpu_busy_pct": 90, "memory_busy_pct": 20, "device_memory_used_mib": 23,
                       "device_power_w": None, "sm_clock_mhz": 1800}
            self.write_json(f"s42-m{m}-k{k}-n{n}.json", {
                "environment": {"gpu": "Test GPU", "compute_capability": [9, 0], "driver": "550",
                                "cuda_runtime": "12.8", "torch": "2.7.1", "python": "3.12",
                                "total_memory_bytes": 80000000000, "sm_count": 132},
                "workload": {**shape, "dtype": "bfloat16", "seed": 42, "bf16_reduced_precision_reduction": False,
                             "operation": "torch.mm(A[M,K], B[K,N], out=C[M,N])",
                             "reuse": "same contiguous A, B and preallocated C throughout",
                             "operand_bytes": 2 * (m * k + k * n + m * n)},
                "correctness": {"passed": True, "reference": "FP32 dense products over full K, TF32 disabled",
                                "sampled_rows": list(range(m)), "sampled_columns": list(range(n)), "entries": m * n,
                                "rtol": 0.005, "atol": 2e-5 * math.sqrt(k), "max_absolute_error": 0.01},
                "timing": {"requested_seconds": 5, "warmup_s": 2, "warmup_iterations": 8,
                           "wall_elapsed_s": 5.25, "batch_iterations": 4, "iterations": 8,
                           "cuda_event_total_s": 4, "batches": [{"iterations": 4, "event_elapsed_s": 2}] * 2,
                           "flops_per_iteration": flops, "cuda_event_tflops": 8 * flops / 4 / 1e12,
                           "wall_tflops": 8 * flops / 5.25 / 1e12},
                "telemetry": {"samples": [{"query_start_s": 0.1, "query_end_s": 0.15, **metrics}],
                              "summary": {key: {"count": int(value is not None), "peak_observed": value, "mean": value}
                                          for key, value in metrics.items()}},
            })
        self.cli("init", "--plan", str(self.directory / "gemm.json"))

    @unittest.skipUnless(shutil.which("git"), "requires Git")
    def test_local_gemm_step_records_verified_source_revision(self):
        import hashlib
        from lab import local_gemm_revision

        self.prepare_gemm(local_source=(
            "import json, os\nfrom pathlib import Path\n"
            "result = json.loads(Path('s42-m2-k3-n4.json').read_text())\n"
            "result['code_revision'] = os.environ['BENCH_REVISION']\n"
            "print(json.dumps(result))\n"
        ))
        source = Path("experiments/prefill-gemm/benchmark.py")
        git = ["git", "-C", str(self.directory), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
               "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
        subprocess.run([*git, "init", "-q"], check=True)
        original = (self.directory / source).read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        self.assertEqual(local_gemm_revision(self.directory, digest), "unversioned")
        subprocess.run([*git, "add", "--", str(source)], check=True)
        subprocess.run([*git, "commit", "-qm", "Fixture"], check=True)
        head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        # Exercise the default local entry point with an unrelated value in the caller's environment.
        subprocess.run([sys.executable, str(self.directory / "lab.py"), "step", "--state-dir", str(self.state)],
                       cwd=self.directory, check=True, capture_output=True, env={**os.environ, "BENCH_REVISION": "wrong"})
        saved = json.loads((self.state / "tasks/s42-m2-k3-n4.json").read_text())
        self.assertEqual(saved["result"]["code_revision"], head)
        modified = original + b"# local changes\n"
        (self.directory / source).write_bytes(modified)
        self.assertEqual(local_gemm_revision(self.directory, hashlib.sha256(modified).hexdigest()), f"{head}-modified")

    def test_gemm_manifest_and_resume_multiple_shapes(self):
        self.prepare_gemm()
        self.assertIn("0/2 tasks completed", self.cli("status").stdout)
        manifest = json.loads((self.state / "manifest.json").read_text())
        self.assertEqual(set(manifest["sources"]), {"experiments/prefill-gemm/benchmark.py"})
        first, second = "s42-m2-k3-n4", "s42-m4-k3-n4"
        failure = self.directory / ("fail-" + second)
        failure.touch()
        self.cli("run", expect_success=False)
        self.assertEqual(self.completed(), {first + ".json"})
        failure.unlink()
        self.cli("run")
        self.cli("run")
        self.assertEqual(self.calls(), [first, second, second])
        self.assertIn("complete.", self.cli("status").stdout)

    def test_gemm_rejects_inconsistent_evidence(self):
        self.prepare_gemm()
        fixture = self.directory / "s42-m2-k3-n4.json"
        good = fixture.read_text()
        for section, key, value in (("correctness", "passed", False), ("workload", "k", 4),
                                    ("telemetry", "samples", []), ("timing", "wall_tflops", 8),
                                    ("timing", "batches", []), ("timing", "wall_elapsed_s", 4)):
            with self.subTest(key=key):
                bad = json.loads(good)
                bad[section][key] = value
                fixture.write_text(json.dumps(bad))
                self.cli("step", expect_success=False)
                self.assertEqual(self.completed(), set())
        fixture.write_text(good)
        self.cli("step")
        self.assertEqual(self.completed(), {fixture.name})

    def test_gemm_rejects_consistent_cuda_times_exceeding_wall_window(self):
        self.prepare_gemm()
        fixture = self.directory / "s42-m2-k3-n4.json"
        result = json.loads(fixture.read_text())
        timing = result["timing"]
        for batch in timing["batches"]:
            batch["event_elapsed_s"] *= 100
        timing["cuda_event_total_s"] *= 100
        timing["cuda_event_tflops"] /= 100
        fixture.write_text(json.dumps(result))
        failure = self.cli("step", expect_success=False)
        self.assertIn("exceeds the sequential wall window", failure.stderr)
        self.assertEqual(self.completed(), set())

    def test_interrupted_process_group_can_resume_pending_benchmark(self):
        self.cli("step")
        hold = self.directory / "hold"
        hold.touch()
        worker = subprocess.Popen(self.command("run"), cwd=self.directory,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
        try:
            self.wait_for_calls(["check", "s42"])
            # Ctrl+C in a terminal reaches both the coordinator and its GPU runner.
            os.killpg(worker.pid, signal.SIGINT)
            self.assertNotEqual(worker.wait(timeout=5), 0)
            self.assertEqual(self.completed(), {"check.json"})
        finally:
            hold.unlink(missing_ok=True)
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
                worker.wait(timeout=5)
        self.cli("run")
        self.assertEqual(self.calls(), ["check", "s42", "s42", "s43"])

    def test_pause_during_active_work_survives_process_restart(self):
        hold = self.directory / "hold"
        hold.touch()
        worker = subprocess.Popen(self.command("run"), cwd=self.directory, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        try:
            self.wait_for_calls(["check"])
            self.assertIn("running.", self.cli("status").stdout)
            self.cli("step", expect_success=False)
            self.assertEqual(self.calls(), ["check"])
            self.cli("pause")
            self.assertIn("pausing.", self.cli("status").stdout)
            hold.unlink()
            worker.communicate(timeout=5)
            self.assertIn("paused.", self.cli("status").stdout)
            self.assertEqual(self.calls(), ["check"])
            self.assertEqual(self.completed(), {"check.json"})
            # A new runner must honor the persisted pause regardless of its exit code.
            subprocess.run(self.command("run"), cwd=self.directory, capture_output=True, timeout=5)
            self.assertEqual(self.calls(), ["check"])
            self.cli("resume")
            self.assertIn("ready.", self.cli("status").stdout)
            self.cli("run")
            self.assertEqual(self.calls(), ["check", "s42", "s43"])
            self.assertIn("complete.", self.cli("status").stdout)
        finally:
            hold.unlink(missing_ok=True)
            if worker.poll() is None:
                worker.terminate()
            worker.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
