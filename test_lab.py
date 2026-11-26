"""CPU-only CLI tests for checkpointing GPU runs. Run with python -m unittest test_lab."""

import json
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
        for name in ("lab.py", "kernels.py", "check.py", "benchmark.py"):
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
