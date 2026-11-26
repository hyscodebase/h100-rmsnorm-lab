"""Offline observer archive regression checks using generated controller envelopes."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import archive


class ObserverArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.state, self.out = base / "state", base / "export"
        (self.state / "tasks").mkdir(parents=True)
        (self.state / "attempts").mkdir()
        plan = archive.lab.read_json(archive.ROOT / "plans/observer-coverage.json")
        self.rev = "a" * 40
        manifest = archive.lab.manifest(plan)
        self.digest = manifest["experiment_hash"]
        for name, data in (("observer-revision.json", self.rev), ("plan.json", plan), ("manifest.json", manifest)):
            archive.save(self.state / name, data)
        raw = archive.ROOT / "experiments/prefill-gemm/results/s44-m8192-k4096-n11008.json"
        self.compute = json.loads(raw.read_text())["result"]
        del self.compute["telemetry"]
        env = self.compute["environment"]
        env["observer_versions"] = {"binding_version": "nvml-ctypes-v1", "nvml_version": "12.1",
                                     "driver_version": env["driver"], "nvidia_smi_version": env["driver"]}
        self.compute.update(seed=51, arm="A", code_revision=self.rev,
                            observers={"cli": [{"slot": i, "status": "skipped", "query_start_s": None, "query_end_s": None,
                                                "values": dict.fromkeys(archive.analyze.METRICS)} for i in range(45)]})
        self.compute["workload"]["seed"] = 51
        self.preflight = {"preflight": True, "passed": True, "physical_uuid_matched": True, "environment": deepcopy(env),
                          "code_revision": self.rev, "checks": {k: {"passed": True, "values": {
                              field: 5 if field == "gpu_busy_pct" else None for field in archive.analyze.METRICS}}
                              for k in ("cli", "nvml")}}
        verifier = patch.object(archive.lab, "local_gemm_revision", return_value=self.rev)
        self.verify = verifier.start()
        self.addCleanup(verifier.stop)
        self.put("preflight", self.preflight)

    def put(self, name, value):
        archive.save(self.state / "tasks" / f"{name}.json", {"task_id": name, "experiment_hash": self.digest,
                     "completed_utc": "2026-11-27T12:00:00+00:00", "result": value})

    def test_incremental_exchange_and_conflict_preservation(self):
        self.assertEqual(archive.snapshot(self.state, self.out), (1, 0))
        self.assertEqual(self.verify.call_count, 3)
        self.assertTrue((self.out / "preflight.json").exists())
        self.assertEqual(archive.snapshot(self.state, self.out), (1, 0))
        self.put("s51-A", self.compute)
        self.assertEqual(archive.snapshot(self.state, self.out), (2, 0))
        self.assertEqual(len(archive.analyze.load(self.out)), 1)
        original = (self.out / "preflight.json").read_bytes()
        self.preflight["checks"]["cli"]["values"]["gpu_busy_pct"] = 10
        self.put("preflight", self.preflight)
        with self.assertRaisesRegex(ValueError, "conflict"):
            archive.snapshot(self.state, self.out)
        self.assertEqual((self.out / "preflight.json").read_bytes(), original)

    def test_schedule_source_and_sensitive_fields_rejected(self):
        self.put("s51-B", self.compute)
        with self.assertRaisesRegex(ValueError, "prefix"):
            archive.snapshot(self.state, self.out)
        (self.state / "tasks/s51-B.json").unlink()
        for key, value in (("gpu", "NVIDIA H100 GPU-12345678-1234-1234-1234-123456789abc"),
                           ("python", "/home/private/python")):
            broken = deepcopy(self.preflight)
            broken["environment"][key] = value
            self.put("preflight", broken)
            with self.assertRaises(ValueError):
                archive.snapshot(self.state, self.out)
        self.verify.return_value = self.rev + "-modified"
        with self.assertRaisesRegex(ValueError, "three source blobs"):
            archive.snapshot(self.state, self.out)
        self.assertFalse(self.out.exists())

    def test_terminal_attempts_only(self):
        failed = {"attempt_id": "b"*32, "task_id": "s51-A", "experiment_hash": self.digest,
                  "started_utc": "2026-11-27T12:00:00+00:00", "finished_utc": "2026-11-27T12:00:10+00:00",
                  "controller_elapsed_s": 10.0, "outcome": "failed", "error_type": "CalledProcessError", "exit_code": 1,
                  "reproduce_argv": ["python3", archive.observer_plan.SOURCE, "--seed", "51", "--arm", "A"]}
        file = self.state / "attempts" / f"s51-A-{'b'*32}.json"
        archive.save(file, failed)
        self.assertEqual(archive.snapshot(self.state, self.out), (1, 1))
        started = {k:v for k,v in failed.items() if k not in ("controller_elapsed_s", "finished_utc", "exit_code", "error_type")}
        started.update(attempt_id="c"*32, outcome="started")
        archive.save(self.state / "attempts" / f"s51-A-{'c'*32}.json", started)
        self.assertEqual(archive.snapshot(self.state, self.out), (1, 1))
        failed["reproduce_argv"].append("/home/private")
        archive.save(file, failed)
        with self.assertRaisesRegex(ValueError, "arguments"):
            archive.snapshot(self.state, self.out)


if __name__ == "__main__":
    unittest.main()
