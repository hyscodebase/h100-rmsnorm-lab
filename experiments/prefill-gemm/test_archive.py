"""Offline archive checks using one published checkpoint and synthetic controller records."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import archive


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.state = root / "state"
        self.output = root / "archive"
        for folder in ("tasks", "attempts"):
            (self.state / folder).mkdir(parents=True)
        raw = Path(__file__).resolve().parent / "results"
        self.cell = next(iter(sorted(raw.glob("s*.json"))))
        shutil.copyfile(self.cell, self.state / "tasks" / self.cell.name)
        shutil.copyfile(raw / "manifest.json", self.state / "manifest.json")
        shutil.copyfile(archive.ROOT / "plans/prefill-gemm.json", self.state / "plan.json")
        plan = json.loads((self.state / "plan.json").read_text())
        task = self.cell.stem
        shape = dict(archive.lab.tasks(plan))[task]
        self.record = {
            "task_id": task, "attempt_id": "a" * 32,
            "experiment_hash": json.loads((self.state / "manifest.json").read_bytes())["experiment_hash"],
            "reproduce_argv": ["python3", archive.gemm_plan.SOURCE, *archive.gemm_plan.arguments(shape, plan)],
            "started_utc": "2026-01-01T00:00:00+00:00", "finished_utc": "2026-01-01T00:01:00+00:00",
            "controller_elapsed_s": 60.0, "outcome": "failed", "error_type": "CalledProcessError", "exit_code": 1,
        }
        self.attempt = self.state / "attempts" / f"{task}-{'a' * 32}.json"
        self.attempt.write_text(json.dumps(self.record))

    def publish(self):
        with contextlib.redirect_stdout(io.StringIO()):
            archive.snapshot(self.state, self.output)

    def test_verified_copy_retry_and_destination_conflict(self):
        self.publish()
        result = self.output / "results" / self.cell.name
        self.assertEqual(result.read_bytes(), self.cell.read_bytes())
        self.assertEqual(json.loads((self.output / "evidence/attempts" / self.attempt.name).read_text()), self.record)
        self.assertTrue((self.output / "evidence/run-index.csv").is_file())
        self.publish()
        result.write_text("existing data")
        with self.assertRaisesRegex(ValueError, "different snapshot"):
            self.publish()
        self.assertEqual(result.read_text(), "existing data")

    def test_unsafe_attempt_fails_before_publishing(self):
        for change in ({"reproduce_argv": self.record["reproduce_argv"] + ["private"]},
                       {"experiment_hash": "0" * 64}, {"stderr": "private"},
                       {"error_type": "PrivateIdentifier"}, {"started_utc": "private"}):
            with self.subTest(change=change):
                self.attempt.write_text(json.dumps({**self.record, **change}))
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(self.output.exists())

    def test_extra_raw_metadata_and_source_mismatch_rejected(self):
        path = self.state / "tasks" / self.cell.name
        data = json.loads(path.read_bytes())
        data["result"]["telemetry"]["samples"][0]["private_host"] = "private"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unexpected measurement fields"):
            self.publish()
        self.assertFalse(self.output.exists())
        path.write_bytes(self.cell.read_bytes())
        manifest = self.state / "manifest.json"
        metadata = json.loads(manifest.read_bytes())
        metadata["sources"][archive.gemm_plan.SOURCE] = "0" * 64
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "source manifest"):
            self.publish()
        self.assertFalse(self.output.exists())

    def test_accepts_failed_only_and_omits_in_progress_attempt(self):
        (self.state / "tasks" / self.cell.name).unlink()
        pending = {k: v for k, v in self.record.items() if k not in (
            "finished_utc", "controller_elapsed_s", "exit_code", "error_type")}
        pending.update(attempt_id="b"*32, outcome="started")
        (self.state / "attempts" / f"{self.cell.stem}-{'b'*32}.json").write_text(json.dumps(pending))
        self.publish()
        self.assertEqual([self.attempt.name], [p.name for p in (self.output / "evidence/attempts").iterdir()])
        self.assertFalse((self.output / "evidence/run-index.csv").exists())


if __name__ == "__main__":
    unittest.main()
