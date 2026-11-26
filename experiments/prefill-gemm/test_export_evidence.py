"""CPU-only regression checks for export boundaries and public sidecar privacy."""
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import export_evidence as evidence


class EvidenceExport(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.results, self.logs, self.output = (root / name for name in ("results", "logs", "evidence"))
        self.results.mkdir()
        self.logs.mkdir()
        archive = Path(__file__).resolve().parent / "results"
        checkpoint = next(iter(sorted(archive.glob("s*.json"))))
        shutil.copy2(checkpoint, self.results / checkpoint.name)
        shutil.copy2(archive / "manifest.json", self.results / "manifest.json")
        result = json.loads(checkpoint.read_bytes())["result"]
        self.task = checkpoint.stem
        shape = tuple(result["workload"][k] for k in ("seed", "m", "k", "n"))
        data = ({self.task: shape}, {shape: result})
        # The archive loader separately verifies the Git provenance and full benchmark contract.
        mock = patch.object(evidence, "read_results", return_value=data)
        mock.start()
        self.addCleanup(mock.stop)
        self.start = evidence.utc(result["timed_start_utc"])
        self.end = self.start + timedelta(seconds=result["timing"]["wall_elapsed_s"])

    def log(self, name, samples):
        (self.logs / name).write_text("".join(json.dumps(s) + "\n" for s in samples))

    def run_export(self, logs):
        with contextlib.redirect_stdout(io.StringIO()):
            evidence.export(self.results, self.output, logs)

    def test_utc_windows_whitelist_deduplication_and_offline_regeneration(self):
        at_start = {"timestamp_utc": evidence.iso(self.start), "util_pct": 92, "device_id": "private-device"}
        duplicate = {**at_start, "power_w": 200, "hostname": "private-host"}
        at_end = {"timestamp_utc": evidence.iso(self.end), "util_pct": 67}
        self.log("gemm-1.jsonl", [at_start, at_end])
        self.log("gemm-2.jsonl", [duplicate])
        self.log("snapshot-secret.jsonl", [{"timestamp_utc": evidence.iso(self.start + timedelta(seconds=1)), "util_pct": 8}])
        self.run_export(self.logs)
        sidecar = self.output / "supplemental" / f"{self.task}.csv"
        with sidecar.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"timestamp_utc", *evidence.METRICS})
        self.assertEqual(float(rows[0]["power_w"]), 200)
        self.assertEqual(rows[0]["temperature_c"], "")
        self.assertNotIn("private", sidecar.read_text())
        index = self.output / "run-index.csv"
        originals = (index.read_bytes(), sidecar.read_bytes())
        self.run_export(None)
        self.assertEqual(originals, (index.read_bytes(), sidecar.read_bytes()))

    def test_conflicting_duplicates_leave_old_csvs_intact(self):
        sample = {"timestamp_utc": evidence.iso(self.start), "util_pct": 59}
        self.log("gemm-1.jsonl", [sample])
        self.run_export(self.logs)
        old = {f: f.read_bytes() for f in self.output.rglob("*.csv")}
        self.log("gemm-2.jsonl", [{**sample, "util_pct": 73}])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.run_export(self.logs)
        self.assertEqual(old, {f: f.read_bytes() for f in self.output.rglob("*.csv")})

    def test_power_integration_excludes_gaps_nulls_and_out_of_window(self):
        readings = [(-1, 1000), (0, 0), (2.5, 20), (5.1, 500), (6, None),
                    (6.5, 30), (7.5, 40), (9, 1000)]
        self.assertEqual(evidence.observed_energy(readings, 8), (60, 3.5))
        self.assertEqual(evidence.observed_energy([(0, 0), (1, 0)], 1), (0, 1))
        self.assertEqual(evidence.observed_energy([(0, 100), (3, 300)], 3), (None, 0))
        with self.assertRaisesRegex(ValueError, "power"):
            evidence.observed_energy([(0, -1)], 1)

    def test_all_null_supplementary_sample_breaks_power_interval(self):
        self.log("gemm-power.jsonl", [{"timestamp_utc": evidence.iso(self.start + timedelta(seconds=s)), "power_w": watts}
                                      for s, watts in [(0, 200), (1, None), (2, 200)]])
        self.run_export(self.logs)
        with (self.output / "run-index.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["supplemental_samples"], "3")
        self.assertEqual(row["supplemental_observed_energy_estimate_j"], "")
        self.assertEqual(float(row["supplemental_observed_power_duration_s"]), 0)


if __name__ == "__main__":
    unittest.main()
