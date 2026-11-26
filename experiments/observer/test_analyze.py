"""CPU-only tests for the observer protocol's independent, paired and partial decisions."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import analyze


def slots(valid=range(45), *, delay=0, activity=100):
    rows = []
    for i in range(45):
        present = i in valid
        rows.append({"slot": i, "status": "valid" if present else "skipped",
                     "query_start_s": i + delay if present else None,
                     "query_end_s": i + delay if present else None,
                     "values": {k: activity if k == "gpu_busy_pct" and present else None for k in analyze.METRICS}})
    return rows


def minimal(tflops, readers):
    return {"timing": {"wall_elapsed_s": 45, "wall_tflops": tflops}, "observers": readers}


class AnalysisTest(unittest.TestCase):
    def test_successful_query_deadline(self):
        rows = slots([0])
        rows[0]["query_end_s"] = 5.0
        analyze.validate_slots(rows, 45)
        rows[0]["query_end_s"] = 6.0
        with self.assertRaisesRegex(ValueError, "deadline"):
            analyze.validate_slots(rows, 45)
        rows[0]["status"] = "timeout"
        rows[0]["values"]["gpu_busy_pct"] = None
        analyze.validate_slots(rows, 45)

    def test_a_single_reader_cannot_overlap_attempted_queries(self):
        rows = slots([0, 2])
        rows[0]["status"] = "error"
        rows[0]["values"]["gpu_busy_pct"] = None
        rows[0]["query_end_s"] = 2.0
        rows.reverse()  # The slot IDs determine order, even across a skipped slot.
        analyze.validate_slots(rows, 45)
        rows[-1]["query_end_s"] = 2.1
        with self.assertRaisesRegex(ValueError, "overlap"):
            analyze.validate_slots(rows, 45)

    def test_count_gap_and_zero_boundary(self):
        sparse = set(range(45)) - set(range(1, 22, 2))
        self.assertEqual(len(sparse), 34)
        self.assertTrue(analyze.coverage(slots(sparse), 45)[2])
        self.assertFalse(analyze.coverage(slots(sparse - {44}), 45)[2])
        self.assertTrue(analyze.coverage(slots(range(5, 45)), 45)[2])
        self.assertFalse(analyze.coverage(slots(range(6, 45)), 45)[2])
        self.assertEqual(analyze.coverage(slots([]), 45), (0, 45, False))

    def test_same_slot_midpoint_tolerance_and_minimum_pairs(self):
        obs = {"cli": slots(), "nvml": slots(range(20), delay=.5, activity=95)}
        self.assertEqual(analyze.agreement(obs), (20, 25, 5))
        obs["nvml"][0]["query_start_s"] = .501
        obs["nvml"][0]["query_end_s"] = .501
        self.assertEqual(analyze.agreement(obs), (19, 26, None))
        obs["nvml"] = slots([])
        self.assertEqual(analyze.agreement(obs), (0, 45, None))

    def test_frozen_decision_boundaries(self):
        records = {}
        for i, s in enumerate(analyze.ORDER):
            records[s, "A"] = minimal(100, {"cli": slots(range(45) if i < 3 else [])})
            records[s, "B"] = minimal(98, {"nvml": slots(range(45) if i < 5 else [])})
            records[s, "C"] = minimal(90, {"cli": slots(), "nvml": slots(range(20) if i < 4 else [], activity=95)})
        self.assertEqual(analyze.decision(records)[3], (True, True, True))
        records[51, "C"]["observers"]["nvml"] = slots(range(20), activity=94.999)
        self.assertFalse(analyze.decision(records)[3][2])
        records[56, "A"]["observers"]["cli"] = slots()
        self.assertFalse(analyze.decision(records)[3][0])
        for s in analyze.ORDER:
            records[s, "B"]["timing"]["wall_tflops"] = 97.999
        self.assertFalse(analyze.decision(records)[3][1])
        del records[56, "C"]
        self.assertIsNone(analyze.decision(records))

    def test_full_contract_and_partial_report(self):
        fixture = (Path(__file__).resolve().parents[1] / "prefill-gemm/results/s44-m8192-k4096-n11008.json")
        result = json.loads(fixture.read_text())["result"]
        del result["telemetry"]
        result.update(seed=51, arm="A", observers={"cli": slots([])})
        result["environment"]["observer_versions"] = {"binding_version": None, "nvml_version": None,
            "driver_version": result["environment"]["driver"], "nvidia_smi_version": "fixture"}
        result["workload"]["seed"] = 51
        analyze.validate(result, 51, "A")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(analyze.load(folder), {})
            (folder / "s51-A.json").write_text(json.dumps(result))
            partial = analyze.load(folder)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                analyze.show(partial)
            self.assertIn("Decision deferred", output.getvalue())
            self.assertIn("1/18", output.getvalue())
        for part in ("duplicates", "out-of-window", "missing metric", "bad throughput", "failed check"):
            with self.subTest(part=part):
                modified = copy.deepcopy(result)
                r = modified["observers"]["cli"]
                if part == "duplicates":
                    r[1]["slot"] = 0
                elif part == "out-of-window":
                    r[1] = slots()[1]
                    r[1]["query_end_s"] = 100
                elif part == "missing metric":
                    r[1]["values"].pop("sm_clock_mhz")
                elif part == "bad throughput":
                    modified["timing"]["wall_tflops"] *= 1.01
                else:
                    modified["correctness"]["passed"] = False
                with self.assertRaises(ValueError):
                    analyze.validate(modified, 51, "A")


if __name__ == "__main__":
    unittest.main()
