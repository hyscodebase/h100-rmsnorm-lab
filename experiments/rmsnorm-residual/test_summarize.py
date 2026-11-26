"""CPU-only regression checks for the seed-wise timing gate and correctness veto."""

import copy
import unittest
from pathlib import Path

import summarize


def measured(ms):
    return {"status": "measured", "timing": {"graph_elapsed_ms": [ms] * 21}}


class ResidualDecisionTest(unittest.TestCase):
    def setUp(self):
        self.plan = summarize.protocol.FIXED
        self.cells = {}
        for rows in self.plan["rows"]:
            for seed in self.plan["seeds"]:
                p = summarize.protocol
                entries = {name: measured(1.1) for name in p.BACKENDS}
                entries[p.COMPILED[1]] = measured(1)
                entries[p.CANDIDATE] = measured(0.95)
                self.cells[seed, rows] = ({"backends": entries}, Path(f"s{seed}-r{rows}-w4096.json"))

    def test_unrounded_target_gate_for_every_seed(self):
        self.assertIn("Recommend", summarize.verdict(self.plan, self.cells))
        cell = self.cells[101, 1024][0]
        self.assertEqual(summarize.compare(cell)[1], summarize.protocol.COMPILED[1])
        cell["backends"][summarize.protocol.CANDIDATE] = measured(0.95000001)
        self.assertIn("2 of three", summarize.verdict(self.plan, self.cells))

    def test_control_performance_does_not_vote_but_wrong_output_vetoes(self):
        control = self.cells[101, 1][0]
        for key in summarize.protocol.COMPILED:
            control["backends"][key] = measured(0.1)
        self.assertIn("Recommend", summarize.verdict(self.plan, self.cells))
        del self.cells[103, 128]
        self.assertIn("Partial", summarize.verdict(self.plan, self.cells))
        control["backends"][summarize.protocol.CANDIDATE] = {
            "status": "excluded", "correctness": {"passed": False}}
        self.assertIn("correctness failed", summarize.verdict(self.plan, self.cells))
        control["backends"][summarize.protocol.CANDIDATE] = {
            "status": "excluded", "excluded_at": "verification_replay", "error": {"exception_types": ["AssertionError"]}}
        self.assertIn("correctness failed", summarize.verdict(self.plan, self.cells))

    def test_missing_oracle_or_compiled_reference(self):
        for missing in ([summarize.protocol.EAGER], summarize.protocol.COMPILED):
            cells = copy.deepcopy(self.cells)
            for name in missing:
                cells[103, 1024][0]["backends"][name] = {"status": "excluded"}
            decision = summarize.verdict(self.plan, cells)
            self.assertIn("reference is unavailable", decision)
            self.assertNotIn("Retain the compiled path", decision)


if __name__ == "__main__":
    unittest.main()
