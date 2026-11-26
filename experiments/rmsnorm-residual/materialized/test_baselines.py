"""CPU regression tests for finite BF16 encodings and contender selection."""
import unittest

import torch

import campaign
from baselines import bitcast_reference, normalize_updated, round_float32, rounded_add
import check


class BaselineTests(unittest.TestCase):
    def test_midpoints_and_signed_zero_rounding(self):
        words, expected = campaign.protocol.midpoint_bits()
        values = torch.tensor(words, dtype=torch.int32).view(torch.float32)
        encoded, promoted = round_float32(values)
        self.assertEqual(campaign.raw_bf16(encoded), expected)
        self.assertEqual(campaign.raw_bf16(values.bfloat16()), expected)
        self.assertTrue(torch.equal(promoted.view(torch.int32), encoded.float().view(torch.int32)))

    def test_reference_including_denominator_tie(self):
        def materialized(x, residual, weight):
            rounded = rounded_add(x, residual)
            return rounded, normalize_updated(rounded, weight)

        for reference in (bitcast_reference, materialized):
            result = check.validate_backend(reference, 4, 105, "cpu")
            self.assertTrue(result["passed"], result["failure"])
            self.assertEqual(campaign.signed_zero_check(reference, "cpu"),
                             {"updated_bits": [32768, 0], "normalized_bits": [32768, 0]})

    def test_comparison_uses_fastest_correct_single_graph(self):
        plan = campaign.protocol
        timings = {plan.EAGER: 100, plan.TRITON: 10, "compiled_materialized_default": 14,
                   "compiled_bitcast_default": 10.4}
        self.assertFalse(plan.comparison(timings)["passes_5pct"])
        self.assertEqual(plan.comparison(timings)["fastest_correct_compiled"], "compiled_bitcast_default")
        del timings["compiled_bitcast_default"]
        self.assertFalse(plan.gate(timings))
        self.assertIsNone(plan.comparison(timings)["latency_reduction_pct"])


if __name__ == "__main__":
    unittest.main()
